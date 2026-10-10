import MetalKit
import SwiftUI
import os

/// A faint glow behind a view that the pointer stirs like liquid; while an image is made, an iris of
/// fibers whose aperture opens with the engine's progress.
///
/// Rendered by a small Metal shader at half resolution, so it costs little next to the engine; with
/// Reduce Motion (or the setting off) it is a single still frame that ignores the pointer.
struct IrisCanvas<Content: View>: View {
    /// 0 idle, 1 generating.
    var energy: Double = 0
    /// Sampling progress in 0...1, which opens the aperture; nil when unknown.
    var progress: Double?
    /// Draws on its own dark base, as the generation canvas does, instead of over the window.
    var opaque = false
    /// Where the iris sits, in unit coordinates of the view.
    var center = UnitPoint.center
    @ViewBuilder var content: Content

    @State private var pointer = FieldPointer()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorScheme) private var colorScheme
    @AppStorage(FieldSettings.animatesKey) private var animates = true
    @Environment(Generator.self) private var generator

    var body: some View {
        content
            .background {
                IrisFieldView(
                    parameters: FieldParameters(
                        energy: Float(energy), progress: progress.map(Float.init) ?? -1,
                        dark: colorScheme == .dark || opaque, opaque: opaque,
                        center: SIMD2(Float(center.x), Float(center.y))
                    ),
                    animated: animates && !reduceMotion,
                    engineBusy: generator.isBusy,
                    pointer: pointer
                )
                .accessibilityHidden(true)
            }
            .onGeometryChange(for: CGSize.self, of: \.size) { pointer.size = $0 }
            .onContinuousHover { phase in
                switch phase {
                case let .active(location): pointer.location = location
                case .ended: pointer.location = nil
                }
            }
    }
}

enum FieldSettings {
    static let animatesKey = "animatesCanvas"
}

/// The pointer, outside SwiftUI's observation so that moving it redraws only the shader.
@MainActor
final class FieldPointer {
    var location: CGPoint?
    var size = CGSize(width: 1, height: 1)

    /// The pointer in unit coordinates, origin top left.
    var unitLocation: SIMD2<Float>? {
        guard let location, size.width > 0, size.height > 0 else { return nil }
        return SIMD2(Float(location.x / size.width), Float(location.y / size.height))
    }
}

struct FieldParameters: Equatable {
    var energy: Float = 0
    var progress: Float = -1
    var dark = true
    var opaque = false
    var center = SIMD2<Float>(0.5, 0.5)
}

private struct IrisFieldView: NSViewRepresentable {
    let parameters: FieldParameters
    let animated: Bool
    let engineBusy: Bool
    let pointer: FieldPointer

    func makeCoordinator() -> FieldRenderer {
        FieldRenderer(pointer: pointer)
    }

    func makeNSView(context: Context) -> NSView {
        guard let pipeline = FieldPipeline.shared else { return NSView() }
        let view = FieldMTKView(frame: .zero, device: pipeline.device)
        view.colorPixelFormat = .bgra8Unorm
        view.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.layer?.isOpaque = false
        view.framebufferOnly = true
        view.autoResizeDrawable = false
        view.delegate = context.coordinator
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        guard let view = view as? FieldMTKView else { return }
        let renderer = context.coordinator
        renderer.target = parameters
        renderer.animated = animated
        // leave the GPU to the engine while it works, whichever chat shows
        view.preferredFramesPerSecond = engineBusy ? 30 : 60
        view.isPaused = !animated
        view.enableSetNeedsDisplay = !animated
        if !animated { view.needsDisplay = true }
    }

    static func dismantleNSView(_ view: NSView, coordinator: FieldRenderer) {
        (view as? MTKView)?.isPaused = true
        (view as? MTKView)?.delegate = nil
    }
}

/// Draws at half the backing resolution (the field is soft) and lets clicks through.
private final class FieldMTKView: MTKView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func layout() {
        super.layout()
        resizeDrawable()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        resizeDrawable()
    }

    private func resizeDrawable() {
        let scale = (window?.backingScaleFactor ?? 2) * 0.5
        drawableSize = CGSize(width: max(1, (bounds.width * scale).rounded()), height: max(1, (bounds.height * scale).rounded()))
        if enableSetNeedsDisplay { needsDisplay = true }
    }
}

@MainActor
private final class FieldRenderer: NSObject, MTKViewDelegate {
    var target = FieldParameters()
    var animated = true

    private let pointer: FieldPointer
    private var current = FieldParameters()
    private var trail = [SIMD2<Float>](repeating: SIMD2(0.5, 0.5), count: 8)
    private var touch: Float = 0
    private var motion: Float = 0
    private var wasInside = false
    private var lastFrame: CFTimeInterval?
    private let start = CACurrentMediaTime()

    init(pointer: FieldPointer) {
        self.pointer = pointer
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pipeline = FieldPipeline.shared,
              !animated || view.window?.occlusionState.contains(.visible) == true,  // a still frame is drawn once
              let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commands = pipeline.queue.makeCommandBuffer(),
              let encoder = commands.makeRenderCommandEncoder(descriptor: pass)
        else { return }

        let now = CACurrentMediaTime()
        let time: Float
        if animated {
            step(Float(min(max(now - (lastFrame ?? now), 1.0 / 240), 1.0 / 20)))
            time = Float((now - start).truncatingRemainder(dividingBy: 10_000))
        } else {
            current = target
            touch = 0
            time = 12  // a settled moment of the flow
        }
        lastFrame = now

        let size = view.drawableSize
        var uniforms: [SIMD4<Float>] = [
            SIMD4(Float(size.width), Float(size.height), time, current.energy),
            SIMD4(current.progress, current.dark ? 1 : 0, touch, motion),
            SIMD4(current.center.x, current.center.y, 0, current.opaque ? 1 : 0),
        ]
        for i in stride(from: 0, to: trail.count, by: 2) {
            uniforms.append(SIMD4(trail[i].x, trail[i].y, trail[i + 1].x, trail[i + 1].y))
        }
        encoder.setRenderPipelineState(pipeline.state)
        uniforms.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            encoder.setFragmentBytes(base, length: bytes.count, index: 0)
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.present(drawable)
        commands.commit()
    }

    /// Eases toward the requested state and lets the trail follow the pointer like a liquid.
    private func step(_ dt: Float) {
        func approach(_ rate: Float) -> Float { 1 - exp(-dt * rate) }
        current.energy += (target.energy - current.energy) * approach(1.6)
        if target.progress < 0 {
            current.progress = -1
        } else {
            current.progress = current.progress < 0 ? target.progress : current.progress + (target.progress - current.progress) * approach(3)
        }
        current.dark = target.dark
        current.opaque = target.opaque
        current.center = target.center

        if let location = pointer.unitLocation {
            if !wasInside { trail = Array(repeating: location, count: trail.count) }
            trail[0] += (location - trail[0]) * approach(22)
            touch += (1 - touch) * approach(4)
        } else {
            touch -= touch * approach(2.5)
        }
        wasInside = pointer.unitLocation != nil
        for i in 1..<trail.count {
            trail[i] += (trail[i - 1] - trail[i]) * approach(13)
        }
        let stretch = simd_length(trail[0] - trail[trail.count - 1])
        motion += (min(stretch * 8, 1) - motion) * approach(6)
    }
}

/// The compiled shader, shared by every canvas. Compiled from source at first use, so
/// building the app needs no Metal toolchain.
/// A unit test compiles the source too, since the build cannot.
@MainActor
struct FieldPipeline {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let state: MTLRenderPipelineState

    static let shared: FieldPipeline? = {
        let logger = Logger(subsystem: "io.github.tky27.iris", category: "Canvas")
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            logger.error("No Metal device; the canvas stays empty")
            return nil
        }
        do {
            return FieldPipeline(device: device, queue: queue, state: try compile(on: device))
        } catch {
            logger.error("Canvas shader failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }()

    static func compile(on device: MTLDevice) throws -> MTLRenderPipelineState {
        let library = try device.makeLibrary(source: source, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "field_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "field_fragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }
}

private let source = """
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 position [[position]]; float2 uv; };

vertex VOut field_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    VOut out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(p.x, 1.0 - p.y);  // origin top left, as SwiftUI's
    return out;
}

static float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

static float vnoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    float2 w = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + float2(1, 0)), w.x),
               mix(hash21(i + float2(0, 1)), hash21(i + float2(1, 1)), w.x), w.y);
}

static float fbm(float2 p) {
    float sum = 0.0, amplitude = 0.5;
    for (int i = 0; i < 4; i++) {
        sum += amplitude * vnoise(p);
        p = p * 2.03 + float2(1.7, 9.2);
        amplitude *= 0.5;
    }
    return sum;
}

// u[0] size.xy, time, energy; u[1] progress, dark, touch, motion; u[2] center.xy, -, opaque;
// u[3...6] the pointer trail, two points each
fragment float4 field_fragment(VOut in [[stage_in]], constant float4 *u [[buffer(0)]]) {
    float2 size = u[0].xy;
    float time = u[0].z, energy = u[0].w;
    float progress = u[1].x, dark = u[1].y, touch = u[1].z, motion = u[1].w;
    float2 center = u[2].xy;
    float opaque = u[2].w;
    float2 aspect = size / min(size.x, size.y);
    float2 p = (in.uv - center) * aspect;

    // the pointer trail as a liquid lens that pulls the field along
    float2 pull = float2(0.0);
    float wet = 0.0;
    for (int i = 0; i < 8; i++) {
        float4 pair = u[3 + i / 2];
        float2 c = ((i & 1) == 0 ? pair.xy : pair.zw);
        float2 d = p - (c - center) * aspect;
        float r = 0.11 - 0.008 * float(i);
        float w = exp(-dot(d, d) / (r * r)) * (1.0 - float(i) / 9.0);
        pull += d * w;
        wet += w;
    }
    // a resting pointer leaves a faint lens; a moving one, a visible liquid trail
    float strength = touch * (0.4 + 0.6 * motion);
    pull *= strength / 4.0;
    wet = saturate(wet / 4.0) * strength;
    float2 q = p - pull * 1.6;

    float rad = max(length(q), 1e-4);
    float2 dir = q / rad;
    float theta = atan2(dir.y, dir.x);
    float flow = time * (0.025 + 0.11 * energy);

    // fibers vary fast around the ring and slowly along it
    float fibers = fbm(dir * 6.5 + float2(rad * 1.7 - flow, flow * 0.6));
    fibers = fbm(dir * 4.5 + fibers * 1.9 + float2(-flow * 0.5, rad * 1.3));

    // the aperture opens as sampling progresses, leaving room for the readout
    float opening = progress < 0.0 ? 0.2 : progress;
    float pupil = mix(0.1, 0.18, opening);
    float outer = 0.37 * (1.0 + 0.015 * sin(time * 0.9) * energy);
    float ring = smoothstep(pupil - 0.012, pupil + 0.04, rad) * (1.0 - smoothstep(outer * 0.6, outer * 1.25, rad));
    float iris = ring * (0.25 + 1.25 * fibers * fibers);
    float rim = exp(-pow((rad - pupil - 0.018) / 0.02, 2.0)) * energy;
    float halo = exp(-pow((rad - 0.34) / 0.2, 2.0)) * (0.35 + fbm(q * 2.2 + float2(flow, -flow)));
    float shape = mix(halo * 0.42, iris, energy);

    float3 indigo = float3(0.36, 0.31, 1.0), violet = float3(0.62, 0.40, 1.0);
    float3 azure = float3(0.22, 0.70, 1.0), ember = float3(1.0, 0.66, 0.40);
    float a1 = 0.5 + 0.5 * sin(theta * 2.0 + flow * 2.0 + fibers * 3.0);
    float a2 = 0.5 + 0.5 * sin(theta * 3.0 - flow * 1.3);
    float3 color = mix(mix(indigo, azure, a1), violet, a2 * 0.6);
    color = mix(color, ember, saturate(rim * 0.85));

    // a spectral sheen where the liquid lies, and a thin bright edge around it
    float edge = pow(4.0 * wet * (1.0 - wet), 4.0);
    color += wet * 0.3 * mix(azure, violet, a2) + edge * 0.25;
    float intensity = shape * (0.4 + 0.8 * energy) + rim * 0.9 + wet * 0.12 + edge * (0.1 + 0.2 * motion);

    // contained: the field fades before the edges of its view
    float2 e = abs(in.uv - 0.5);
    float fade = (1.0 - smoothstep(0.3, 0.5, e.x)) * (1.0 - smoothstep(0.3, 0.5, e.y));
    float alpha = saturate(intensity) * (opaque > 0.5 ? 1.0 : fade);
    alpha = saturate(alpha + (hash21(in.uv * size + time) - 0.5) / 255.0);  // dither against banding

    if (opaque > 0.5) {
        float3 base = float3(0.035, 0.035, 0.05);
        return float4(mix(base, color, alpha), 1.0);
    }
    if (dark < 0.5) {  // over a light window: deeper hues, less light
        color *= 0.8;
        alpha *= 0.55;
    }
    return float4(color * alpha, alpha);
}
"""
