import Metal
import MetalKit
import XCTest

@testable import Iris

@MainActor
final class CanvasTests: XCTestCase {
    func testBackgroundKeepsDrawingWhileEngineIsBusy() async throws {
        let pipeline = try XCTUnwrap(FieldPipeline.shared)
        let view = FieldMTKView(frame: NSRect(x: 0, y: 0, width: 128, height: 128), device: pipeline.device)
        let renderer = FieldRenderer(pointer: FieldPointer())
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm
        let window = NSWindow(contentRect: view.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = view
        window.orderFrontRegardless()
        defer { view.stop(); window.orderOut(nil) }
        // Window-server visibility arrives after ordering the test window.
        let deadline = Date().addingTimeInterval(2)
        while !window.occlusionState.contains(.visible), Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(window.occlusionState.contains(.visible))
        func draw() throws {
            renderer.draw(in: view)
            let completed = try XCTUnwrap(pipeline.queue.makeCommandBuffer())
            completed.commit()
            completed.waitUntilCompleted()
            XCTAssertEqual(completed.status, .completed)
        }

        renderer.lastInteraction = CACurrentMediaTime() - 5
        renderer.engineBusy = true
        view.isPaused = false
        try draw()
        XCTAssertFalse(view.isPaused)

        renderer.engineBusy = false
        try draw()
        XCTAssertTrue(view.isPaused)

        renderer.engineBusy = true
        renderer.animated = false
        view.isPaused = false
        try draw()
        XCTAssertTrue(view.isPaused)
    }

    /// The canvas shader is compiled at runtime, so the build cannot catch a mistake in it.
    func testCanvasShaderCompiles() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        XCTAssertNoThrow(try FieldPipeline.compile(on: device))
    }

    func testPointerChangesTheRenderedField() throws {
        let left = try render(pointer: SIMD2(0.25, 0.5), touch: 1)
        let right = try render(pointer: SIMD2(0.75, 0.5), touch: 1)
        XCTAssertGreaterThan(zip(left, right).filter { $0 != $1 }.count, 256)
    }

    func testStillFieldIgnoresPointerAndKeepsOpaqueTile() throws {
        let left = try render(pointer: SIMD2(0.25, 0.5), touch: 0)
        let right = try render(pointer: SIMD2(0.75, 0.5), touch: 0)
        XCTAssertEqual(left, right)
        XCTAssertTrue(stride(from: 3, to: left.count, by: 4).allSatisfy { left[$0] == 255 })
    }

    private func render(pointer: SIMD2<Float>, touch: Float) throws -> [UInt8] {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let pipeline = try FieldPipeline.compile(on: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: 128, height: 128, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget]
        let texture = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let commands = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commands.makeRenderCommandEncoder(descriptor: pass))
        var uniforms = [SIMD4<Float>(128, 128, 12, 1), SIMD4(0.5, 1, touch, 0.5), SIMD4(0.5, 0.5, 0, 1)]
        uniforms += Array(repeating: SIMD4(pointer.x, pointer.y, pointer.x, pointer.y), count: 4)
        encoder.setRenderPipelineState(pipeline)
        uniforms.withUnsafeBytes {
            encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        XCTAssertEqual(commands.status, .completed)
        var pixels = [UInt8](repeating: 0, count: 128 * 128 * 4)
        pixels.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: 128 * 4,
                             from: MTLRegionMake2D(0, 0, 128, 128), mipmapLevel: 0)
        }
        return pixels
    }
}
