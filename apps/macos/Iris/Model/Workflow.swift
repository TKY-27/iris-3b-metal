import Foundation

/// What a conversation entry asks the engine to do.
enum Workflow: String, Codable, CaseIterable, Identifiable, Sendable {
    case create, depth, upscale

    var id: Self { self }

    var title: String {
        switch self {
        case .create: "Create Image"
        case .depth: "Estimate Depth"
        case .upscale: "Upscale ×4"
        }
    }

    var symbol: String {
        switch self {
        case .create: "photo"
        case .depth: "square.3.layers.3d.down.right"
        case .upscale: "arrow.up.left.and.arrow.down.right"
        }
    }

    /// Depth and upscaling take an image instead of a prompt.
    var needsImage: Bool { self != .create }
}

/// The native Iris-3B resolutions: the aspect buckets of its 1024-pixel
/// training stages (`src/iris3b/data/buckets.py`), about one megapixel each.
struct ImageSize: Hashable, Identifiable, Codable, Sendable {
    let width: Int
    let height: Int

    var id: String { "\(width)x\(height)" }
    var label: String { "\(width) × \(height)" }

    static let square = ImageSize(width: 1024, height: 1024)
    static let landscape: [ImageSize] = [
        (1088, 960), (1152, 896), (1280, 832), (1344, 768), (1472, 704),
        (1600, 640), (1664, 640), (1792, 576), (1856, 576), (2048, 512),
    ].map { ImageSize(width: $0.0, height: $0.1) }
    static let portrait = landscape.map { ImageSize(width: $0.height, height: $0.width) }
}

/// Text-to-image settings for the next prompt; kept between launches.
struct GenerationSettings: Codable, Equatable, Sendable {
    static let stepRange = 1...500
    static let guidanceRange = 1.0...10.0
    static let seedRange = 0...Int(Int32.max)

    var size = ImageSize.square
    var steps = 100
    var cfgScale = 3.0
    var seed = 0
    var randomizeSeed = true
    var negativePrompt = ""

    /// Upstream's evaluated defaults: 100 steps at guidance 3.
    var isDefaultQuality: Bool { steps == 100 && cfgScale == 3 }

    var summary: String {
        "\(size.label) · \(steps) steps · guidance \(cfgScale.formatted())"
    }

    /// The request for `prompt`; draws a new seed first when asked to.
    mutating func request(for prompt: String) -> GenerationRequest {
        if randomizeSeed { seed = Int.random(in: Self.seedRange) }
        return GenerationRequest(
            prompt: prompt, width: size.width, height: size.height,
            steps: steps.clamped(to: Self.stepRange),
            cfgScale: cfgScale.clamped(to: Self.guidanceRange),
            seed: seed.clamped(to: Self.seedRange),
            negativePrompt: negativePrompt
        )
    }

    mutating func reuse(_ request: GenerationRequest) {
        size = ImageSize(width: request.width, height: request.height)
        steps = request.steps
        cfgScale = request.cfgScale
        seed = request.seed
        randomizeSeed = false
        negativePrompt = request.negativePrompt
    }

    private static let defaultsKey = "generationSettings"

    static func load(from defaults: UserDefaults = .standard) -> GenerationSettings {
        guard let data = defaults.data(forKey: defaultsKey) else { return GenerationSettings() }
        return (try? JSONDecoder().decode(GenerationSettings.self, from: data)) ?? GenerationSettings()
    }

    func save(to defaults: UserDefaults = .standard) {
        defaults.set(try? JSONEncoder().encode(self), forKey: Self.defaultsKey)
    }
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
