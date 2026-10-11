import CoreGraphics
import Foundation

/// What a conversation entry asks the engine to do.
enum Workflow: String, Codable, CaseIterable, Identifiable, Sendable {
    case create, depth, upscale

    var id: Self { self }

    var title: String {
        switch self {
        case .create: "Create Image"
        case .depth: "Estimate Depth"
        case .upscale: "Restore and Upscale"
        }
    }

    var symbol: String {
        switch self {
        case .create: "photo"
        case .depth: "square.3.layers.3d.down.right"
        case .upscale: "wand.and.rays"
        }
    }

    /// Depth and upscaling take an image instead of a prompt.
    var needsImage: Bool { self != .create }

    /// What the composer offers for an attached image. Restoration comes only with upscaling:
    /// the model was trained on 4× enlargements, and at the same size it redraws fine structure
    /// (`scripts/restore_study.py`).
    static let imageWorkflows: [Workflow] = [.upscale, .depth]
}

/// The size the engine's restorer makes of an image. Like `iris-metal serve` by default, it
/// first reduces an input larger than 512 pixels on the short side or 1024 on the long side
/// (upstream `fit_budget`), then enlarges it four times.
enum Upscaling {
    static let factor = 4
    /// The widest input the engine accepts (`check_upscale_input`).
    static let maxAspect = 4

    /// The copy the restorer works on: the input, or a reduced copy of a large one.
    static func workingSize(for input: CGSize) -> CGSize {
        let (width, height) = (input.width, input.height)
        let scale = min(1, 512 / min(width, height), 1024 / max(width, height))
        guard scale < 1 else { return input }
        // Python's round(), as fit_budget uses
        func side(_ length: CGFloat) -> CGFloat { max(1, (length * scale).rounded(.toNearestOrEven)) }
        return CGSize(width: side(width), height: side(height))
    }

    static func output(for input: CGSize) -> CGSize {
        let working = workingSize(for: input)
        return CGSize(width: working.width * CGFloat(factor), height: working.height * CGFloat(factor))
    }

    static func accepts(_ input: CGSize) -> Bool {
        min(input.width, input.height) > 0 && max(input.width, input.height) <= CGFloat(maxAspect) * min(input.width, input.height)
    }

    /// Why an image is not upscaled, or nil when it can be: the engine refuses a long, thin one,
    /// and one larger than the result would only come out smaller.
    static func refusal(for input: CGSize) -> String? {
        guard accepts(input) else {
            return "Too long and thin to upscale: the long side may be at most \(maxAspect) times the short side (this is \(input.pixelLabel))."
        }
        guard output(for: input).width > input.width else {
            return "Already larger than an upscale (\(output(for: input).pixelLabel)): the restorer works on a copy reduced to \(workingSize(for: input).pixelLabel)."
        }
        return nil
    }
}

extension CGSize {
    /// "2048 × 1536"
    var pixelLabel: String { "\(Int(width)) × \(Int(height))" }
}

extension String {
    /// The caption as VoiceOver should say it: "896 by 1152 to 2048 by 2632, 43 sec".
    var spokenSizes: String {
        replacingOccurrences(of: " × ", with: " by ")
            .replacingOccurrences(of: " → ", with: " to ")
            .replacingOccurrences(of: " · ", with: ", ")
    }
}

/// The native Iris-3B resolutions: the aspect buckets of its 1024-pixel
/// training stages (`src/iris3b/data/buckets.py`), about one megapixel each.
struct ImageSize: Hashable, Identifiable, Codable, Sendable {
    let width: Int
    let height: Int

    var id: String { "\(width)x\(height)" }
    var label: String { "\(width) × \(height)" }
    var aspectRatio: Double { Double(width) / Double(height) }

    static let square = ImageSize(width: 1024, height: 1024)
    static let landscape: [ImageSize] = [
        (1088, 960), (1152, 896), (1280, 832), (1344, 768), (1472, 704),
        (1600, 640), (1664, 640), (1792, 576), (1856, 576), (2048, 512),
    ].map { ImageSize(width: $0.0, height: $0.1) }
    static let portrait = landscape.map { ImageSize(width: $0.height, height: $0.width) }

    /// A short ratio such as "3:2"; "≈" marks the nearest common ratio when the exact one is unwieldy.
    var ratioLabel: String {
        let divisor = Self.gcd(width, height)
        let (w, h) = (width / divisor, height / divisor)
        if max(w, h) <= 16 { return "\(w):\(h)" }
        let wide = max(aspectRatio, 1 / aspectRatio)
        let nearest = Self.commonRatios.min { abs($0.value - wide) < abs($1.value - wide) }!
        let (a, b) = (nearest.w, nearest.h)
        return width >= height ? "≈\(a):\(b)" : "≈\(b):\(a)"
    }

    private static let commonRatios: [(w: Int, h: Int, value: Double)] = [
        (1, 1), (9, 8), (5, 4), (4, 3), (3, 2), (16, 9), (2, 1), (5, 2), (3, 1), (4, 1),
    ].map { ($0.0, $0.1, Double($0.0) / Double($0.1)) }

    private static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
}

/// Five step counts named by what they trade; the number stays one click away.
/// Draft (4 steps) shows only the composition; Standard is the app's default.
enum QualityPreset: Int, CaseIterable, Identifiable, Sendable {
    case draft = 4, fast = 10, standard = 20, high = 30, maximum = 50

    var id: Int { rawValue }
    var steps: Int { rawValue }

    var title: String {
        switch self {
        case .draft: "Draft"
        case .fast: "Fast"
        case .standard: "Standard"
        case .high: "High"
        case .maximum: "Maximum"
        }
    }

    /// The preset for `steps`, or nil for a number typed in by hand.
    init?(steps: Int) { self.init(rawValue: steps) }

    /// The nearest preset, for placing the slider when a custom number is set.
    static func nearest(to steps: Int) -> QualityPreset {
        allCases.min { abs($0.steps - steps) < abs($1.steps - steps) }!
    }
}

/// How closely an image follows its prompt (classifier-free guidance), in words.
enum AdherencePreset: Int, CaseIterable, Identifiable, Sendable {
    case creative, loose, balanced, close, literal

    var id: Int { rawValue }

    var cfgScale: Double {
        switch self {
        case .creative: 1.5
        case .loose: 2
        case .balanced: 3  // the checkpoint's default
        case .close: 4
        case .literal: 5
        }
    }

    var title: String {
        switch self {
        case .creative: "Creative"
        case .loose: "Loose"
        case .balanced: "Balanced"
        case .close: "Close"
        case .literal: "Literal"
        }
    }

    init?(cfgScale: Double) {
        guard let match = Self.allCases.first(where: { $0.cfgScale == cfgScale }) else { return nil }
        self = match
    }

    static func nearest(to cfgScale: Double) -> AdherencePreset {
        allCases.min { abs($0.cfgScale - cfgScale) < abs($1.cfgScale - cfgScale) }!
    }
}

/// Text-to-image settings for the next prompt; kept between launches.
struct GenerationSettings: Codable, Equatable, Sendable {
    static let stepRange = 1...500  // the API's limit
    static let guidanceRange = 1.0...10.0
    static let seedRange = 0...Int(Int32.max)
    /// What Iris-3B was evaluated at; the app starts faster and says so.
    static let evaluatedSteps = 100

    var size = ImageSize.square
    var steps = QualityPreset.standard.steps
    var cfgScale = AdherencePreset.balanced.cfgScale
    var seed = 0
    var randomizeSeed = true
    var negativePrompt = ""

    var quality: QualityPreset? { QualityPreset(steps: steps) }
    var adherence: AdherencePreset? { AdherencePreset(cfgScale: cfgScale) }

    /// "Standard", or "48 steps" for a number typed in by hand.
    var qualityTitle: String { quality?.title ?? "\(steps) steps" }
    var adherenceTitle: String { adherence?.title ?? "Guidance \(cfgScale.formatted(.number.precision(.fractionLength(0...2))))" }

    /// `count` requests for `prompt`: each with a new seed when randomizing, otherwise the
    /// seed and the ones after it, so a batch can be made again exactly.
    mutating func requests(for prompt: String, count: Int) -> [GenerationRequest] {
        let first = request(for: prompt)
        return (0..<max(count, 1)).map { offset in
            var request = first
            if offset > 0 {
                request.seed = randomizeSeed ? Int.random(in: Self.seedRange) : (first.seed + offset) % (Self.seedRange.upperBound + 1)
            }
            return request
        }
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

    /// Quality, adherence, seed and negative prompt back to their defaults; the size stays.
    mutating func restoreDefaults() {
        self = GenerationSettings(size: size)
    }

    /// Whether `restoreDefaults()` would change anything (a drawn seed does not count).
    var hasDefaultTuning: Bool {
        var defaults = GenerationSettings(size: size)
        if randomizeSeed { defaults.seed = seed }
        return self == defaults
    }

    mutating func reuse(_ request: GenerationRequest) {
        size = ImageSize(width: request.width, height: request.height)
        steps = request.steps
        cfgScale = request.cfgScale
        seed = request.seed
        randomizeSeed = false
        negativePrompt = request.negativePrompt
    }

    // MARK: - Persistence

    private static let defaultsKey = "generationSettings.v2"
    /// Iris 0.2 saved its settings, including its 100-step default, here.
    private static let legacyKey = "generationSettings"

    static func load(from defaults: UserDefaults = .standard) -> GenerationSettings {
        if let data = defaults.data(forKey: defaultsKey) {
            return (try? JSONDecoder().decode(GenerationSettings.self, from: data)) ?? GenerationSettings()
        }
        guard let data = defaults.data(forKey: legacyKey),
              var legacy = try? JSONDecoder().decode(GenerationSettings.self, from: data)
        else { return GenerationSettings() }
        // the old default was always saved; only a deliberate choice of steps survives
        if legacy.steps == evaluatedSteps { legacy.steps = GenerationSettings().steps }
        return legacy
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
