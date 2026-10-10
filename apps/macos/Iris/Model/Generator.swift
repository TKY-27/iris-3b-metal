import Foundation
import Observation

/// The native Iris-3B resolutions: the aspect buckets of its 1024-pixel
/// training stages (`src/iris3b/data/buckets.py`), about one megapixel each.
struct ImageSize: Hashable, Identifiable, Sendable {
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

@MainActor
@Observable
final class Generator {
    enum Phase: Equatable {
        case idle
        case waiting  // submitted, queued behind another request or loading the model
        case sampling(step: Int, of: Int, secondsLeft: Double?)
        case failed(String)
    }

    var prompt = ""
    var negativePrompt = ""
    var size = ImageSize.square
    var steps = 100
    var cfgScale = 3.0
    var seed = 0
    var randomizeSeed = true

    private(set) var phase: Phase = .idle
    private var job: String?

    var isBusy: Bool {
        switch phase {
        case .waiting, .sampling: true
        case .idle, .failed: false
        }
    }

    var canGenerate: Bool { !isBusy && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    func reuse(_ request: GenerationRequest) {
        prompt = request.prompt
        negativePrompt = request.negativePrompt
        size = ImageSize(width: request.width, height: request.height)
        steps = request.steps
        cfgScale = request.cfgScale
        seed = request.seed
        randomizeSeed = false
    }

    /// Runs one request to completion; returns the saved item, or nil if cancelled or failed.
    func generate(engine: EngineController, history: History) async -> HistoryItem? {
        guard canGenerate else { return nil }
        if randomizeSeed {
            seed = Int.random(in: 0..<Int(Int32.max))
        }
        let request = GenerationRequest(
            prompt: prompt, width: size.width, height: size.height, steps: steps,
            cfgScale: cfgScale, seed: seed, negativePrompt: negativePrompt
        )
        phase = .waiting
        let client = engine.client
        do {
            let id = try await client.submit(request)
            job = id
            defer { job = nil }
            let status = try await poll(id, client: client)
            switch status.status {
            case .done:
                let png = try await client.image(of: id)
                let item = try history.add(png: png, request: request, seconds: status.seconds ?? 0)
                phase = .idle
                return item
            case .cancelled:
                phase = .idle
            default:
                phase = .failed(status.error ?? "The engine could not generate this image.")
            }
        } catch EngineError.unreachable {
            phase = .idle
            engine.connectionLost()
        } catch {
            phase = .failed(error.localizedDescription)
        }
        return nil
    }

    func cancel(engine: EngineController) {
        guard let job else { return }
        Task { try? await engine.client.cancel(job) }
    }

    func dismissError() {
        if case .failed = phase { phase = .idle }
    }

    private func poll(_ id: String, client: EngineClient) async throws -> JobStatus {
        var firstSample: (step: Int, time: ContinuousClock.Instant)?
        while true {
            let status = try await client.status(of: id)
            switch status.status {
            case .queued:
                phase = .waiting
            case .running where status.step == 0:
                phase = .waiting  // first request loads the model before step 1
            case .running:
                let now = ContinuousClock.now
                if firstSample == nil { firstSample = (status.step, now) }
                var secondsLeft: Double?
                if let first = firstSample, status.step > first.step {
                    let perStep = (now - first.time) / Double(status.step - first.step)
                    secondsLeft = Double((perStep * Double(status.steps - status.step)).components.seconds)
                }
                phase = .sampling(step: status.step, of: status.steps, secondsLeft: secondsLeft)
            case .done, .failed, .cancelled:
                return status
            }
            try await Task.sleep(for: .milliseconds(250))
        }
    }
}
