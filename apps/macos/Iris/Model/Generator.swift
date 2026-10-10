import Foundation
import Observation

/// Runs one library entry at a time on the engine and records its outcome.
@MainActor
@Observable
final class Generator {
    struct Progress: Equatable {
        /// 0 while queued or loading the model.
        var step = 0
        var steps = 0
        var secondsLeft: Double?
    }

    /// The entry being run and its conversation.
    private(set) var active: (entry: UUID, conversation: UUID)?
    private(set) var progress = Progress()

    @ObservationIgnored private var job: String?
    @ObservationIgnored private var cancelRequested = false
    @ObservationIgnored private var task: Task<Void, Never>?

    var isBusy: Bool { active != nil }

    /// Adds a text-to-image entry to `conversation` (a new one when nil) and runs it.
    @discardableResult
    func generate(
        _ request: GenerationRequest, in conversation: UUID?, project: UUID? = nil,
        library: Library, engine: EngineController
    ) -> UUID? {
        guard !isBusy, !library.isReadOnly else { return nil }
        let entry = Entry(workflow: .create, request: request)
        let id = library.append(entry, to: conversation, project: project)
        start(entry, in: id, input: nil, library: library, engine: engine)
        return id
    }

    /// Adds a depth or upscaling entry for an image file or an earlier result and runs it.
    @discardableResult
    func process(
        _ workflow: Workflow, image: URL, named name: String, in conversation: UUID?, project: UUID? = nil,
        library: Library, engine: EngineController
    ) -> UUID? {
        guard !isBusy, !library.isReadOnly, workflow.needsImage else { return nil }
        let entry = Entry(workflow: workflow, inputName: name)
        let id = library.append(entry, to: conversation, project: project)
        start(entry, in: id, input: image, library: library, engine: engine)
        return id
    }

    /// Runs a finished, failed or cancelled entry's request again as a new entry.
    func rerun(_ entry: Entry, newSeed: Bool, in conversation: UUID, library: Library, engine: EngineController) {
        if var request = entry.request {
            if newSeed { request.seed = Int.random(in: GenerationSettings.seedRange) }
            generate(request, in: conversation, library: library, engine: engine)
        } else if FileManager.default.fileExists(atPath: library.inputURL(entry.id).path) {
            process(
                entry.workflow, image: library.inputURL(entry.id), named: entry.inputName ?? "image",
                in: conversation, library: library, engine: engine
            )
        }
    }

    func cancel(engine: EngineController) {
        guard isBusy else { return }
        cancelRequested = true
        if let job {
            Task { try? await engine.client.cancel(job) }
        }
    }

    /// Cancels the running entry and waits briefly for the engine to confirm, e.g. before quitting.
    func cancelAndWait(engine: EngineController) async {
        guard let task else { return }
        cancel(engine: engine)
        let deadline = ContinuousClock.now + .seconds(2)
        while isBusy, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
        if isBusy { task.cancel() }
    }

    // MARK: - Private

    private func start(_ entry: Entry, in conversation: UUID, input: URL?, library: Library, engine: EngineController) {
        active = (entry.id, conversation)
        progress = Progress()
        cancelRequested = false
        task = Task {
            let outcome = await run(entry, input: input, library: library, engine: engine)
            library.setOutcome(outcome, of: entry.id)
            active = nil
            job = nil
            task = nil
        }
    }

    private func run(_ entry: Entry, input: URL?, library: Library, engine: EngineController) async -> Entry.Outcome {
        let client = engine.client
        do {
            let id: String
            if let request = entry.request {
                id = try await client.submit(request)
            } else if let input {
                let png = try await Task.detached(priority: .userInitiated) { try ImageFiles.uploadPNG(from: input) }.value
                try await library.write(png, for: entry.id, asInput: true)  // kept for "Run Again"
                id = try await client.submit(entry.workflow, png: png)
            } else {
                return .failed("This entry has no input.")
            }
            job = id
            if cancelRequested { try? await client.cancel(id) }
            let status = try await poll(id, client: client)
            switch status.status {
            case .done:
                let png = try await client.image(of: id)
                try await library.write(png, for: entry.id)
                return .done(seconds: status.seconds ?? 0)
            case .cancelled:
                return .cancelled
            default:
                return .failed(status.error ?? "The engine could not finish this request.")
            }
        } catch is CancellationError {
            return .interrupted
        } catch EngineError.unreachable {
            engine.connectionLost()
            return .failed(EngineError.unreachable.localizedDescription)
        } catch {
            return cancelRequested ? .cancelled : .failed(error.localizedDescription)
        }
    }

    private func poll(_ id: String, client: EngineClient) async throws -> JobStatus {
        var firstSample: (step: Int, time: ContinuousClock.Instant)?
        while true {
            let status = try await client.status(of: id)
            switch status.status {
            case .queued:
                progress = Progress()
            case .running:
                var next = Progress(step: status.step, steps: status.steps)
                if status.step > 0 {  // step 0: the first request loads the model
                    let now = ContinuousClock.now
                    if firstSample == nil { firstSample = (status.step, now) }
                    if let first = firstSample, status.step > first.step {
                        let perStep = (now - first.time) / Double(status.step - first.step)
                        next.secondsLeft = Double((perStep * Double(status.steps - status.step)).components.seconds)
                    }
                }
                progress = next
            case .done, .failed, .cancelled:
                return status
            }
            try await Task.sleep(for: .milliseconds(250))
        }
    }
}
