import Foundation
import Observation

/// Runs one library entry at a time on the engine and records its outcome.
@MainActor
@Observable
final class Generator {
    /// What the engine reports about the running entry, and nothing more.
    struct Progress: Equatable {
        enum Phase: Equatable {
            /// Queued behind another request, or not yet submitted.
            case waiting
            /// Running before the first step or tile: loading the model if needed and reading the prompt.
            case preparing
            /// Steps of an image, or tiles of an upscale, under way.
            case sampling
            /// The last step is done; the image is being decoded and sent.
            case finishing
            /// Depth runs in one pass and reports nothing until it is done.
            case working
        }

        var phase = Phase.waiting
        var step = 0
        var steps = 0
        var secondsLeft: Double?

        init(phase: Phase = .waiting, step: Int = 0, steps: Int = 0, secondsLeft: Double? = nil) {
            self.phase = phase
            self.step = step
            self.steps = steps
            self.secondsLeft = secondsLeft
        }

        init(_ status: JobStatus.State, step: Int, steps: Int, workflow: Workflow) {
            self.step = step
            self.steps = steps
            switch status {
            case .queued, .done, .failed, .cancelled:
                phase = .waiting
            case .running where workflow == .depth:
                phase = .working
            case .running where steps == 0, .running where workflow == .create && step == 0:
                phase = .preparing
            case .running:
                // an upscale reports finished tiles before each one, so 0 of 9 is under way
                phase = step < steps ? .sampling : .finishing
            }
        }

        /// Completed sampling steps as a fraction, once sampling has started.
        var fraction: Double? {
            guard steps > 0, phase == .sampling || phase == .finishing else { return nil }
            return Double(step) / Double(steps)
        }

        var percent: Int? { fraction == nil ? nil : step * 100 / steps }
    }

    /// The entry being run and its conversation.
    private(set) var active: (entry: UUID, conversation: UUID)?
    private(set) var progress = Progress()
    /// The running entry whose input image is on disk, so its thumbnail can show.
    private(set) var storedInput: UUID?

    /// Entries of the same batch waiting their turn; the engine makes one image at a time.
    private(set) var queued: [Entry] = []

    @ObservationIgnored private var job: String?
    @ObservationIgnored private var cancelRequested = false
    @ObservationIgnored private var task: Task<Void, Never>?

    var isBusy: Bool { active != nil }

    func isQueued(_ entry: UUID) -> Bool { queued.contains { $0.id == entry } }

    /// Adds one text-to-image entry per request to `conversation` (a new one when nil) and runs
    /// them in order.
    @discardableResult
    func generate(
        _ requests: [GenerationRequest], in conversation: UUID?, project: UUID? = nil,
        library: Library, engine: EngineController
    ) -> UUID? {
        guard !isBusy, !library.isReadOnly, !requests.isEmpty else { return nil }
        let entries = requests.map { Entry(workflow: .create, request: $0) }
        guard let id = library.append(entries[0], to: conversation, project: project) else { return nil }
        for entry in entries.dropFirst() { library.append(entry, to: id) }
        queued = Array(entries.dropFirst())
        start(entries[0], in: id, input: nil, library: library, engine: engine)
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
        guard let id = library.append(entry, to: conversation, project: project) else { return nil }
        start(entry, in: id, input: image, library: library, engine: engine)
        return id
    }

    /// Runs a finished, failed or cancelled entry's request again as a new entry.
    func rerun(_ entry: Entry, newSeed: Bool, in conversation: UUID, library: Library, engine: EngineController) {
        if var request = entry.request {
            if newSeed { request.seed = Int.random(in: GenerationSettings.seedRange) }
            generate([request], in: conversation, library: library, engine: engine)
        } else if FileManager.default.fileExists(atPath: library.inputURL(entry.id).path) {
            process(
                entry.workflow, image: library.inputURL(entry.id), named: entry.inputName ?? "image",
                in: conversation, library: library, engine: engine
            )
        }
    }

    /// Stops the running entry and the rest of its batch.
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
        while isBusy, !Task.isCancelled, ContinuousClock.now < deadline {
            do {
                try await Task.sleep(for: .milliseconds(50))
            } catch {
                break
            }
        }
        if isBusy { task.cancel() }
        await task.value
    }

    // MARK: - Private

    private func start(_ entry: Entry, in conversation: UUID, input: URL?, library: Library, engine: EngineController) {
        active = (entry.id, conversation)
        progress = Progress()
        cancelRequested = false
        task = Task {
            let outcome = await run(entry, input: input, library: library, engine: engine)
            library.setOutcome(outcome, of: entry.id)
            job = nil
            task = nil
            // the rest of a batch runs unless it was stopped or the engine went away
            if cancelRequested || engine.state != .ready {
                let rest: Entry.Outcome = cancelRequested ? .cancelled : .interrupted
                for waiting in queued { library.setOutcome(rest, of: waiting.id) }
                queued = []
            }
            if queued.isEmpty {
                active = nil
            } else {
                start(queued.removeFirst(), in: conversation, input: nil, library: library, engine: engine)
            }
        }
    }

    private func run(_ entry: Entry, input: URL?, library: Library, engine: EngineController) async -> Entry.Outcome {
        let client = engine.client
        do {
            try checkCancellation()
            let id: String
            if let request = entry.request {
                guard engine.health?.supports(request.model) == true else {
                    return .failed("This engine does not support \(request.model.fullTitle). Update the engine in Settings.")
                }
                id = try await client.submit(request)
            } else if let input {
                let png = try await Task.detached(priority: .userInitiated) { try ImageFiles.uploadPNG(from: input) }.value
                try checkCancellation()
                try await library.write(png, for: entry.id, asInput: true)  // kept for "Run Again"
                storedInput = entry.id
                try checkCancellation()
                id = try await client.submit(entry.workflow, png: png)
            } else {
                return .failed("This entry has no input.")
            }
            job = id
            if cancelRequested { try? await client.cancel(id) }
            let status = try await poll(id, workflow: entry.workflow, client: client)
            switch status.status {
            case .done:
                try checkCancellation()
                let png = try await client.image(of: id)
                try checkCancellation()
                try await library.write(png, for: entry.id)
                try checkCancellation()
                library.setMetadata(status.metadata, of: entry.id)
                return .done(seconds: status.seconds ?? 0)
            case .cancelled:
                return .cancelled
            default:
                return .failed(status.error ?? "The engine could not finish this request.")
            }
        } catch is CancellationError {
            return cancelRequested ? .cancelled : .interrupted
        } catch EngineError.unreachable {
            if cancelRequested { return .cancelled }
            engine.connectionLost()
            return .failed(EngineError.unreachable.diagnosticDescription)
        } catch let error as EngineError {
            return cancelRequested ? .cancelled : .failed(error.diagnosticDescription)
        } catch {
            return cancelRequested ? .cancelled : .failed(error.localizedDescription)
        }
    }

    private func checkCancellation() throws {
        try Task.checkCancellation()
        if cancelRequested { throw CancellationError() }
    }

    private func poll(_ id: String, workflow: Workflow, client: EngineClient) async throws -> JobStatus {
        var firstSample: (step: Int, time: ContinuousClock.Instant)?
        while true {
            let status = try await client.status(of: id)
            switch status.status {
            case .queued:
                progress = Progress()
            case .running:
                var next = Progress(.running, step: status.step, steps: status.steps, workflow: workflow)
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
