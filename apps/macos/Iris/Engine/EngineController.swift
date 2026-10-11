import Foundation
import Observation

/// Owns the connection to `iris-metal serve`: attaches to a running server or
/// launches one, and reports what it is doing in user terms.
@MainActor
@Observable
final class EngineController {
    enum State: Equatable {
        case stopped
        case starting
        case ready
        case failed(String)
    }

    private(set) var state: State = .stopped
    /// The last lines the engine printed, for diagnosing a slow or failed start.
    private(set) var logTail: [String] = []

    var client: EngineClient { EngineClient(port: port) }

    static let portRange = 1024...65535

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var port: Int {
        get {
            let stored = defaults.object(forKey: "enginePort")
            // a `-enginePort 8790` launch argument arrives as a string
            let port = stored as? Int ?? (stored as? String).flatMap { Int($0) } ?? 8765
            return port.clamped(to: Self.portRange)
        }
        set { defaults.set(newValue.clamped(to: Self.portRange), forKey: "enginePort") }
    }

    /// An explicit path to the `iris-metal` executable; empty means search the usual places.
    var executablePath: String {
        get { defaults.string(forKey: "enginePath") ?? "" }
        set { defaults.set(newValue, forKey: "enginePath") }
    }

    private var process: Process?
    private var connectionID: UUID?
    private var stopping = false
    private(set) var isRestarting = false

    static let searchPaths = [
        "~/.local/bin/iris-metal",
        "/opt/homebrew/bin/iris-metal",
        "/usr/local/bin/iris-metal",
    ]

    func resolvedExecutable() -> URL? {
        let candidates = executablePath.isEmpty ? Self.searchPaths : [executablePath]
        return candidates
            .map { URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath) }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    func start() async {
        guard state != .starting, state != .ready, !isRestarting else { return }
        await connect()
    }

    /// Attaches to a running server or launches one. `.starting` is set before the
    /// first await, so a second call cannot launch a second engine meanwhile.
    private func connect() async {
        let id = UUID()
        connectionID = id
        state = .starting
        guard !Task.isCancelled else { stop(); return }
        if (try? await client.health()) != nil {
            guard connectionID == id, state == .starting else { return }
            guard !Task.isCancelled else { stop(); return }
            state = .ready
            return
        }
        guard connectionID == id, state == .starting else { return }
        guard !Task.isCancelled else { stop(); return }
        if let process {
            guard !process.isRunning else {
                state = .failed("The engine is still running but did not answer. Restart it in Settings.")
                return
            }
            self.process = nil
        }
        guard let executable = resolvedExecutable() else {
            state = .failed("Could not find iris-metal. Install it (see the README) or set its location in Settings.")
            return
        }
        launch(executable)
        guard !Task.isCancelled else { stop(); return }
        await waitUntilReady(id: id)
    }

    /// A request found the server gone while it was believed ready. A process we
    /// launched is kept, so quitting or restarting still terminates it if it hangs.
    func connectionLost() {
        guard state == .ready else { return }
        connectionID = nil
        state = .failed("The engine stopped responding.")
    }

    func stop() {
        connectionID = nil
        stopping = true
        if let process {
            if process.isRunning { process.terminate() }
            if !process.isRunning { self.process = nil }
        }
        state = .stopped
    }

    /// Stops a launched engine and waits for it to exit, so the new one can
    /// bind the port and the old one cannot answer the first health check.
    func restart() async {
        guard !isRestarting else { return }
        isRestarting = true
        defer { isRestarting = false }
        let old = process
        stop()
        guard !Task.isCancelled else { return }
        let id = UUID()
        connectionID = id
        state = .starting
        if let old {
            let deadline = ContinuousClock.now + .seconds(10)
            while old.isRunning, ContinuousClock.now < deadline {
                do {
                    try await Task.sleep(for: .milliseconds(100))
                } catch {
                    guard connectionID == id, state == .starting else { return }
                    connectionID = nil
                    state = .stopped
                    return
                }
                guard connectionID == id, state == .starting else { return }
            }
            guard connectionID == id, state == .starting else { return }
            if Task.isCancelled {
                connectionID = nil
                state = .stopped
                return
            }
            if old.isRunning { kill(old.processIdentifier, SIGKILL) }
        }
        guard connectionID == id, state == .starting else { return }
        await connect()
    }

    private func launch(_ executable: URL) {
        state = .starting
        logTail = []
        stopping = false
        let process = Process()
        process.executableURL = executable
        process.arguments = ["serve", "--port", String(port)]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else {  // end of file: without this the handler keeps firing
                handle.readabilityHandler = nil
                return
            }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.appendLog(text) }
        }
        process.terminationHandler = { [weak self] finished in
            let code = finished.terminationStatus
            let exited = ObjectIdentifier(finished)
            Task { @MainActor in self?.processExited(exited, code: code) }
        }
        do {
            try process.run()
            self.process = process
        } catch {
            state = .failed("Could not start \(executable.path): \(error.localizedDescription)")
        }
    }

    private func waitUntilReady(id: UUID) async {
        while connectionID == id, state == .starting {
            guard !Task.isCancelled else { stop(); return }
            if (try? await client.health()) != nil {
                guard connectionID == id, state == .starting else { return }
                guard !Task.isCancelled else { stop(); return }
                state = .ready
                return
            }
            guard connectionID == id, state == .starting else { return }
            guard !Task.isCancelled else { stop(); return }
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                if connectionID == id, state == .starting { stop() }
                return
            }
        }
    }

    private func appendLog(_ text: String) {
        // progress bars redraw with carriage returns; keep only the latest frame of each line
        let lines = text.components(separatedBy: "\n").compactMap {
            $0.components(separatedBy: "\r").last { !$0.isEmpty }
        }
        logTail = Array((logTail + lines).suffix(12))
    }

    private func processExited(_ exited: ObjectIdentifier, code: Int32) {
        // after a restart the old process may exit late; it says nothing about the new one
        guard let process, ObjectIdentifier(process) == exited else { return }
        self.process = nil
        guard !stopping else { return }
        state = .failed("The engine stopped unexpectedly (exit code \(code)).")
    }
}
