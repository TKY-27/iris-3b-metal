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

    var port: Int {
        get { UserDefaults.standard.object(forKey: "enginePort") as? Int ?? 8765 }
        set { UserDefaults.standard.set(newValue, forKey: "enginePort") }
    }

    /// An explicit path to the `iris-metal` executable; empty means search the usual places.
    var executablePath: String {
        get { UserDefaults.standard.string(forKey: "enginePath") ?? "" }
        set { UserDefaults.standard.set(newValue, forKey: "enginePath") }
    }

    private var process: Process?
    private var stopping = false

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
        guard state != .starting, state != .ready else { return }
        if (try? await client.health()) != nil {
            state = .ready
            return
        }
        guard let executable = resolvedExecutable() else {
            state = .failed("Could not find iris-metal. Install it (see the README) or set its location in Settings.")
            return
        }
        launch(executable)
        await waitUntilReady()
    }

    /// A request found the server gone while it was believed ready.
    func connectionLost() {
        guard state == .ready else { return }
        process = nil
        state = .failed("The engine stopped responding.")
    }

    func stop() {
        stopping = true
        process?.terminate()
        process = nil
        state = .stopped
    }

    func restart() async {
        stop()
        await start()
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
            let text = String(decoding: handle.availableData, as: UTF8.self)
            Task { @MainActor in self?.appendLog(text) }
        }
        process.terminationHandler = { [weak self] finished in
            let code = finished.terminationStatus
            Task { @MainActor in self?.processExited(code: code) }
        }
        do {
            try process.run()
            self.process = process
        } catch {
            state = .failed("Could not start \(executable.path): \(error.localizedDescription)")
        }
    }

    private func waitUntilReady() async {
        while state == .starting {
            if (try? await client.health()) != nil {
                state = .ready
                return
            }
            try? await Task.sleep(for: .seconds(1))
        }
    }

    private func appendLog(_ text: String) {
        // progress bars redraw with carriage returns; keep only the latest frame of each line
        let lines = text.components(separatedBy: "\n").compactMap {
            $0.components(separatedBy: "\r").last { !$0.isEmpty }
        }
        logTail = Array((logTail + lines).suffix(12))
    }

    private func processExited(code: Int32) {
        process = nil
        guard !stopping else { return }
        state = .failed("The engine stopped unexpectedly (exit code \(code)).")
    }
}
