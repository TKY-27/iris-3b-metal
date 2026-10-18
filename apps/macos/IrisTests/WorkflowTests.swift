import Darwin
import ImageIO
import Synchronization
import UniformTypeIdentifiers
import XCTest

@testable import Iris

@MainActor
final class WorkflowTests: XCTestCase {
    func testImmediateCancellationDoesNotSubmitABatch() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = Library(store: LibraryStore(root: root))
        let engine = EngineController()
        let generator = Generator()
        var settings = GenerationSettings()
        let chat = try XCTUnwrap(generator.generate(
            settings.requests(for: "fox", count: 3), in: nil, library: library, engine: engine
        ))
        await generator.cancelAndWait(engine: engine)

        XCTAssertFalse(generator.isBusy)
        XCTAssertTrue(generator.queued.isEmpty)
        XCTAssertEqual(library.conversation(chat)?.entries.map(\.outcome), [.cancelled, .cancelled, .cancelled])
        library.flush()
        XCTAssertEqual(LibraryStore(root: root).load(readOnly: true).snapshot.conversations.first?.entries.map(\.outcome),
                       [.cancelled, .cancelled, .cancelled])
    }

    /// Scripted HTTP engine recording each request.
    private func stubEngine(
        in root: URL, health: String = #"{"resident":null,"queued":0}"#, statuses: [String]
    ) async throws -> (engine: EngineController, server: Process) {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("[\(statuses.joined(separator: ","))]".utf8).write(to: root.appending(path: "statuses.json"))
        let marker = root.appending(path: "port")
        let script = """
        import json, pathlib, threading, time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        root = pathlib.Path('\(root.path)')
        statuses = json.loads((root / 'statuses.json').read_text())
        served = 0
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def reply(self, body):
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.reply(b'{"id":"test"}')
            def do_DELETE(self): self.reply(b'{}')
            def do_GET(self):
                global served
                with lock, open(root / 'requests', 'a') as log:
                    log.write(f'{self.path}\\t{time.monotonic()}\\n')
                if self.path.endswith('/health'):
                    self.reply(b'\(health)')
                elif self.path.endswith('/image'):
                    (root / 'downloading').touch()
                    deadline = time.monotonic() + 5
                    while not (root / 'release').exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.reply(b'result')
                else:
                    with lock:
                        status = statuses[min(served, len(statuses) - 1)]
                        served += 1
                    self.reply(json.dumps(status).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        (root / 'port.tmp').write_text(str(server.server_port))
        (root / 'port.tmp').replace(root / 'port')
        server.serve_forever()
        """
        let server = Process()
        server.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        server.arguments = ["-c", script]
        let log = root.appending(path: "server.log")
        try Data().write(to: log)
        let output = try FileHandle(forWritingTo: log)
        defer { try? output.close() }
        server.standardOutput = output
        server.standardError = output
        try server.run()
        var handedOff = false
        defer { if !handedOff && server.isRunning { server.terminate() } }
        let startupDeadline = ContinuousClock.now + .seconds(15)
        while !FileManager.default.fileExists(atPath: marker.path), server.isRunning,
              ContinuousClock.now < startupDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let portText = (try? String(contentsOf: marker, encoding: .utf8)) ?? ""
        let state = server.isRunning ? "still running" : "exited with status \(server.terminationStatus)"
        let diagnostics = (try? String(contentsOf: log, encoding: .utf8)) ?? ""
        let port = try XCTUnwrap(Int(portText), "Stub engine \(state); output: \(diagnostics)")
        let suite = "io.github.tky27.iris.stub-engine-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.register(defaults: ["enginePort": port])
        let engine = EngineController(defaults: defaults)
        await engine.start()
        XCTAssertEqual(engine.state, .ready)
        handedOff = true
        return (engine, server)
    }

    private func requests(in root: URL) throws -> [(path: String, time: Double)] {
        try String(contentsOf: root.appending(path: "requests"), encoding: .utf8).split(separator: "\n").map { line in
            let fields = line.split(separator: "\t")
            return (String(fields[0]), Double(fields[1]) ?? 0)
        }
    }

    func testCancellationDuringResultDownloadDoesNotPublishImage() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let downloading = root.appending(path: "downloading")
        let release = root.appending(path: "release")
        let (engine, server) = try await stubEngine(
            in: root, statuses: [#"{"id":"test","status":"done","step":1,"steps":1,"seconds":1}"#]
        )
        defer { if server.isRunning { server.terminate() } }
        let library = Library(store: LibraryStore(root: root))
        let generator = Generator()
        var settings = GenerationSettings()
        let chat = try XCTUnwrap(generator.generate(
            [settings.request(for: "fox")], in: nil, library: library, engine: engine
        ))
        let downloadDeadline = ContinuousClock.now + .seconds(5)
        while !FileManager.default.fileExists(atPath: downloading.path), ContinuousClock.now < downloadDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: downloading.path))
        generator.cancel(engine: engine)
        try Data().write(to: release)
        await generator.cancelAndWait(engine: engine)

        let entry = try XCTUnwrap(library.conversation(chat)?.entries.first)
        XCTAssertEqual(entry.outcome, .cancelled)
        XCTAssertFalse(FileManager.default.fileExists(atPath: library.imageURL(entry.id).path))
        XCTAssertFalse(generator.isBusy)
        library.flush()
    }

    /// Outcome and metadata are saved together.
    func testFinishedEntryKeepsOutcomeAndMetadataTogether() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data().write(to: root.appending(path: "release"))
        let (engine, server) = try await stubEngine(in: root, statuses: [
            #"{"id":"test","status":"done","step":1,"steps":1,"seconds":2.5,"metadata":{"sampler":"euler","hf_revision":"abc"}}"#,
        ])
        defer { if server.isRunning { server.terminate() } }
        let library = Library(store: LibraryStore(root: root))
        let generator = Generator()
        var settings = GenerationSettings()
        let chat = try XCTUnwrap(generator.generate([settings.request(for: "fox")], in: nil, library: library, engine: engine))
        while generator.isBusy { try await Task.sleep(for: .milliseconds(20)) }
        library.flush()

        let saved = try XCTUnwrap(LibraryStore(root: root).load(readOnly: true).snapshot.conversations.first { $0.id == chat }?.entries.first)
        XCTAssertEqual(saved.outcome, .done(seconds: 2.5))
        XCTAssertEqual(saved.metadata?.sampler, "euler")
        XCTAssertEqual(saved.metadata?.hfRevision, "abc")
        XCTAssertEqual(try Data(contentsOf: library.imageURL(saved.id)), Data("result".utf8))
    }

    /// Long polls send revisions without hot-looping.
    func testStatusWaitsForRevisionsWhenTheEngineOffersIt() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data().write(to: root.appending(path: "release"))
        let running = #"{"id":"test","status":"running","step":1,"steps":4,"seconds":null,"revision":7}"#
        let (engine, server) = try await stubEngine(
            in: root, health: #"{"resident":null,"queued":0,"max_job_wait_seconds":10}"#,
            statuses: [running, running, #"{"id":"test","status":"done","step":4,"steps":4,"seconds":1,"revision":8}"#]
        )
        defer { if server.isRunning { server.terminate() } }
        XCTAssertEqual(engine.health?.maxJobWaitSeconds, 10)
        let library = Library(store: LibraryStore(root: root))
        let generator = Generator()
        var settings = GenerationSettings()
        generator.generate([settings.request(for: "fox")], in: nil, library: library, engine: engine)
        while generator.isBusy { try await Task.sleep(for: .milliseconds(20)) }
        library.flush()

        let polls = try requests(in: root).filter { $0.path.contains("/jobs/test") && !$0.path.hasSuffix("/image") }
        XCTAssertEqual(polls.map(\.path), ["/v1/jobs/test", "/v1/jobs/test?after=7&wait=5.0", "/v1/jobs/test?after=7&wait=5.0"])
        // unchanged answer: next request waited
        XCTAssertGreaterThanOrEqual(polls[2].time - polls[1].time, 0.2)
    }

    func testAlreadyCancelledStartupDoesNotConnect() async {
        let engine = EngineController()
        let startup = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            await engine.start()
        }
        await startup.value
        XCTAssertEqual(engine.state, .stopped)
    }

    func testCancellingStartupStopsItsProcess() async throws {
        let suite = "io.github.tky27.iris.engine-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appending(path: "launched")
        let executable = directory.appending(path: "silent-engine")
        try Data("#!/bin/sh\nprintf '%s' \"$$\" > '\(marker.path)'\nexec /bin/sleep 30\n".utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)

        let engine = EngineController(defaults: defaults)
        engine.port = 62413
        engine.executablePath = executable.path
        guard (try? await engine.client.health()) == nil else {
            throw XCTSkip("port 62413 is occupied")
        }
        let startup = Task { await engine.start() }
        defer { startup.cancel() }
        let deadline = ContinuousClock.now + .seconds(3)
        while !FileManager.default.fileExists(atPath: marker.path), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let pid = try XCTUnwrap(Int32(String(contentsOf: marker, encoding: .utf8)))
        startup.cancel()
        await startup.value
        XCTAssertEqual(engine.state, .stopped)
        let exitDeadline = ContinuousClock.now + .seconds(3)
        while kill(pid, 0) == 0, ContinuousClock.now < exitDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertEqual(kill(pid, 0), -1)
    }

    func testAlreadyCancelledRestartDoesNotLaunch() async throws {
        let suite = "io.github.tky27.iris.engine-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let engine = EngineController(defaults: defaults)
        let restart = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            await engine.restart()
        }
        await restart.value
        XCTAssertEqual(engine.state, .stopped)
        XCTAssertFalse(engine.isRestarting)
    }

    /// An engine stand-in ignoring SIGTERM.
    private func stubbornEngine(in directory: URL) throws -> (executable: URL, marker: URL) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let marker = directory.appending(path: "launched")
        let executable = directory.appending(path: "stubborn-engine")
        let script = """
        #!/usr/bin/python3
        import os, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open("\(marker.path).tmp", "w") as marker:
            marker.write(str(os.getpid()))
        os.replace("\(marker.path).tmp", "\(marker.path)")
        time.sleep(30)
        """
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
        return (executable, marker)
    }

    func testStopDuringRestartWaitDoesNotStartAgain() async throws {
        let suite = "io.github.tky27.iris.engine-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let (executable, marker) = try stubbornEngine(in: directory)

        let engine = EngineController(defaults: defaults)
        engine.port = 62413
        engine.executablePath = executable.path
        guard (try? await engine.client.health()) == nil else {
            throw XCTSkip("port 62413 is occupied")
        }
        let startup = Task { await engine.start() }
        defer { startup.cancel() }
        let deadline = ContinuousClock.now + .seconds(3)
        while !FileManager.default.fileExists(atPath: marker.path), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let pid = try XCTUnwrap(Int32(String(contentsOf: marker, encoding: .utf8)))
        defer { kill(pid, SIGKILL) }
        let restarting = Task { await engine.restart() }
        defer { restarting.cancel() }
        while !engine.isRestarting, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(engine.isRestarting)
        engine.stop()
        await restarting.value
        XCTAssertEqual(engine.state, .stopped)
        XCTAssertFalse(engine.isRestarting)
        startup.cancel()
        await startup.value
    }

    /// The old engine exits before reconnecting.
    func testRestartReapsAStubbornEngineBeforeReconnecting() async throws {
        let suite = "io.github.tky27.iris.engine-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let (executable, marker) = try stubbornEngine(in: directory)
        let probe = Mutex<(restarting: Bool, oldAlive: Bool?)>((false, nil))
        let engine = EngineController(defaults: defaults) { _ in
            let pid = Int32(try String(contentsOf: marker, encoding: .utf8))
            return try probe.withLock { state in
                state.oldAlive = pid.map { kill($0, 0) == 0 }
                guard state.restarting else { throw URLError(.cannotConnectToHost) }
                return EngineHealth(resident: nil, queued: 0, engineVersion: nil, apiVersion: nil)
            }
        }
        engine.executablePath = executable.path
        let startup = Task { await engine.start() }
        let deadline = ContinuousClock.now + .seconds(3)
        while !FileManager.default.fileExists(atPath: marker.path), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let pid = try XCTUnwrap(Int32(String(contentsOf: marker, encoding: .utf8)))
        defer { kill(pid, SIGKILL) }
        startup.cancel()
        await startup.value
        XCTAssertEqual(engine.state, .stopped)
        XCTAssertEqual(kill(pid, 0), 0, "SIGTERM was ignored")

        probe.withLock { $0.restarting = true }
        await engine.restart(grace: .milliseconds(200))
        XCTAssertEqual(engine.state, .ready)
        XCTAssertEqual(probe.withLock { $0.oldAlive }, false, "the health check ran while the old engine existed")
        XCTAssertEqual(kill(pid, 0), -1)
    }

    /// A 20x10 JPEG whose EXIF orientation rotates it to portrait.
    private func rotatedJPEG() throws -> URL {
        let context = try XCTUnwrap(CGContext(
            data: nil, width: 20, height: 10, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ))
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 20, height: 10))
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).jpg")
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        let properties = [kCGImagePropertyOrientation: CGImagePropertyOrientation.right.rawValue] as CFDictionary
        CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), properties)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }

    func testInputIsSentAsUprightPNG() throws {
        let png = try ImageFiles.uploadPNG(from: rotatedJPEG())
        let source = try XCTUnwrap(CGImageSourceCreateWithData(png as CFData, nil))
        XCTAssertEqual(CGImageSourceGetType(source) as String?, UTType.png.identifier)
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual([image.width, image.height], [10, 20])  // EXIF rotation applied
    }

    func testPixelSizeIsUpright() throws {
        XCTAssertEqual(ImageFiles.pixelSize(of: try rotatedJPEG()), CGSize(width: 10, height: 20))
    }

    /// Cancellation before decoding returns no image.
    func testCancelledThumbnailRequestReturnsNilBeforeDecode() async throws {
        let url = try landscapePNG()
        defer { try? FileManager.default.removeItem(at: url) }
        let gate = AsyncStream<Void>.makeStream()
        let request = Task {
            for await _ in gate.stream { break }
            return await ImageFiles.thumbnail(of: url, maxPixels: 600)
        }
        request.cancel()
        gate.continuation.yield(())
        gate.continuation.finish()
        let image = await request.value
        XCTAssertNil(image)
    }

    private func solidBMP(blue: Bool) -> Data {
        var data = Data([0x42, 0x4d])
        func append32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        func append16(_ value: UInt16) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        append32(70)
        append32(0)
        append32(54)
        append32(40)
        append32(2)
        append32(2)
        append16(1)
        append16(24)
        append32(0)
        append32(16)
        for _ in 0..<4 { append32(0) }
        let pixel: [UInt8] = blue ? [255, 0, 0] : [0, 0, 255]
        for _ in 0..<2 { data.append(contentsOf: pixel + pixel + [0, 0]) }
        return data
    }

    /// Reselecting refreshes replaced images with identical metadata.
    func testReattachingReplacedFileRefreshesThumbnail() async throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).bmp")
        defer { try? FileManager.default.removeItem(at: url) }
        try solidBMP(blue: false).write(to: url)
        let originalDate = Date(timeIntervalSince1970: 1_700_000_000)
        try FileManager.default.setAttributes([.modificationDate: originalDate], ofItemAtPath: url.path)
        let original = try FileManager.default.attributesOfItem(atPath: url.path)
        let workspace = Workspace()
        workspace.attach(url)
        let firstID = try XCTUnwrap(workspace.attachment?.id)
        let firstThumbnail = await ImageFiles.thumbnail(of: url, maxPixels: 64, version: firstID.uuidString)
        let firstImage = try XCTUnwrap(firstThumbnail)

        try solidBMP(blue: true).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.modificationDate: originalDate], ofItemAtPath: url.path)
        let replaced = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual(original[.size] as? Int, replaced[.size] as? Int)
        XCTAssertEqual(original[.modificationDate] as? Date, replaced[.modificationDate] as? Date)

        workspace.attach(url)
        let secondID = try XCTUnwrap(workspace.attachment?.id)
        XCTAssertNotEqual(firstID, secondID)
        let freshThumbnail = await ImageFiles.thumbnail(of: url, maxPixels: 64, version: secondID.uuidString)
        let freshImage = try XCTUnwrap(freshThumbnail)
        XCTAssertNotEqual(try XCTUnwrap(firstImage.dataProvider?.data) as Data,
                          try XCTUnwrap(freshImage.dataProvider?.data) as Data)
    }

    /// A 1024 x 768 PNG, which the engine reduces before enlarging it.
    private func landscapePNG() throws -> URL {
        let context = try XCTUnwrap(CGContext(
            data: nil, width: 1024, height: 768, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ))
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).png")
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }

    func testSeedIsDrawnOnlyWhenRandomized() {
        var settings = GenerationSettings()
        settings.randomizeSeed = false
        settings.seed = 42
        settings.steps = 9999  // typed past the stepper's range
        let request = settings.request(for: "fox")
        XCTAssertEqual(request.seed, 42)
        XCTAssertEqual(request.steps, GenerationSettings.stepRange.upperBound)
        settings.reuse(GenerationRequest(
            prompt: "fox", width: 1344, height: 768, steps: 50, cfgScale: 4, seed: 7, negativePrompt: "blur"
        ))
        XCTAssertEqual(settings.size, ImageSize(width: 1344, height: 768))
        XCTAssertEqual([settings.steps, settings.seed], [50, 7])
        XCTAssertFalse(settings.randomizeSeed)
    }

    func testConversationTitleIsShortenedAtAWord() {
        let entry = Entry(workflow: .create, request: GenerationRequest(
            prompt: "A lighthouse on a rocky coast at dusk, waves breaking against the rocks",
            width: 1024, height: 1024, steps: 100, cfgScale: 3, seed: 0, negativePrompt: ""
        ))
        XCTAssertEqual(Conversation.title(for: entry), "A lighthouse on a rocky coast at dusk, waves…")
    }

    private func runningTestEngine() async throws -> EngineController {
        let environment = ProcessInfo.processInfo.environment
        guard environment["IRIS_ENGINE_TEST"] == "1" else {
            throw XCTSkip("Set TEST_RUNNER_IRIS_ENGINE_TEST=1 for the stand-in engine.")
        }
        let port = try XCTUnwrap(Int(environment["IRIS_TEST_PORT"] ?? "8765"))
        guard EngineController.portRange.contains(port) else {
            XCTFail("Invalid test engine port: \(port)")
            throw URLError(.badURL)
        }
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "IrisOptInEngine-\(UUID().uuidString)"))
        defaults.register(defaults: ["enginePort": port])
        let engine = EngineController(defaults: defaults)
        _ = try await engine.client.health()
        await engine.start()
        XCTAssertEqual(engine.state, .ready)
        return engine
    }

    /// A batch runs one image after another in one chat; Stop ends the rest of it.
    /// Opt-in against the stand-in engine, like the test below.
    func testBatchRunsInOrderAndStopEndsTheRest() async throws {
        let engine = try await runningTestEngine()
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = Library(store: LibraryStore(root: root))
        let generator = Generator()
        var settings = GenerationSettings()
        settings.steps = 3
        settings.size = ImageSize(width: 64, height: 64)

        let chat = try XCTUnwrap(generator.generate(
            settings.requests(for: "fox", count: 3), in: nil, library: library, engine: engine
        ))
        XCTAssertEqual(generator.queued.count, 2)
        while generator.isBusy { try await Task.sleep(for: .milliseconds(20)) }
        let entries = try XCTUnwrap(library.conversation(chat)?.entries)
        XCTAssertEqual(entries.map(\.isDone), [true, true, true])
        XCTAssertEqual(Set(entries.compactMap(\.request?.seed)).count, 3)

        let stopped = try XCTUnwrap(generator.generate(
            settings.requests(for: "owl", count: 3), in: nil, library: library, engine: engine
        ))
        generator.cancel(engine: engine)
        while generator.isBusy { try await Task.sleep(for: .milliseconds(20)) }
        let outcomes = try XCTUnwrap(library.conversation(stopped)?.entries.map(\.outcome))
        XCTAssertEqual(outcomes.dropFirst().map { $0 }, [.cancelled, .cancelled])
        XCTAssertTrue(generator.queued.isEmpty)
        library.flush()
    }

    /// End to end against the stand-in engine (`python apps/macos/fake_engine.py`) on the
    /// default port. Opt-in, so it never sends work to a real engine by accident:
    /// `TEST_RUNNER_IRIS_ENGINE_TEST=1 xcodebuild -scheme Iris test`.
    func testDepthWorkflowAgainstRunningEngine() async throws {
        let engine = try await runningTestEngine()
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = Library(store: LibraryStore(root: root))
        let generator = Generator()

        let conversation = try XCTUnwrap(generator.process(
            .depth, image: rotatedJPEG(), named: "photo.jpg", in: nil, library: library, engine: engine
        ))
        while generator.isBusy { try await Task.sleep(for: .milliseconds(50)) }
        let entry = try XCTUnwrap(library.conversation(conversation)?.entries.first)
        XCTAssertTrue(entry.isDone, "\(entry.outcome)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: library.imageURL(entry.id).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: library.inputURL(entry.id).path))
    }

    /// An upscale reports its tiles as it goes, keeps its original for Compare, and comes out
    /// at the size the composer predicts. Opt-in against the stand-in engine, like the others.
    func testUpscaleReportsTilesAndMatchesThePredictedSize() async throws {
        let engine = try await runningTestEngine()
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = Library(store: LibraryStore(root: root))
        let generator = Generator()
        let input = try landscapePNG()
        defer { try? FileManager.default.removeItem(at: input) }

        let conversation = try XCTUnwrap(generator.process(
            .upscale, image: input, named: "photo.png", in: nil, library: library, engine: engine
        ))
        var tiles: Set<Int> = []
        while generator.isBusy {
            if generator.progress.phase == .sampling { tiles.insert(generator.progress.steps) }
            try await Task.sleep(for: .milliseconds(20))
        }
        let entry = try XCTUnwrap(library.conversation(conversation)?.entries.first)
        XCTAssertTrue(entry.isDone, "\(entry.outcome)")
        XCTAssertEqual(tiles.count, 1, "one tile count, reported while running")
        XCTAssertEqual(ImageFiles.pixelSize(of: library.imageURL(entry.id)), Upscaling.output(for: CGSize(width: 1024, height: 768)))
        XCTAssertEqual(ImageFiles.pixelSize(of: library.inputURL(entry.id)), CGSize(width: 1024, height: 768))
        library.flush()
    }
}
