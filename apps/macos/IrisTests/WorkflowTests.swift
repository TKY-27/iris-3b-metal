import Darwin
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import Iris

@MainActor
final class WorkflowTests: XCTestCase {
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

    func testStopDuringRestartWaitDoesNotStartAgain() async throws {
        let suite = "io.github.tky27.iris.engine-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appending(path: "launched")
        let executable = directory.appending(path: "stubborn-engine")
        let script = """
        #!/usr/bin/python3
        import os, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open("\(marker.path)", "w") as marker:
            marker.write(str(os.getpid()))
        time.sleep(30)
        """
        try Data(script.utf8).write(to: executable)
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

    /// A batch runs one image after another in one chat; Stop ends the rest of it.
    /// Opt-in against the stand-in engine, like the test below.
    func testBatchRunsInOrderAndStopEndsTheRest() async throws {
        guard ProcessInfo.processInfo.environment["IRIS_ENGINE_TEST"] == "1" else {
            throw XCTSkip("set TEST_RUNNER_IRIS_ENGINE_TEST=1 with a stand-in engine on 127.0.0.1:8765")
        }
        let engine = EngineController()
        guard (try? await engine.client.health()) != nil else {
            throw XCTSkip("no engine listening on 127.0.0.1:\(engine.port)")
        }
        await engine.start()
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
        guard ProcessInfo.processInfo.environment["IRIS_ENGINE_TEST"] == "1" else {
            throw XCTSkip("set TEST_RUNNER_IRIS_ENGINE_TEST=1 with a stand-in engine on 127.0.0.1:8765")
        }
        let engine = EngineController()
        guard (try? await engine.client.health()) != nil else {
            throw XCTSkip("no engine listening on 127.0.0.1:\(engine.port)")
        }
        await engine.start()
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
        guard ProcessInfo.processInfo.environment["IRIS_ENGINE_TEST"] == "1" else {
            throw XCTSkip("set TEST_RUNNER_IRIS_ENGINE_TEST=1 with a stand-in engine on 127.0.0.1:8765")
        }
        let engine = EngineController()
        guard (try? await engine.client.health()) != nil else {
            throw XCTSkip("no engine listening on 127.0.0.1:\(engine.port)")
        }
        await engine.start()
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
