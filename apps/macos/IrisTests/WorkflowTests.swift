import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import Iris

@MainActor
final class WorkflowTests: XCTestCase {
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
}
