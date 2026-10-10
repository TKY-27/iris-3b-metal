import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import Iris

@MainActor
final class WorkflowTests: XCTestCase {
    /// A 20x10 JPEG whose EXIF orientation rotates it to portrait.
    private func rotatedJPEG() throws -> URL {
        let context = CGContext(
            data: nil, width: 20, height: 10, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 20, height: 10))
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).jpg")
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        let properties = [kCGImagePropertyOrientation: CGImagePropertyOrientation.right.rawValue] as CFDictionary
        CGImageDestinationAddImage(destination, context.makeImage()!, properties)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }

    func testInputIsSentAsUprightPNG() throws {
        let png = try Generator.png(from: rotatedJPEG())
        let source = try XCTUnwrap(CGImageSourceCreateWithData(png as CFData, nil))
        XCTAssertEqual(CGImageSourceGetType(source) as String?, UTType.png.identifier)
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual([image.width, image.height], [10, 20])  // EXIF rotation applied
    }

    /// End to end against `iris-metal serve` (or the stand-in engine) on the default port.
    func testDepthWorkflowAgainstRunningEngine() async throws {
        let engine = EngineController()
        guard (try? await engine.client.health()) != nil else {
            throw XCTSkip("no engine listening on 127.0.0.1:\(engine.port)")
        }
        await engine.start()
        let history = History()
        let generator = Generator()
        generator.workflow = .depth
        generator.input = try rotatedJPEG()

        let result = await generator.run(engine: engine, history: history)
        let item = try XCTUnwrap(result)
        defer { history.remove(item) }
        XCTAssertEqual(item.workflow, .depth)
        XCTAssertNil(item.request)
        XCTAssertEqual(generator.phase, .idle)
        XCTAssertTrue(FileManager.default.fileExists(atPath: item.imageURL.path))
    }
}
