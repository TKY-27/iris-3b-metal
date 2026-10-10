import Metal
import XCTest

@testable import Iris

@MainActor
final class CanvasTests: XCTestCase {
    /// The canvas shader is compiled at runtime, so the build cannot catch a mistake in it.
    func testCanvasShaderCompiles() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        XCTAssertNoThrow(try FieldPipeline.compile(on: device))
    }
}
