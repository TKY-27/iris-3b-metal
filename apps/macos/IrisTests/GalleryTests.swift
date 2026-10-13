import XCTest

@testable import Iris

/// What the gallery lists, in what order and under which day, and how the arrow keys move.
final class GalleryTests: XCTestCase {
    private let start = Date(timeIntervalSinceReferenceDate: 813_000_000)

    private func entry(
        _ prompt: String, at offset: TimeInterval, seed: Int = 1, negative: String = "",
        outcome: Entry.Outcome = .done(seconds: 1)
    ) -> Entry {
        Entry(
            created: start.addingTimeInterval(offset), workflow: .create,
            request: GenerationRequest(
                prompt: prompt, width: 1024, height: 1024, steps: 20, cfgScale: 3, seed: seed, negativePrompt: negative
            ),
            outcome: outcome
        )
    }

    func testNewestFirstWithEachBatchInTheOrderItWasMade() {
        // a batch of three, then a variation of it in the same conversation, then another request
        let batch = (0..<3).map { entry("fox \($0)", at: Double($0) * 0.001) }
        let variation = entry("fox again", at: 60)
        let later = entry("heron", at: 120)
        let conversations = [
            Conversation(title: "fox", projectID: nil, entries: batch + [variation]),
            Conversation(title: "heron", projectID: nil, entries: [later]),
        ]
        let items = Gallery.items(of: conversations, in: .allImages)
        XCTAssertEqual(items.map(\.entry.title), ["heron", "fox again", "fox 0", "fox 1", "fox 2"])
        XCTAssertEqual(items.first?.conversation, conversations[1].id)
    }

    func testProjectsSearchAndRequestsWithoutImages() {
        let project = UUID()
        let filed = entry("a lighthouse at dusk", at: 0, seed: 42)
        let stopped = entry("a lighthouse at dawn", at: 10, outcome: .cancelled)
        let failed = entry("harbor", at: 20, negative: "fog", outcome: .failed("no"))
        let conversations = [
            Conversation(title: "a", projectID: project, entries: [filed]),
            Conversation(title: "b", projectID: nil, entries: [stopped, failed]),
        ]
        // stopped and failed requests stay, to be tried again or deleted
        XCTAssertEqual(Gallery.items(of: conversations, in: .allImages).map(\.id), [failed.id, stopped.id, filed.id])
        XCTAssertEqual(Gallery.items(of: conversations, in: .project(project)).map(\.id), [filed.id])
        XCTAssertEqual(Gallery.items(of: conversations, in: .project(project)).first?.project, project)
        XCTAssertEqual(Gallery.items(of: conversations, in: .allImages, matching: " Lighthouse ").map(\.id), [stopped.id, filed.id])
        XCTAssertEqual(Gallery.items(of: conversations, in: .allImages, matching: "42").map(\.id), [filed.id])
        XCTAssertEqual(Gallery.items(of: conversations, in: .allImages, matching: "fog").map(\.id), [])  // not what it avoids
        XCTAssertEqual(Gallery.items(of: conversations, in: .allImages, matching: "4").map(\.id), [])  // seeds match whole
    }

    /// One image of a batch filed in a project keeps its place among the others in All Images.
    func testAMovedImageStaysInItsBatch() {
        let batch = (0..<4).map { entry("fox \($0)", at: Double($0) * 0.001) }
        let conversations = [
            Conversation(title: "fox", projectID: nil, entries: [batch[0], batch[1], batch[3]]),
            Conversation(title: "fox", projectID: UUID(), entries: [batch[2]]),
            Conversation(title: "later", projectID: nil, entries: [entry("heron", at: 30)]),
        ]
        XCTAssertEqual(Gallery.items(of: conversations, in: .allImages).map(\.entry.title), ["heron", "fox 0", "fox 1", "fox 2", "fox 3"])
    }

    func testSectionsAreDays() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "UTC"))
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 10, day: 13, hour: 9)))
        func item(daysAgo: Double) -> GalleryItem {
            let entry = Entry(created: now.addingTimeInterval(-daysAgo * 86_400), workflow: .depth, inputName: "x.png", outcome: .done(seconds: 1))
            return GalleryItem(entry: entry, conversation: UUID(), project: nil)
        }
        let sections = Gallery.sections(
            [item(daysAgo: 0), item(daysAgo: 0.1), item(daysAgo: 1), item(daysAgo: 3), item(daysAgo: 3), item(daysAgo: 400)],
            now: now, calendar: calendar
        )
        XCTAssertEqual(sections.map(\.items.count), [2, 1, 2, 1])
        XCTAssertEqual(Array(sections.map(\.title).prefix(2)), ["Today", "Yesterday"])
        XCTAssertEqual(Set(sections.map(\.title)).count, 4)
        XCTAssertTrue(try XCTUnwrap(sections.last?.title).contains("2025"))  // another year says so
    }

    func testArrowKeysMoveAlongRowsAndAcrossDays() {
        func section(_ title: String, _ count: Int) -> GallerySection {
            GallerySection(title: title, items: (0..<count).map { index in
                GalleryItem(entry: entry("\(title)\(index)", at: Double(index)), conversation: UUID(), project: nil)
            })
        }
        // three columns: today has two rows (3 + 2), yesterday one (2)
        let sections = [section("t", 5), section("y", 2)]
        let ids = sections.flatMap { $0.items.map(\.id) }
        func move(_ from: Int?, _ move: Gallery.Move) -> Int? {
            Gallery.neighbor(of: from.map { ids[$0] }, in: sections, columns: 3, moving: move).flatMap(ids.firstIndex(of:))
        }
        XCTAssertEqual(move(nil, .down), 0)  // nothing selected: the first
        XCTAssertEqual(move(0, .left), 0)
        XCTAssertEqual(move(2, .right), 3)  // on to the next row
        XCTAssertEqual(move(2, .down), 4)  // the shorter row's last
        XCTAssertEqual(move(4, .down), 6)  // into the next day
        XCTAssertEqual(move(5, .up), 3)
        XCTAssertEqual(move(1, .up), 1)
        XCTAssertEqual(move(6, .right), 6)
    }
}
