import XCTest

@testable import Iris

/// Projects, filing images in them, and what the window shows.
@MainActor
final class OrganizationTests: XCTestCase {
    private var root: URL!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func entry(_ prompt: String, workflow: Workflow = .create, outcome: Entry.Outcome = .done(seconds: 1)) -> Entry {
        Entry(
            workflow: workflow,
            request: workflow == .create ? GenerationRequest(
                prompt: prompt, width: 1024, height: 1024, steps: 20, cfgScale: 3, seed: 1, negativePrompt: ""
            ) : nil,
            inputName: workflow == .create ? nil : "photo.png",
            outcome: outcome
        )
    }

    private func scratchDefaults() throws -> UserDefaults {
        let suite = "io.github.tky27.iris.tests"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        addTeardownBlock { defaults.removePersistentDomain(forName: suite) }
        return defaults
    }

    func testALibraryWithoutPinsStillOpens() throws {
        // Library.json as Iris 0.2 wrote it: conversations have no "pinned" key
        let json = """
        {"version":1,"projects":[],"conversations":[{"id":"5CBA716E-9FBE-48F5-B838-C4054C323CF7",
        "title":"Lighthouse","created":813345345.9,"updated":813345345.9,"entries":[]}]}
        """
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: root.appending(path: "Library.json"))
        let library = Library(store: LibraryStore(root: root))
        XCTAssertNil(library.problem)
        XCTAssertFalse(library.isReadOnly)
        XCTAssertEqual(library.conversations.map(\.title), ["Lighthouse"])
        XCTAssertNil(library.conversations.first?.pinned)
    }

    /// Pins are no longer shown, but a library saved now keeps those of Iris 0.3 and 0.4.
    func testPinsOfEarlierVersionsSurviveASave() throws {
        let json = """
        {"version":1,"projects":[],"conversations":[{"id":"5CBA716E-9FBE-48F5-B838-C4054C323CF7",
        "title":"Lighthouse","created":813345345.9,"updated":813345345.9,"pinned":813345400,"entries":[]}]}
        """
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: root.appending(path: "Library.json"))
        var library: Library? = Library(store: LibraryStore(root: root))
        library?.append(entry("a"), to: nil)
        library?.flush()
        library = nil

        let reopened = Library(store: LibraryStore(root: root))
        XCTAssertEqual(reopened.conversations.count, 2)
        XCTAssertEqual(reopened.conversations.first?.pinned, Date(timeIntervalSinceReferenceDate: 813345400))
    }

    func testMovingOneImageOfABatchLeavesTheOthers() throws {
        let library = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library.createProject(named: "Coasts"))
        let (first, second) = (entry("a"), entry("a"))
        let batch = try XCTUnwrap(library.append(first, to: nil))
        library.append(second, to: batch)

        library.move(entry: second.id, to: project)
        XCTAssertEqual(library.conversation(batch)?.entries.map(\.id), [first.id])
        XCTAssertNil(library.conversation(batch)?.projectID)
        let moved = try XCTUnwrap(library.entry(second.id))
        XCTAssertEqual(moved.conversation.projectID, project)
        XCTAssertEqual(moved.entry, second)  // its request, outcome and date are kept

        // alone in its conversation, it moves with it
        library.move(entry: second.id, to: nil)
        XCTAssertEqual(library.entry(second.id)?.conversation.id, moved.conversation.id)
        XCTAssertNil(library.entry(second.id)?.conversation.projectID)
        library.flush()
    }

    func testARunningEntryStaysWhereTheGeneratorKnowsIt() throws {
        let library = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library.createProject(named: "Coasts"))
        let running = entry("a", outcome: .running)
        let conversation = try XCTUnwrap(library.append(running, to: nil))
        library.append(entry("a"), to: conversation)
        library.move(entry: running.id, to: project)
        XCTAssertEqual(library.entry(running.id)?.conversation.id, conversation)
        XCTAssertNil(library.entry(running.id)?.conversation.projectID)
        library.flush()
    }

    func testDeletingAllImagesKeepsOnlyWhatIsStillBeingMade() throws {
        let library = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library.createProject(named: "Coasts"))
        // an old chat of many images, one of them now being made again
        let (done, running, queued) = (entry("a"), entry("a", outcome: .running), entry("a", outcome: .running))
        let chat = try XCTUnwrap(library.append(done, to: nil, project: project))
        library.append(running, to: chat)
        library.append(queued, to: chat)
        library.append(entry("b"), to: nil)

        library.deleteAllImages()
        XCTAssertEqual(library.conversations.map(\.id), [chat])
        XCTAssertEqual(library.conversation(chat)?.entries.map(\.id), [running.id, queued.id])
        XCTAssertEqual(library.projects.map(\.id), [project])
        library.delete(entry: running.id)  // the generator's until it ends
        XCTAssertNotNil(library.entry(running.id))
        library.flush()
    }

    func testDeletingTheLastImageRemovesItsConversation() throws {
        let library = Library(store: LibraryStore(root: root))
        let (first, second) = (entry("a"), entry("b"))
        let conversation = try XCTUnwrap(library.append(first, to: nil))
        library.append(second, to: conversation)
        library.delete(entry: first.id)
        XCTAssertNotNil(library.conversation(conversation))
        library.delete(entry: second.id)
        XCTAssertNil(library.conversation(conversation))
        library.flush()
    }

    func testOnlyImagesOfTheLibraryAreFoundByTheirFile() throws {
        let library = Library(store: LibraryStore(root: root))
        let image = entry("a")
        library.append(image, to: nil)
        XCTAssertEqual(library.entryID(forImage: library.imageURL(image.id)), image.id)
        XCTAssertNil(library.entryID(forImage: library.inputURL(image.id)))
        XCTAssertNil(library.entryID(forImage: library.imageURL(UUID())))
        XCTAssertNil(library.entryID(forImage: URL(fileURLWithPath: "/tmp/\(image.id.uuidString).png")))
        library.flush()
    }

    func testNewImagesGoToTheShownProjectAndRevealShowsAnImage() throws {
        let library = Library(store: LibraryStore(root: root))
        let workspace = Workspace(defaults: try scratchDefaults())
        XCTAssertEqual(workspace.selection, .allImages)
        XCTAssertNil(workspace.project)
        let project = try XCTUnwrap(library.createProject(named: "Coasts"))
        let filed = entry("a")
        library.append(filed, to: nil, project: project)
        let unfiled = entry("b")
        library.append(unfiled, to: nil)

        workspace.selectedEntry = unfiled.id
        workspace.selection = .project(project)
        XCTAssertEqual(workspace.project, project)
        XCTAssertNil(workspace.selectedEntry)  // it is not among the shown images
        workspace.selection = nil  // a click on the sidebar's empty space
        XCTAssertEqual(workspace.selection, .project(project))

        // a click on the sidebar's empty space keeps the shown project and the selected image
        workspace.selectedEntry = filed.id
        workspace.selection = nil
        XCTAssertEqual(workspace.selection, .project(project))
        XCTAssertEqual(workspace.selectedEntry, filed.id)

        // an image elsewhere: shown where it is kept
        workspace.reveal(unfiled.id, library: library)
        XCTAssertEqual(workspace.selection, .allImages)
        XCTAssertEqual(workspace.selectedEntry, unfiled.id)
        // All Images shows every image, so it stays
        workspace.reveal(filed.id, library: library)
        XCTAssertEqual(workspace.selection, .allImages)
        XCTAssertEqual(workspace.selectedEntry, filed.id)
        library.flush()
    }
}
