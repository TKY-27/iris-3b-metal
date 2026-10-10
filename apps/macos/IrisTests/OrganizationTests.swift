import XCTest

@testable import Iris

/// Pins, projects and what the window shows.
@MainActor
final class OrganizationTests: XCTestCase {
    private var root: URL!

    override func setUp() {
        root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
    }

    private func entry(_ prompt: String, workflow: Workflow = .create, done: Bool = true) -> Entry {
        Entry(
            workflow: workflow,
            request: workflow == .create ? GenerationRequest(
                prompt: prompt, width: 1024, height: 1024, steps: 20, cfgScale: 3, seed: 1, negativePrompt: ""
            ) : nil,
            inputName: workflow == .create ? nil : "photo.png",
            outcome: done ? .done(seconds: 1) : .failed("no")
        )
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

    func testPinsAreKeptAndOrderedNewestFirst() throws {
        var library: Library? = Library(store: LibraryStore(root: root))
        let first = try XCTUnwrap(library?.append(entry("a"), to: nil))
        let second = try XCTUnwrap(library?.append(entry("b"), to: nil))
        library?.setPinned(true, conversation: first)
        Thread.sleep(forTimeInterval: 0.01)
        library?.setPinned(true, conversation: second)
        XCTAssertEqual(library?.pinnedConversations.map(\.id), [second, first])
        XCTAssertEqual(library?.recentConversations.map(\.id), [])  // pinned chats leave the recent list
        library?.setPinned(false, conversation: first)
        library?.flush()
        library = nil

        let reopened = Library(store: LibraryStore(root: root))
        XCTAssertEqual(reopened.pinnedConversations.map(\.id), [second])
        XCTAssertEqual(reopened.recentConversations.map(\.id), [first])
    }

    func testAPinnedChatInAProjectStaysInTheProject() throws {
        let library = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library.createProject(named: "Coasts"))
        let chat = library.append(entry("a"), to: nil, project: project)
        library.setPinned(true, conversation: chat)
        XCTAssertEqual(library.conversations(in: project).map(\.id), [chat])
        XCTAssertEqual(library.pinnedConversations.map(\.id), [chat])
        library.flush()  // before tearDown removes the folder
    }

    func testThumbnailsShowTheNewestResult() {
        var conversation = Conversation(title: "x", projectID: nil, entries: [entry("a"), entry("b", workflow: .depth)])
        let newest = conversation.entries[1].id
        XCTAssertEqual(conversation.latestResult?.id, newest)
        conversation.entries.append(entry("c", done: false))
        XCTAssertEqual(conversation.latestResult?.id, newest)  // a failure has no image
    }

    func testOpeningAProjectStartsChatsInItAndNewChatLeavesIt() throws {
        let suite = "io.github.tky27.iris.tests"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let library = Library(store: LibraryStore(root: root))
        let workspace = Workspace(library: library, defaults: defaults)
        XCTAssertEqual(workspace.selection, .newChat)
        let project = try XCTUnwrap(library.createProject(named: "Coasts"))

        workspace.selection = .project(project)
        XCTAssertEqual(workspace.draftProject, project)
        XCTAssertNil(workspace.conversationID)
        workspace.selection = .newChat
        XCTAssertNil(workspace.draftProject)
        workspace.newConversation(in: project)
        XCTAssertEqual(workspace.selection, .newChat)
        XCTAssertEqual(workspace.draftProject, project)

        // leaving the project for All Images: New Chat there is not filed in it
        workspace.selection = .project(project)
        workspace.selection = .allImages
        workspace.newConversationHere(library: library)
        XCTAssertNil(workspace.draftProject)
        // inside a project's chat, New Chat stays in the project
        let chat = library.append(Entry(workflow: .create, request: nil, outcome: .cancelled), to: nil, project: project)
        workspace.show(chat)
        workspace.newConversationHere(library: library)
        XCTAssertEqual(workspace.draftProject, project)
        library.flush()
    }
}
