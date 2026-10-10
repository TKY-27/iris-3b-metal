import XCTest

@testable import Iris

@MainActor
final class LibraryTests: XCTestCase {
    private var root: URL!

    override func setUp() {
        root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
    }

    private func request(_ prompt: String) -> GenerationRequest {
        GenerationRequest(prompt: prompt, width: 1024, height: 1024, steps: 100, cfgScale: 3, seed: 1, negativePrompt: "")
    }

    func testChangesSurviveARelaunch() throws {
        var library: Library? = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library?.createProject(named: "Coasts"))
        let conversation = try XCTUnwrap(library?.append(Entry(workflow: .create, request: request("lighthouse")), to: nil, project: project))
        library?.append(Entry(workflow: .create, request: request("harbor"), outcome: .done(seconds: 3)), to: conversation)
        library?.rename(conversation: conversation, to: "Lighthouses")
        library?.flush()
        library = nil  // "quits", releasing the lock

        let reopened = Library(store: LibraryStore(root: root))
        XCTAssertEqual(reopened.projects.map(\.name), ["Coasts"])
        let saved = try XCTUnwrap(reopened.conversation(conversation))
        XCTAssertEqual(saved.title, "Lighthouses")
        XCTAssertEqual(saved.projectID, project)
        XCTAssertEqual(saved.entries.map(\.request?.prompt), ["lighthouse", "harbor"])
        // the first entry never finished before the "quit"
        XCTAssertEqual(saved.entries.map(\.outcome), [.interrupted, .done(seconds: 3)])
    }

    func testDeletingAConversationRemovesItsFilesButDeletingAProjectKeepsConversations() async throws {
        let library = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library.createProject(named: "Tests"))
        let entry = Entry(workflow: .create, request: request("fox"), outcome: .done(seconds: 1))
        let conversation = library.append(entry, to: nil, project: project)
        try await library.write(Data([1, 2, 3]), for: entry.id)

        library.delete(project: project)
        XCTAssertNil(library.conversation(conversation)?.projectID)
        XCTAssertNotNil(library.conversation(conversation))

        library.delete(conversation: conversation)
        library.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: library.imageURL(entry.id).path))
    }

    func testIris01HistoryIsImportedWithoutLoss() throws {
        let legacy = root.appending(path: "History", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        let ids = [UUID(), UUID()]
        let records = [
            // the Iris 0.1 format: no workflow key for images made before depth and upscaling existed
            #"{"id":"\#(ids[0])","request":{"cfg_scale":3,"seed":5,"prompt":"a fox","height":1024,"width":1024,"negative_prompt":"","steps":100},"created":800000000,"seconds":400}"#,
            #"{"id":"\#(ids[1])","workflow":"upscale","source":"photo.jpg","created":800000100,"seconds":20}"#,
        ]
        for (id, record) in zip(ids, records) {
            try Data(record.utf8).write(to: legacy.appending(path: "\(id).json"))
            try Data([UInt8(id.hashValue & 0xff)]).write(to: legacy.appending(path: "\(id).png"))
        }

        var library: Library? = Library(store: LibraryStore(root: root))
        let conversation = try XCTUnwrap(library?.conversations.first)
        XCTAssertEqual(conversation.entries.map(\.id), ids)  // oldest first
        XCTAssertEqual(conversation.entries.map(\.workflow), [.create, .upscale])
        XCTAssertEqual(conversation.entries[0].request?.seed, 5)
        XCTAssertEqual(conversation.entries[1].inputName, "photo.jpg")
        for id in ids {
            XCTAssertTrue(FileManager.default.fileExists(atPath: LibraryStore(root: root).imageURL(id).path))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacy.path))
        library = nil

        // a second launch neither duplicates nor loses anything
        let reopened = Library(store: LibraryStore(root: root))
        XCTAssertEqual(reopened.conversations.flatMap(\.entries).map(\.id), ids)
    }

    func testUnaccountedLegacyFilesKeepTheOldFolderAndDeletedImagesStayDeleted() throws {
        let legacy = root.appending(path: "History", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: legacy.appending(path: "\(UUID()).json"))
        let id = UUID()
        try Data(#"{"id":"\#(id)","created":800000000,"seconds":1}"#.utf8).write(to: legacy.appending(path: "\(id).json"))
        try Data([1]).write(to: legacy.appending(path: "\(id).png"))

        var library: Library? = Library(store: LibraryStore(root: root))
        XCTAssertTrue(FileManager.default.fileExists(atPath: legacy.path))
        library?.delete(entry: id)
        library?.flush()
        library = nil

        // the original went with the entry, so it is not imported again
        let reopened = Library(store: LibraryStore(root: root))
        XCTAssertFalse(reopened.conversations.flatMap(\.entries).contains { $0.id == id })
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacy.appending(path: "\(id).png").path))
    }

    func testASecondInstanceIsReadOnly() throws {
        let first = Library(store: LibraryStore(root: root))
        first.createProject(named: "Mine")
        first.flush()
        let second = Library(store: LibraryStore(root: root))
        XCTAssertTrue(second.isReadOnly)
        XCTAssertNotNil(second.problem)
        XCTAssertNil(second.createProject(named: "Theirs"))
        XCTAssertFalse(first.isReadOnly)
    }

    func testALibraryFromANewerVersionIsLeftAlone() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let index = root.appending(path: "Library.json")
        let newer = Data(#"{"version":2,"projects":[],"conversations":[],"albums":[]}"#.utf8)
        try newer.write(to: index)

        let library = Library(store: LibraryStore(root: root))
        XCTAssertTrue(library.isReadOnly)
        library.createProject(named: "x")
        library.flush()
        XCTAssertEqual(try Data(contentsOf: index), newer)
    }

    func testAnUnreadableIndexIsKeptAside() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("{".utf8).write(to: root.appending(path: "Library.json"))

        let library = Library(store: LibraryStore(root: root))
        XCTAssertNotNil(library.problem)
        let files = try FileManager.default.contentsOfDirectory(atPath: root.path)
        XCTAssertTrue(files.contains { $0.hasPrefix("Library-unreadable-") })
    }
}
