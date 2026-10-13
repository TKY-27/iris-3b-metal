import XCTest

@testable import Iris

@MainActor
final class LibraryTests: XCTestCase {
    private var root: URL!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    }

    override func tearDown() async throws {
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
        library?.flush()
        library = nil  // "quits", releasing the lock

        let reopened = Library(store: LibraryStore(root: root))
        XCTAssertEqual(reopened.projects.map(\.name), ["Coasts"])
        let saved = try XCTUnwrap(reopened.conversation(conversation))
        XCTAssertEqual(saved.title, "lighthouse")
        XCTAssertEqual(saved.projectID, project)
        XCTAssertEqual(saved.entries.map(\.request?.prompt), ["lighthouse", "harbor"])
        // the first entry never finished before the "quit"
        XCTAssertEqual(saved.entries.map(\.outcome), [.interrupted, .done(seconds: 3)])
    }

    func testDeletingAnImageRemovesItsFilesButDeletingAProjectKeepsImages() async throws {
        let library = Library(store: LibraryStore(root: root))
        let project = try XCTUnwrap(library.createProject(named: "Tests"))
        let entry = Entry(workflow: .create, request: request("fox"), outcome: .done(seconds: 1))
        let conversation = try XCTUnwrap(library.append(entry, to: nil, project: project))
        try await library.write(Data([1, 2, 3]), for: entry.id)

        library.delete(project: project)
        XCTAssertNil(library.conversation(conversation)?.projectID)
        XCTAssertNotNil(library.conversation(conversation))

        library.delete(entry: entry.id)
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

    func testASecondInstanceKeepsRunningEntriesVisible() throws {
        let first = Library(store: LibraryStore(root: root))
        let entry = Entry(workflow: .create, request: request("fox"))
        let conversation = first.append(entry, to: nil)
        first.flush()

        let second = Library(store: LibraryStore(root: root))
        XCTAssertTrue(second.isReadOnly)
        XCTAssertEqual(second.conversation(conversation)?.entries.first?.outcome, .running)
    }

    func testReadOnlyLibraryRejectsEntryAndImageChanges() async throws {
        let first = Library(store: LibraryStore(root: root))
        let entry = Entry(workflow: .create, request: request("fox"))
        let conversation = try XCTUnwrap(first.append(entry, to: nil))
        try await first.write(Data([1]), for: entry.id)
        first.flush()
        let index = try Data(contentsOf: first.store.indexURL)
        let second = Library(store: LibraryStore(root: root))

        XCTAssertNil(second.append(Entry(workflow: .depth), to: conversation))
        second.setOutcome(.cancelled, of: entry.id)
        XCTAssertEqual(second.conversation(conversation)?.entries, [entry])
        for asInput in [false, true] {
            do {
                try await second.write(Data([2]), for: entry.id, asInput: asInput)
                XCTFail("Read-only writes must fail")
            } catch {
                XCTAssertEqual((error as? CocoaError)?.code, .fileWriteNoPermission)
            }
        }
        second.flush()
        XCTAssertEqual(try Data(contentsOf: first.store.indexURL), index)
        XCTAssertEqual(try Data(contentsOf: first.imageURL(entry.id)), Data([1]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.inputURL(entry.id).path))
    }

    func testMigrationRestoresIndexedImagesBeforeRemovingHistory() throws {
        let store = LibraryStore(root: root)
        let entry = Entry(workflow: .create, request: request("fox"), outcome: .done(seconds: 1))
        let conversation = Conversation(title: "fox", projectID: nil, entries: [entry])
        try store.save(LibrarySnapshot(conversations: [conversation]))
        try FileManager.default.createDirectory(at: store.legacyDirectory, withIntermediateDirectories: true)
        let record = #"{"id":"\#(entry.id)","created":800000000,"seconds":1}"#
        try Data(record.utf8).write(to: store.legacyDirectory.appending(path: "\(entry.id).json"))
        let png = Data([1, 2, 3])
        try png.write(to: store.legacyDirectory.appending(path: "\(entry.id).png"))

        let loaded = store.load()
        XCTAssertNil(loaded.problem)
        XCTAssertEqual(loaded.snapshot.conversations, [conversation])
        XCTAssertEqual(try Data(contentsOf: store.imageURL(entry.id)), png)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.legacyDirectory.path))
    }

    func testMigrationImportsDuplicateLegacyIDsOnce() throws {
        let store = LibraryStore(root: root)
        try FileManager.default.createDirectory(at: store.legacyDirectory, withIntermediateDirectories: true)
        let id = UUID()
        let record = #"{"id":"\#(id)","created":800000000,"seconds":1}"#
        for name in [id.uuidString, "duplicate"] {
            try Data(record.utf8).write(to: store.legacyDirectory.appending(path: "\(name).json"))
        }
        try Data([1]).write(to: store.legacyDirectory.appending(path: "\(id).png"))

        let loaded = store.load()
        XCTAssertNil(loaded.problem)
        XCTAssertEqual(loaded.snapshot.conversations.flatMap(\.entries).map(\.id), [id])
    }

    func testVersionOneHistoryPreservesImagesAndMigratesOnSave() throws {
        let store = LibraryStore(root: root)
        let entry = Entry(workflow: .create, request: request("legacy fox"), outcome: .done(seconds: 2))
        let conversation = Conversation(title: "Legacy chat", projectID: nil, entries: [entry])
        var snapshot = LibrarySnapshot(conversations: [conversation])
        snapshot.version = 1
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(snapshot)) as? [String: Any])
        var chats = try XCTUnwrap(json["conversations"] as? [[String: Any]])
        var entries = try XCTUnwrap(chats[0]["entries"] as? [[String: Any]])
        var legacyRequest = try XCTUnwrap(entries[0]["request"] as? [String: Any])
        legacyRequest.removeValue(forKey: "model")
        entries[0]["request"] = legacyRequest
        chats[0]["entries"] = entries
        json["conversations"] = chats
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: json).write(to: store.indexURL)
        let image = Data([1, 2, 3])
        try store.writeImage(image, for: entry.id)

        let library = Library(store: store)
        XCTAssertFalse(library.isReadOnly)
        XCTAssertEqual(library.conversations.first?.entries.first?.request?.model, .original)
        XCTAssertEqual(library.conversations.first?.entries.first?.id, entry.id)
        library.createProject(named: "New project")
        library.flush()
        let saved = try JSONDecoder().decode(LibrarySnapshot.self, from: Data(contentsOf: store.indexURL))
        XCTAssertEqual(saved.version, LibrarySnapshot.currentVersion)
        XCTAssertEqual(saved.conversations, [conversation])
        XCTAssertEqual(try Data(contentsOf: store.imageURL(entry.id)), image)
    }

    func testALibraryFromANewerVersionIsLeftAlone() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let index = root.appending(path: "Library.json")
        let newer = Data(#"{"version":\#(LibrarySnapshot.currentVersion + 1),"projects":[],"conversations":[],"albums":[]}"#.utf8)
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
