import Foundation
import Observation
import os

/// Projects, conversations and entries, saved to `LibraryStore` after every change.
@MainActor
@Observable
final class Library {
    let store: LibraryStore
    private(set) var projects: [Project]
    private(set) var conversations: [Conversation]
    /// A load or save problem to show the user once.
    var problem: String?
    /// Nothing is changed or saved; see `LibraryStore`.
    let isReadOnly: Bool

    @ObservationIgnored private let lock: LibraryLock?
    @ObservationIgnored private let writer = DispatchQueue(label: "Iris.LibraryWriter", qos: .utility)
    @ObservationIgnored private let logger = Logger(subsystem: "io.github.tky27.iris", category: "Library")

    init(store: LibraryStore = LibraryStore(root: LibraryStore.defaultRoot)) {
        self.store = store
        lock = store.lock()
        let loaded = store.load(readOnly: lock == nil)
        projects = loaded.snapshot.projects
        conversations = loaded.snapshot.conversations
        isReadOnly = loaded.readOnly
        problem = lock != nil ? loaded.problem : "Iris is already running with this library, so this window shows it read-only."
    }

    // MARK: - Queries

    func conversation(_ id: UUID?) -> Conversation? {
        conversations.first { $0.id == id }
    }

    func conversations(in project: UUID?) -> [Conversation] {
        conversations.filter { $0.projectID == project }.sorted { $0.updated > $1.updated }
    }

    func entry(_ id: UUID) -> (conversation: Conversation, entry: Entry)? {
        for conversation in conversations {
            if let entry = conversation.entries.first(where: { $0.id == id }) { return (conversation, entry) }
        }
        return nil
    }

    /// Finished results, newest first.
    var finishedEntries: [(conversation: UUID, entry: Entry)] {
        conversations
            .flatMap { c in c.entries.filter(\.isDone).map { (conversation: c.id, entry: $0) } }
            .sorted { $0.entry.created > $1.entry.created }
    }

    func imageURL(_ entry: UUID) -> URL { store.imageURL(entry) }
    func inputURL(_ entry: UUID) -> URL { store.inputURL(entry) }

    // MARK: - Conversations

    /// Adds `entry` to a conversation, or to a new one titled after it when `conversation` is nil.
    @discardableResult
    func append(_ entry: Entry, to conversation: UUID?, project: UUID? = nil) -> UUID {
        if let index = conversations.firstIndex(where: { $0.id == conversation }) {
            conversations[index].entries.append(entry)
            conversations[index].updated = entry.created
            save()
            return conversations[index].id
        }
        let created = Conversation(title: Conversation.title(for: entry), projectID: project, entries: [entry])
        conversations.append(created)
        save()
        return created.id
    }

    func rename(conversation id: UUID, to title: String) {
        guard !isReadOnly else { return }
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].title = title
        save()
    }

    func move(conversation id: UUID, to project: UUID?) {
        guard !isReadOnly else { return }
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].projectID = project
        save()
    }

    func delete(conversation id: UUID) {
        guard !isReadOnly else { return }
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let removed = conversations.remove(at: index).entries.map(\.id)
        save(thenRemoveFilesOf: removed)
    }

    // MARK: - Entries

    func setOutcome(_ outcome: Entry.Outcome, of id: UUID) {
        for c in conversations.indices {
            if let e = conversations[c].entries.firstIndex(where: { $0.id == id }) {
                conversations[c].entries[e].outcome = outcome
                save()
                return
            }
        }
    }

    func delete(entry id: UUID) {
        guard !isReadOnly else { return }
        guard let c = conversations.firstIndex(where: { $0.entries.contains { $0.id == id } }) else { return }
        conversations[c].entries.removeAll { $0.id == id }
        save(thenRemoveFilesOf: [id])
    }

    /// Writes an entry's result or input PNG off the main thread.
    func write(_ png: Data, for entry: UUID, asInput: Bool = false) async throws {
        let store = store
        try await Task.detached(priority: .userInitiated) {
            if asInput {
                try store.writeInput(png, for: entry)
            } else {
                try store.writeImage(png, for: entry)
            }
        }.value
    }

    // MARK: - Projects

    @discardableResult
    func createProject(named name: String) -> UUID? {
        guard !isReadOnly else { return nil }
        let project = Project(name: name)
        projects.append(project)
        projects.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        save()
        return project.id
    }

    func rename(project id: UUID, to name: String) {
        guard !isReadOnly else { return }
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let index = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[index].name = name
        projects.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        save()
    }

    /// Deletes the project only; its conversations move to the unfiled list.
    func delete(project id: UUID) {
        guard !isReadOnly else { return }
        projects.removeAll { $0.id == id }
        for index in conversations.indices where conversations[index].projectID == id {
            conversations[index].projectID = nil
        }
        save()
    }

    // MARK: - Storage

    var entryCount: Int { conversations.reduce(0) { $0 + $1.entries.count } }

    /// Bytes of all result and input images.
    func byteCount() async -> Int64 {
        let directories = [store.imagesDirectory, store.inputsDirectory]
        return await Task.detached(priority: .utility) {
            directories.reduce(Int64(0)) { total, directory in
                let files = (try? FileManager.default.contentsOfDirectory(
                    at: directory, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]
                )) ?? []
                return total + files.reduce(0) {
                    $0 + Int64((try? $1.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0)
                }
            }
        }.value
    }

    /// Deletes every conversation, keeping projects and any running entry's conversation.
    func deleteAllConversations(except running: UUID?) {
        guard !isReadOnly else { return }
        let removed = conversations.filter { $0.id != running }.flatMap { $0.entries.map(\.id) }
        conversations.removeAll { $0.id != running }
        save(thenRemoveFilesOf: removed)
    }

    /// Snapshots on the main actor, which keeps the order of changes, then
    /// encodes and writes on a serial queue; files of removed entries are
    /// deleted only once the saved index no longer lists them.
    private func save(thenRemoveFilesOf removed: [UUID] = []) {
        guard !isReadOnly else { return }
        let snapshot = LibrarySnapshot(projects: projects, conversations: conversations)
        let store = store
        writer.async { [weak self, logger] in
            do {
                try store.save(snapshot)
                store.removeFiles(of: removed)
            } catch {
                logger.error("Saving the library failed: \(error.localizedDescription, privacy: .public)")
                let message = "Iris could not save the library: \(error.localizedDescription)"
                Task { @MainActor in self?.problem = message }
            }
        }
    }

    /// Waits for pending saves, e.g. before the app quits.
    nonisolated func flush() {
        writer.sync {}
    }
}
