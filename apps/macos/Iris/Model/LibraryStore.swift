import Darwin
import Foundation

/// The library on disk:
///
///     <root>/Library.json      projects, conversations and entries
///     <root>/Images/<id>.png   the result of an entry
///     <root>/Inputs/<id>.png   the input of a depth or upscaling entry
///
/// Iris 0.1 kept `<root>/History/<id>.png` + `<id>.json` per image; `load()`
/// imports those into one conversation and removes the old folder only after
/// the new index is saved, so an interrupted import simply runs again.
///
/// A library that another running Iris holds, that a newer Iris wrote, or that
/// could not be read nor set aside opens read-only, so it is never overwritten.
struct LibraryStore: Sendable {
    let root: URL

    /// `~/Library/Application Support/Iris`; a `-libraryRoot <path>` launch argument (never a saved
    /// preference) points a development build elsewhere.
    static var defaultRoot: URL {
        let arguments = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        if let path = arguments["libraryRoot"] as? String, !path.isEmpty {
            return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        }
        return URL.applicationSupportDirectory.appending(path: "Iris", directoryHint: .isDirectory)
    }

    var indexURL: URL { root.appending(path: "Library.json") }
    var imagesDirectory: URL { root.appending(path: "Images", directoryHint: .isDirectory) }
    var inputsDirectory: URL { root.appending(path: "Inputs", directoryHint: .isDirectory) }
    var legacyDirectory: URL { root.appending(path: "History", directoryHint: .isDirectory) }

    func imageURL(_ entry: UUID) -> URL { imagesDirectory.appending(path: "\(entry.uuidString).png") }
    func inputURL(_ entry: UUID) -> URL { inputsDirectory.appending(path: "\(entry.uuidString).png") }

    struct Loaded {
        var snapshot: LibrarySnapshot
        /// What went wrong reading the library, for the user.
        var problem: String?
        var readOnly = false
    }

    /// Takes the library (an advisory `flock`, held until the lock is released or the
    /// process exits); nil if another Iris holds it.
    func lock() -> LibraryLock? {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let descriptor = open(root.appending(path: ".lock").path, O_CREAT | O_RDWR, 0o644)
        guard descriptor >= 0 else { return nil }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            return nil
        }
        return LibraryLock(descriptor: descriptor)
    }

    /// Reads the index, marks unfinished entries as interrupted, and imports Iris 0.1 history.
    /// With `readOnly`, nothing on disk is changed.
    func load(readOnly: Bool = false) -> Loaded {
        var loaded = readIndex(readOnly: readOnly)
        loaded.readOnly = loaded.readOnly || readOnly
        if !loaded.readOnly {
            for c in loaded.snapshot.conversations.indices {
                for e in loaded.snapshot.conversations[c].entries.indices
                where loaded.snapshot.conversations[c].entries[e].outcome == .running {
                    loaded.snapshot.conversations[c].entries[e].outcome = .interrupted
                }
            }
        }
        guard !loaded.readOnly else { return loaded }
        do {
            try importLegacyHistory(into: &loaded.snapshot)
        } catch {
            loaded.problem = loaded.problem ?? "Earlier images could not be imported: \(error.localizedDescription)"
        }
        return loaded
    }

    func save(_ snapshot: LibrarySnapshot) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(snapshot).write(to: indexURL, options: .atomic)
    }

    func writeImage(_ png: Data, for entry: UUID) throws {
        try FileManager.default.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
        try png.write(to: imageURL(entry), options: .atomic)
    }

    func writeInput(_ png: Data, for entry: UUID) throws {
        try FileManager.default.createDirectory(at: inputsDirectory, withIntermediateDirectories: true)
        try png.write(to: inputURL(entry), options: .atomic)
    }

    /// Removes the files of entries that are no longer in the library, including an
    /// Iris 0.1 original still in a History folder that could not be removed whole,
    /// which would otherwise be imported again.
    func removeFiles(of entries: [UUID]) {
        for id in entries {
            try? FileManager.default.removeItem(at: imageURL(id))
            try? FileManager.default.removeItem(at: inputURL(id))
            try? FileManager.default.removeItem(at: legacyDirectory.appending(path: "\(id.uuidString).png"))
            try? FileManager.default.removeItem(at: legacyDirectory.appending(path: "\(id.uuidString).json"))
        }
    }

    // MARK: - Private

    private struct Version: Decodable {
        let version: Int
    }

    private func readIndex(readOnly: Bool) -> Loaded {
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return Loaded(snapshot: LibrarySnapshot()) }
        let failure: Error
        do {
            let data = try Data(contentsOf: indexURL)
            if let version = try? JSONDecoder().decode(Version.self, from: data).version,
               version > LibrarySnapshot.currentVersion {
                return Loaded(
                    snapshot: LibrarySnapshot(), problem: "This library was saved by a newer version of Iris. Update Iris to open it; it was not changed.",
                    readOnly: true
                )
            }
            return Loaded(snapshot: try JSONDecoder().decode(LibrarySnapshot.self, from: data))
        } catch {
            failure = error
        }
        // keep the file for recovery rather than overwrite it on the next save
        let aside = root.appending(path: "Library-unreadable-\(UUID().uuidString).json")
        let images = "Images remain in \(imagesDirectory.path)."
        guard !readOnly, (try? FileManager.default.moveItem(at: indexURL, to: aside)) != nil else {
            return Loaded(
                snapshot: LibrarySnapshot(),
                problem: "The library could not be read (\(failure.localizedDescription)); it was left unchanged and Iris opened read-only. \(images)",
                readOnly: true
            )
        }
        return Loaded(
            snapshot: LibrarySnapshot(),
            problem: "The library could not be read (\(failure.localizedDescription)). It was kept as \(aside.lastPathComponent). \(images)"
        )
    }

    /// The Iris 0.1 record of one image.
    private struct LegacyItem: Decodable {
        let id: UUID
        let workflow: Workflow?
        let request: GenerationRequest?
        let source: String?
        let created: Date
        let seconds: Double
    }

    private func importLegacyHistory(into snapshot: inout LibrarySnapshot) throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: legacyDirectory.path) else { return }
        let known = Set(snapshot.conversations.flatMap { $0.entries.map(\.id) })
        let files = try fileManager.contentsOfDirectory(at: legacyDirectory, includingPropertiesForKeys: nil)
        let records = files.filter { $0.pathExtension == "json" }
        let items = records
            .compactMap { try? JSONDecoder().decode(LegacyItem.self, from: Data(contentsOf: $0)) }
            .sorted { $0.created < $1.created }
        // the old folder goes only when every file in it is accounted for
        var complete = items.count == records.count
            && files.allSatisfy { ["json", "png"].contains($0.pathExtension) || $0.lastPathComponent == ".DS_Store" }

        var imported: [Entry] = []
        for item in items {
            let legacyImage = legacyDirectory.appending(path: "\(item.id.uuidString).png")
            guard fileManager.fileExists(atPath: legacyImage.path) else { continue }  // Iris 0.1 showed none either
            if known.contains(item.id) { continue }
            try fileManager.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
            if !fileManager.fileExists(atPath: imageURL(item.id).path) {
                try fileManager.copyItem(at: legacyImage, to: imageURL(item.id))  // a clone on APFS
            }
            imported.append(Entry(
                id: item.id, created: item.created, workflow: item.workflow ?? .create,
                request: item.request, inputName: item.source, outcome: .done(seconds: item.seconds)
            ))
        }
        let ids = Set(items.map(\.id))
        complete = complete && files.filter { $0.pathExtension == "png" }.allSatisfy {
            UUID(uuidString: $0.deletingPathExtension().lastPathComponent).map(ids.contains) ?? false
        }
        if !imported.isEmpty {
            snapshot.conversations.append(Conversation(title: "Earlier images", projectID: nil, entries: imported))
        }
        try save(snapshot)
        if complete {
            try fileManager.removeItem(at: legacyDirectory)
        }
    }
}

/// Holds a library's lock file open; closing it releases the lock.
final class LibraryLock: Sendable {
    private let descriptor: Int32

    fileprivate init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    deinit {
        close(descriptor)
    }
}
