import Foundation
import Observation

struct HistoryItem: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    let request: GenerationRequest
    let created: Date
    let seconds: Double

    var imageURL: URL { History.directory.appending(path: "\(id.uuidString).png") }
}

/// Finished images on disk: `<id>.png` next to `<id>.json` in Application Support.
@MainActor
@Observable
final class History {
    nonisolated static let directory = URL.applicationSupportDirectory.appending(path: "Iris/History", directoryHint: .isDirectory)

    private(set) var items: [HistoryItem] = []

    init() {
        load()
    }

    func add(png: Data, request: GenerationRequest, seconds: Double) throws -> HistoryItem {
        try FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        let item = HistoryItem(id: UUID(), request: request, created: .now, seconds: seconds)
        try png.write(to: item.imageURL, options: .atomic)
        try JSONEncoder().encode(item).write(to: metadataURL(item), options: .atomic)
        items.insert(item, at: 0)
        return item
    }

    func remove(_ item: HistoryItem) {
        try? FileManager.default.removeItem(at: item.imageURL)
        try? FileManager.default.removeItem(at: metadataURL(item))
        items.removeAll { $0.id == item.id }
    }

    func removeAll() {
        items.forEach(remove)
    }

    var byteCount: Int64 {
        items.reduce(0) { total, item in
            let size = (try? item.imageURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return total + Int64(size)
        }
    }

    private func metadataURL(_ item: HistoryItem) -> URL {
        Self.directory.appending(path: "\(item.id.uuidString).json")
    }

    private func load() {
        let files = (try? FileManager.default.contentsOfDirectory(at: Self.directory, includingPropertiesForKeys: nil)) ?? []
        items = files
            .filter { $0.pathExtension == "json" }
            .compactMap { try? JSONDecoder().decode(HistoryItem.self, from: Data(contentsOf: $0)) }
            .filter { FileManager.default.fileExists(atPath: $0.imageURL.path) }
            .sorted { $0.created > $1.created }
    }
}
