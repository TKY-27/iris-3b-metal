import Foundation
import Observation

enum Workflow: String, Codable, CaseIterable, Identifiable, Sendable {
    case create, depth, upscale

    var id: Self { self }

    var title: String {
        switch self {
        case .create: "Create"
        case .depth: "Depth"
        case .upscale: "Upscale"
        }
    }
}

struct HistoryItem: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    let workflow: Workflow
    /// The generation settings, for created images.
    let request: GenerationRequest?
    /// The input file name, for depth and upscaling.
    let source: String?
    let created: Date
    let seconds: Double

    var imageURL: URL { History.directory.appending(path: "\(id.uuidString).png") }

    var title: String {
        switch workflow {
        case .create: request?.prompt ?? ""
        case .depth: "Depth of \(source ?? "image")"
        case .upscale: "\(source ?? "Image") ×4"
        }
    }

    init(workflow: Workflow, request: GenerationRequest?, source: String?, seconds: Double) {
        id = UUID()
        self.workflow = workflow
        self.request = request
        self.source = source
        created = .now
        self.seconds = seconds
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(UUID.self, forKey: .id)
        workflow = try values.decodeIfPresent(Workflow.self, forKey: .workflow) ?? .create
        request = try values.decodeIfPresent(GenerationRequest.self, forKey: .request)
        source = try values.decodeIfPresent(String.self, forKey: .source)
        created = try values.decode(Date.self, forKey: .created)
        seconds = try values.decode(Double.self, forKey: .seconds)
    }
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

    func add(png: Data, workflow: Workflow, request: GenerationRequest? = nil, source: String? = nil, seconds: Double) throws -> HistoryItem {
        try FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        let item = HistoryItem(workflow: workflow, request: request, source: source, seconds: seconds)
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
