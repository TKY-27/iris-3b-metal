import Foundation

/// A named group of conversations.
struct Project: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    var name: String
    let created: Date

    init(name: String) {
        id = UUID()
        self.name = name
        created = .now
    }
}

/// A sequence of independent requests. Iris-3B has no memory: every entry is
/// generated on its own, never from earlier entries of the conversation.
struct Conversation: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    var title: String
    var projectID: UUID?
    let created: Date
    var updated: Date
    var entries: [Entry]

    init(id: UUID = UUID(), title: String, projectID: UUID?, entries: [Entry] = []) {
        self.id = id
        self.title = title
        self.projectID = projectID
        created = entries.first?.created ?? .now
        updated = entries.last?.created ?? created
        self.entries = entries
    }

    /// A short title from a request: the start of its prompt or the input name.
    static func title(for entry: Entry) -> String {
        let text = entry.title.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard text.count > 48 else { return text }
        let cut = text.prefix(48)
        return String(cut[..<(cut.lastIndex(of: " ") ?? cut.endIndex)]) + "…"
    }

    func matches(_ query: String) -> Bool {
        title.localizedStandardContains(query) || entries.contains { $0.title.localizedStandardContains(query) }
    }
}

/// One request and its result.
struct Entry: Codable, Identifiable, Hashable, Sendable {
    enum Outcome: Codable, Hashable, Sendable {
        case running
        case done(seconds: Double)
        case failed(String)
        case cancelled
        /// Iris quit or lost the engine before the result arrived.
        case interrupted
    }

    let id: UUID
    let created: Date
    let workflow: Workflow
    /// The settings of a created image.
    let request: GenerationRequest?
    /// The file name of a depth or upscaling input; the input itself is kept
    /// as `Inputs/<id>.png` (not for entries imported from Iris 0.1).
    let inputName: String?
    var outcome: Outcome

    init(
        id: UUID = UUID(), created: Date = .now, workflow: Workflow,
        request: GenerationRequest? = nil, inputName: String? = nil, outcome: Outcome = .running
    ) {
        self.id = id
        self.created = created
        self.workflow = workflow
        self.request = request
        self.inputName = inputName
        self.outcome = outcome
    }

    var title: String {
        switch workflow {
        case .create: request?.prompt ?? ""
        case .depth: "Depth of \(inputName ?? "an image")"
        case .upscale: "\(inputName ?? "Image") ×4"
        }
    }

    var isDone: Bool {
        if case .done = outcome { true } else { false }
    }
}

/// What `Library.json` holds.
struct LibrarySnapshot: Codable, Equatable, Sendable {
    static let currentVersion = 1

    var version = currentVersion
    var projects: [Project] = []
    var conversations: [Conversation] = []
}
