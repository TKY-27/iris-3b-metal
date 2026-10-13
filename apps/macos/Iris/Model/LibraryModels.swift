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

/// Stored batches and legacy chats within projects.
struct Conversation: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    var title: String
    var projectID: UUID?
    let created: Date
    var updated: Date
    /// Preserve legacy chat pins without displaying them.
    var pinned: Date?
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
        abbreviatedTitle(entry.title)
    }

    static func abbreviatedTitle(_ title: String) -> String {
        let text = title.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard text.count > 48 else { return text }
        let cut = text.prefix(48)
        return String(cut[..<(cut.lastIndex(of: " ") ?? cut.endIndex)]) + "…"
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
    /// as `Inputs/<id>.png` (not for entries imported from Iris 0.1), for Run Again and Compare.
    let inputName: String?
    var outcome: Outcome
    var metadata: GenerationMetadata?

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
        case .upscale: "\(inputName ?? "Image"), restored and upscaled"
        }
    }

    var displayTitle: String {
        switch workflow {
        case .create: request?.prompt ?? ""
        case .depth: String(localized: "Depth of \(inputName ?? String(localized: "an image"))")
        case .upscale: String(localized: "\(inputName ?? String(localized: "Image")), restored and upscaled")
        }
    }

    var isDone: Bool {
        if case .done = outcome { true } else { false }
    }

    /// Search prompts, inputs, models and exact seeds.
    func matches(_ query: String) -> Bool {
        if let seed = request?.seed, Int(query) == seed { return true }
        return title.localizedStandardContains(query)
            || displayTitle.localizedStandardContains(query)
            || request?.model.fullTitle.localizedStandardContains(query) == true
            || request?.model.title.localizedStandardContains(query) == true
    }

    /// A file name for the result: the start of its prompt, or its input's name, then what was
    /// done to it, once: "a-red-fox.png", "a-red-fox-upscaled.png", "a-red-fox-depth.png".
    var fileName: String {
        let source = request?.prompt ?? inputName.map { ($0 as NSString).deletingPathExtension } ?? ""
        var words = source.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
        // an input made by Iris already ends in what was done to it
        while request == nil, let last = words.last, last == "upscaled" || last == "depth" { words.removeLast() }
        let stem = words.isEmpty ? "iris" : words.prefix(6).joined(separator: "-")
        return stem + (workflow == .depth ? "-depth" : workflow == .upscale ? "-upscaled" : "") + ".png"
    }
}

/// What `Library.json` holds.
struct LibrarySnapshot: Codable, Equatable, Sendable {
    // Older apps would discard the model identity and regenerate Turbo entries as Original.
    static let currentVersion = 2

    var version = currentVersion
    var projects: [Project] = []
    var conversations: [Conversation] = []
}
