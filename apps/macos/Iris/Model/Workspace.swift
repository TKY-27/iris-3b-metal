import Foundation
import Observation

/// What the main window shows and what the composer holds.
@MainActor
@Observable
final class Workspace {
    enum Selection: Hashable {
        case allImages
        case conversation(UUID)
    }

    /// An image chosen for depth or upscaling.
    struct Attachment: Equatable {
        let url: URL
        let name: String
    }

    /// nil: a new conversation, created when its first request is sent.
    var selection: Selection? {
        didSet {
            if case let .conversation(id) = selection {
                UserDefaults.standard.set(id.uuidString, forKey: Self.lastConversationKey)
            }
        }
    }

    /// The project a new conversation goes into.
    var draftProject: UUID?
    /// An entry to scroll to once its conversation shows.
    var focusedEntry: UUID?
    /// Set by the menu command; the sidebar asks for the name.
    var requestsNewProject = false

    var prompt = ""
    var workflow = Workflow.create
    var attachment: Attachment?
    var settings = GenerationSettings.load() {
        didSet { settings.save() }
    }

    private static let lastConversationKey = "lastConversation"

    init(library: Library) {
        let last = UserDefaults.standard.string(forKey: Self.lastConversationKey).flatMap(UUID.init(uuidString:))
        if let last, library.conversation(last) != nil {
            selection = .conversation(last)
        }
    }

    var conversationID: UUID? {
        if case let .conversation(id) = selection { id } else { nil }
    }

    func newConversation(in project: UUID? = nil) {
        selection = nil
        draftProject = project
        focusedEntry = nil
    }

    func show(_ conversation: UUID, entry: UUID? = nil) {
        selection = .conversation(conversation)
        focusedEntry = entry
    }

    var canSubmit: Bool {
        switch workflow {
        case .create: !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .depth, .upscale: attachment != nil
        }
    }

    /// Sends the composer's request to a new entry of the current conversation.
    func submit(generator: Generator, library: Library, engine: EngineController) {
        guard canSubmit, !generator.isBusy, engine.state == .ready else { return }
        let conversation: UUID?
        switch workflow {
        case .create:
            let request = settings.request(for: prompt.trimmingCharacters(in: .whitespacesAndNewlines))
            conversation = generator.generate(
                request, in: conversationID, project: draftProject, library: library, engine: engine
            )
            if conversation != nil { prompt = "" }
        case .depth, .upscale:
            guard let attachment else { return }
            conversation = generator.process(
                workflow, image: attachment.url, named: attachment.name, in: conversationID,
                project: draftProject, library: library, engine: engine
            )
            if conversation != nil { self.attachment = nil }
        }
        if let conversation, conversationID != conversation {
            show(conversation)
        }
    }
}
