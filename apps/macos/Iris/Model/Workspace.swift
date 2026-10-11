import Foundation
import Observation

/// What the main window shows and what the composer holds.
@MainActor
@Observable
final class Workspace {
    enum Selection: Hashable {
        /// A chat that exists once its first request is sent.
        case newChat
        case allImages
        case project(UUID)
        case conversation(UUID)
    }

    /// An image chosen for depth or upscaling.
    struct Attachment: Equatable {
        let url: URL
        let name: String
        /// Upright, as it is sent; nil when the file cannot be read.
        var pixelSize: CGSize?
    }

    /// Optional only because a sidebar list may clear it; nil shows a new chat.
    var selection: Selection? = .newChat {
        didSet {
            // a click on the sidebar's empty space clears the list's selection; keep what is shown
            if selection == nil {
                selection = oldValue ?? .newChat
                return
            }
            switch selection {
            case let .conversation(id):
                defaults.set(id.uuidString, forKey: Self.lastConversationKey)
            case let .project(id):
                draftProject = id  // its page starts chats in it
            case .newChat, nil:
                if oldValue != selection { draftProject = nil }
            case .allImages:
                draftProject = nil
            }
        }
    }

    /// The project a new chat goes into.
    var draftProject: UUID?
    /// An entry to scroll to once its conversation shows.
    var focusedEntry: UUID?
    /// A name being asked for, and a deletion being confirmed; the window shows the dialogs.
    var naming: Naming?
    var deleting: Deletion?

    enum Naming: Equatable {
        /// `moving`: a chat to file in the new project.
        case newProject(moving: UUID?)
        /// From the composer of a new chat, which is then saved in it.
        case newProjectForDraft
        case project(UUID)
        case conversation(UUID)
    }

    enum Deletion: Equatable {
        case project(UUID)
        case conversation(UUID)
    }

    var prompt = ""
    /// An attached image switches the composer from creating to processing it.
    var attachment: Attachment?
    /// What to do with an attached image.
    var imageWorkflow = Workflow.upscale
    /// How many images the next prompt makes; back to one at every launch, so a long batch
    /// is never started by accident.
    var imageCount = 1
    static let imageCountRange = 1...4
    var settings: GenerationSettings {
        didSet { settings.save(to: defaults) }
    }

    @ObservationIgnored private let defaults: UserDefaults
    private static let lastConversationKey = "lastConversation"

    init(library: Library, defaults: UserDefaults = .standard) {
        self.defaults = defaults
        settings = GenerationSettings.load(from: defaults)
        let last = defaults.string(forKey: Self.lastConversationKey).flatMap(UUID.init(uuidString:))
        if let last, library.conversation(last) != nil {
            selection = .conversation(last)
        }
    }

    /// What the composer will send.
    var workflow: Workflow { attachment == nil ? .create : imageWorkflow }

    func attach(_ url: URL) {
        // reads only the file's header
        attachment = Attachment(url: url, name: url.lastPathComponent, pixelSize: ImageFiles.pixelSize(of: url))
    }

    /// Puts a request's prompt and settings back into the composer.
    func reuse(_ request: GenerationRequest) {
        attachment = nil
        prompt = request.prompt
        settings.reuse(request)
    }

    var conversationID: UUID? {
        if case let .conversation(id) = selection { id } else { nil }
    }

    func newConversation(in project: UUID? = nil) {
        selection = .newChat
        draftProject = project
        focusedEntry = nil
    }

    /// A new chat where the user is: in the shown project, or the project of the shown chat.
    func newConversationHere(library: Library) {
        switch selection {
        case let .project(id): newConversation(in: id)
        case let .conversation(id): newConversation(in: library.conversation(id)?.projectID)
        default: newConversation(in: draftProject)
        }
    }

    func show(_ conversation: UUID, entry: UUID? = nil) {
        selection = .conversation(conversation)
        focusedEntry = entry
    }

    var canSubmit: Bool {
        switch workflow {
        case .create: !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .depth: attachment != nil
        case .upscale: attachment.map { $0.pixelSize.map { Upscaling.refusal(for: $0) == nil } ?? true } ?? false
        }
    }

    /// Sends the composer's request to a new entry of the current conversation.
    func submit(generator: Generator, library: Library, engine: EngineController) {
        guard canSubmit, !generator.isBusy, engine.state == .ready else { return }
        let conversation: UUID?
        switch workflow {
        case .create:
            let requests = settings.requests(for: prompt.trimmingCharacters(in: .whitespacesAndNewlines), count: imageCount)
            conversation = generator.generate(
                requests, in: conversationID, project: draftProject, library: library, engine: engine
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
