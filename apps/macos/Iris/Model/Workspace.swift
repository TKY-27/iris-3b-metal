import Foundation
import Observation

/// Window selection and the next generation request.
@MainActor
@Observable
final class Workspace {
    /// Gallery scope and destination for new images.
    enum Selection: Hashable {
        case allImages
        case project(UUID)

        func contains(_ conversation: Conversation) -> Bool {
            switch self {
            case .allImages: true
            case let .project(id): conversation.projectID == id
            }
        }
    }

    /// An image chosen for depth or upscaling.
    struct Attachment: Equatable {
        let url: URL
        let name: String
        /// Upright, as it is sent; nil when the file cannot be read.
        var pixelSize: CGSize?
    }

    /// Sidebar selection may temporarily become nil.
    var selection: Selection? = .allImages {
        didSet {
            // a click on the sidebar's empty space clears the list's selection; keep what is shown.
            // Setting it here runs this again with a nil old value, which is not a change of images.
            if selection == nil {
                selection = oldValue ?? .allImages
            } else if let oldValue, selection != oldValue {
                selectedEntry = nil
            }
        }
    }

    /// The result shown in the inspector.
    var selectedEntry: UUID?
    /// Filters the gallery by prompt or seed.
    var searchText = ""
    private(set) var searchFocusRequests = 0
    private(set) var searchFocusPending = false
    /// Prompt focus request counter.
    private(set) var promptFocusRequests = 0
    /// A name being asked for, and a deletion being confirmed; the window shows the dialogs.
    var naming: Naming?
    var deleting: Deletion?

    enum Naming: Equatable {
        /// Move this image into the new project.
        case newProject(moving: UUID?)
        case project(UUID)
    }

    enum Deletion: Equatable {
        case project(UUID)
        case image(UUID)
    }

    static let galleryVisibilityKey = "showsGallery"
    var showsGallery: Bool {
        didSet { defaults.set(showsGallery, forKey: Self.galleryVisibilityKey) }
    }

    func toggleGallery() {
        showsGallery.toggle()
        if !showsGallery { focusPrompt() }
    }

    var prompt = ""
    /// Attachments switch generation to image processing.
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

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        showsGallery = defaults.object(forKey: Self.galleryVisibilityKey) as? Bool ?? true
        settings = GenerationSettings.load(from: defaults)
    }

    /// Destination project for new images.
    var project: UUID? {
        if case let .project(id) = selection { id } else { nil }
    }

    /// What the prompt bar will send.
    var workflow: Workflow { attachment == nil ? .create : imageWorkflow }

    func attach(_ url: URL) {
        // reads only the file's header
        attachment = Attachment(url: url, name: url.lastPathComponent, pixelSize: ImageFiles.pixelSize(of: url))
    }

    func focusPrompt() {
        promptFocusRequests += 1
    }

    func focusSearch() {
        showsGallery = true
        searchFocusPending = true
        searchFocusRequests += 1
    }

    func didFocusSearch() {
        searchFocusPending = false
    }

    /// Clear the attachment and focus the prompt.
    func newImage() {
        attachment = nil
        focusPrompt()
    }

    /// Restore the request, including its seed.
    func reuse(_ request: GenerationRequest) {
        attachment = nil
        prompt = request.prompt
        settings.reuse(request)
        focusPrompt()
    }

    /// Reveal an entry in its gallery scope.
    func reveal(_ entry: UUID, library: Library) {
        showsGallery = true
        if let found = library.entry(entry), selection.map({ !$0.contains(found.conversation) }) ?? true {
            selection = found.conversation.projectID.map(Selection.project) ?? .allImages
        }
        selectedEntry = entry
    }

    /// Select new work and clear obstructing filters.
    func selectStarted(_ entry: UUID?, library: Library) {
        guard let entry else { return }
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !query.isEmpty, library.entry(entry)?.entry.matches(query) == false { searchText = "" }
        selectedEntry = entry
    }

    var canSubmit: Bool {
        switch workflow {
        case .create: !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .depth: attachment != nil
        case .upscale: attachment.map { $0.pixelSize.map { Upscaling.refusal(for: $0) == nil } ?? true } ?? false
        }
    }

    /// Generate in the selected project, retaining prompts.
    func submit(generator: Generator, library: Library, engine: EngineController) {
        guard canSubmit, !generator.isBusy, engine.state == .ready else { return }
        switch workflow {
        case .create:
            guard engine.health?.supports(settings.model) == true else { return }
            let requests = settings.requests(for: prompt.trimmingCharacters(in: .whitespacesAndNewlines), count: imageCount)
            generator.generate(requests, in: nil, project: project, library: library, engine: engine)
        case .depth, .upscale:
            guard let attachment else { return }
            let started = generator.process(
                workflow, image: attachment.url, named: attachment.name, in: nil,
                project: project, library: library, engine: engine
            )
            if started != nil { self.attachment = nil }
        }
        selectStarted(generator.active?.entry, library: library)
    }
}
