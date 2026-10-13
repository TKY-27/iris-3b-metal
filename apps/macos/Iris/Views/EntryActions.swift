import AppKit
import SwiftUI

/// What can be done with a result, for its card, the inspector and their menus.
@MainActor
struct EntryActions {
    let entry: Entry
    let conversation: UUID
    let library: Library
    let generator: Generator
    let engine: EngineController
    let workspace: Workspace

    var imageURL: URL { library.imageURL(entry.id) }

    var canStart: Bool { !generator.isBusy && engine.state == .ready && !library.isReadOnly }

    var canRegenerate: Bool { canStart && (entry.request.map { engine.health?.supports($0.model) == true } ?? true) }

    /// A running entry stays where it is until it is done.
    var canChange: Bool { !library.isReadOnly && entry.outcome != .running }

    /// The input of a depth map or upscale, kept since Iris 0.2.
    var hasInput: Bool {
        entry.request == nil && FileManager.default.fileExists(atPath: library.inputURL(entry.id).path)
    }

    var canCompare: Bool { entry.isDone && hasInput }

    var canRunAgain: Bool { entry.request != nil || hasInput }

    /// Whether the restorer would make this result larger; an upscale of an upscale usually would not.
    var canUpscale: Bool {
        guard entry.isDone, entry.workflow != .depth else { return false }
        let size = entry.request.map { CGSize(width: $0.width, height: $0.height) } ?? ImageFiles.pixelSize(of: imageURL)
        return size.map { Upscaling.refusal(for: $0) == nil } ?? false
    }

    var canEstimateDepth: Bool { entry.isDone && entry.workflow != .depth }

    func copyImage() { ImageActions.copy(imageURL) }

    func saveAs() { ImageActions.saveAs(imageURL, suggestedName: entry.fileName) }

    func showInFinder() { NSWorkspace.shared.activateFileViewerSelecting([imageURL]) }

    func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// The same request as a new entry, with a new seed or exactly as it was; the new one is selected.
    func runAgain(newSeed: Bool) {
        generator.rerun(entry, newSeed: newSeed, in: conversation, library: library, engine: engine)
        selectStarted()
    }

    /// Depth or upscaling of this result as a new entry, which is selected.
    func process(_ workflow: Workflow) {
        generator.process(
            workflow, image: imageURL, named: entry.fileName, in: conversation, library: library, engine: engine
        )
        selectStarted()
    }

    /// An image is deleted after a confirmation; a request without one at once.
    func delete() {
        if entry.isDone {
            workspace.deleting = .image(entry.id)
        } else {
            library.delete(entry: entry.id)
        }
    }

    private func selectStarted() {
        workspace.selectStarted(generator.active?.entry, library: library)
    }
}

/// The context menu of a result, and the inspector's More menu.
struct EntryMenu: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    @Environment(\.openWindow) private var openWindow
    let entry: Entry
    let conversation: UUID
    let project: UUID?
    let preview: (URL) -> Void

    var body: some View {
        let actions = EntryActions(
            entry: entry, conversation: conversation, library: library,
            generator: generator, engine: engine, workspace: workspace
        )
        if entry.isDone {
            Button("Quick Look") { preview(actions.imageURL) }
            Button("Copy Image", action: actions.copyImage)
            Button("Save As…", action: actions.saveAs)
            Button("Show in Finder", action: actions.showInFinder)
            if actions.canCompare {
                Button("Compare with Original") { openWindow(id: ComparisonWindow.id, value: entry.id) }
            }
            Divider()
        }
        if entry.outcome == .running {
            Button("Stop") { generator.cancel(engine: engine) }
            Divider()
        }
        if let request = entry.request {
            Button("Reuse Prompt and Settings") { workspace.reuse(request) }
            if entry.isDone {
                Button("Generate Again") { actions.runAgain(newSeed: false) }
                    .disabled(!actions.canRegenerate)
                Button("Generate with New Seed") { actions.runAgain(newSeed: true) }
                    .disabled(!actions.canRegenerate)
            }
            Button("Copy Prompt") { actions.copy(request.prompt) }
            Button("Copy Seed") { actions.copy(String(request.seed)) }
        }
        if !entry.isDone, entry.outcome != .running, actions.canRunAgain {
            Button("Try Again") { actions.runAgain(newSeed: false) }
                .disabled(!actions.canRegenerate)
        }
        if actions.canUpscale || actions.canEstimateDepth {
            Divider()
            if actions.canUpscale {
                Button(Workflow.upscale.title) { actions.process(.upscale) }.disabled(!actions.canStart)
            }
            if actions.canEstimateDepth {
                Button("Estimate Depth") { actions.process(.depth) }.disabled(!actions.canStart)
            }
        }
        Divider()
        Menu("Move to Project") {
            ForEach(library.projects) { candidate in
                Button(candidate.name) { library.move(entry: entry.id, to: candidate.id) }
                    .disabled(candidate.id == project)
            }
            if !library.projects.isEmpty { Divider() }
            Button("New Project…") { workspace.naming = .newProject(moving: entry.id) }
            if project != nil {
                Button("Remove from Project") { library.move(entry: entry.id, to: nil) }
            }
        }
        .disabled(!actions.canChange)
        Divider()
        Button(entry.isDone ? "Delete…" : "Delete", role: .destructive, action: actions.delete)
            .disabled(!actions.canChange)
    }
}
