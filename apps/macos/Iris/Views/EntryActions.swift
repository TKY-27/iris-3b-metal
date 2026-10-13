import AppKit
import SwiftUI

/// Shared result actions for cards and inspector.
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

    /// Running entries cannot move.
    var canChange: Bool { !library.isReadOnly && entry.outcome != .running }

    /// Original input, retained since Iris 0.2.
    var hasInput: Bool {
        entry.request == nil && FileManager.default.fileExists(atPath: library.inputURL(entry.id).path)
    }

    var canCompare: Bool { entry.isDone && hasInput }

    var canRunAgain: Bool { entry.request != nil || hasInput }

    /// Upscaling must enlarge the result.
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

    /// Regenerate with the saved or new seed.
    func runAgain(newSeed: Bool) {
        generator.rerun(entry, newSeed: newSeed, in: conversation, library: library, engine: engine)
        selectStarted()
    }

    /// Process this result into a new entry.
    func process(_ workflow: Workflow) {
        generator.process(
            workflow, image: imageURL, named: entry.fileName, in: conversation, library: library, engine: engine
        )
        selectStarted()
    }

    /// Confirm before deleting a generated image.
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

/// Result context menu and inspector actions.
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
            Button(String(localized: "Quick Look")) { preview(actions.imageURL) }
            Button(String(localized: "Copy Image"), action: actions.copyImage)
            Button(String(localized: "Save As…"), action: actions.saveAs)
            Button(String(localized: "Show in Finder"), action: actions.showInFinder)
            if actions.canCompare {
                Button(String(localized: "Compare with Original")) { openWindow(id: ComparisonWindow.id, value: entry.id) }
            }
            Divider()
        }
        if entry.outcome == .running {
            Button(String(localized: "Stop")) { generator.cancel(engine: engine) }
            Divider()
        }
        if let request = entry.request {
            Button(String(localized: "Reuse Prompt and Settings")) { workspace.reuse(request) }
            if entry.isDone {
                Button(String(localized: "Generate Again")) { actions.runAgain(newSeed: false) }
                    .disabled(!actions.canRegenerate)
                Button(String(localized: "Generate with New Seed")) { actions.runAgain(newSeed: true) }
                    .disabled(!actions.canRegenerate)
            }
            Button(String(localized: "Copy Prompt")) { actions.copy(request.prompt) }
            Button(String(localized: "Copy Seed")) { actions.copy(String(request.seed)) }
        }
        if !entry.isDone, entry.outcome != .running, actions.canRunAgain {
            Button(String(localized: "Try Again")) { actions.runAgain(newSeed: false) }
                .disabled(!actions.canRegenerate)
        }
        if actions.canUpscale || actions.canEstimateDepth {
            Divider()
            if actions.canUpscale {
                Button(Workflow.upscale.title) { actions.process(.upscale) }.disabled(!actions.canStart)
            }
            if actions.canEstimateDepth {
                Button(String(localized: "Estimate Depth")) { actions.process(.depth) }.disabled(!actions.canStart)
            }
        }
        Divider()
        Menu(String(localized: "Move to Project")) {
            ForEach(library.projects) { candidate in
                Button(candidate.name) { library.move(entry: entry.id, to: candidate.id) }
                    .disabled(candidate.id == project)
            }
            if !library.projects.isEmpty { Divider() }
            Button(String(localized: "New Project…")) { workspace.naming = .newProject(moving: entry.id) }
            if project != nil {
                Button(String(localized: "Remove from Project")) { library.move(entry: entry.id, to: nil) }
            }
        }
        .disabled(!actions.canChange)
        Divider()
        Button(entry.isDone ? String(localized: "Delete…") : String(localized: "Delete"), role: .destructive, action: actions.delete)
            .disabled(!actions.canChange)
    }
}
