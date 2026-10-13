import SwiftUI

/// The selected result: the image, its full prompt and every setting it was made with, and
/// what can be done with it.
struct ImageInspector: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let preview: (URL) -> Void

    var body: some View {
        if let id = workspace.selectedEntry, let found = library.entry(id) {
            EntryDetails(entry: found.entry, conversation: found.conversation, preview: preview)
                .id(id)  // the sizes of one entry are never shown for another
        } else {
            ContentUnavailableView(
                "No Selection", systemImage: "photo",
                description: Text("Select an image to see its prompt, seed and settings.")
            )
        }
    }
}

private struct EntryDetails: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    @Environment(\.openWindow) private var openWindow
    let entry: Entry
    let conversation: Conversation
    let preview: (URL) -> Void
    @State private var sizes = EntrySizes()

    private var actions: EntryActions {
        EntryActions(
            entry: entry, conversation: conversation.id, library: library,
            generator: generator, engine: engine, workspace: workspace
        )
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                picture
                if let request = entry.request {
                    prompt(request)
                } else {
                    input
                }
                primaryActions
                Divider()
                details
            }
            .padding(16)
        }
        // the input is written just after a depth or upscaling entry appears, the result when it is done
        .task(id: SizeKey(outcome: entry.outcome, storedInput: generator.storedInput == entry.id)) {
            let loaded = await EntrySizes.load(entry, library: library)
            if !Task.isCancelled { sizes = loaded }  // not a slower read for an earlier state
        }
    }

    // MARK: - Picture

    @ViewBuilder private var picture: some View {
        switch entry.outcome {
        case .done:
            VStack(alignment: .leading, spacing: 8) {
                Button { preview(actions.imageURL) } label: {
                    StoredImage(url: actions.imageURL, maxPixels: 1200, aspectRatio: entry.aspectRatio ?? sizes.aspectRatio)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.tileRadius, style: .continuous))
                        .frame(maxWidth: .infinity, maxHeight: 360)
                }
                .buttonStyle(.plain)
                .draggable(actions.imageURL)
                .help("Quick Look (Space)")
                .accessibilityLabel(entry.title)
                .accessibilityHint("Opens Quick Look")
                actionBar
            }
        case .running:
            RunningStatus(entry: entry)
        case .failed, .cancelled, .interrupted:
            Label(entry.statusText ?? "", systemImage: entry.outcome == .cancelled ? "stop.circle" : "exclamationmark.triangle")
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: Theme.tileRadius, style: .continuous))
        }
    }

    private var actionBar: some View {
        HStack(spacing: 2) {
            IconButton("Copy Image", systemImage: "doc.on.doc", action: actions.copyImage)
            IconButton("Save As…", systemImage: "square.and.arrow.down", action: actions.saveAs)
            if actions.canCompare {
                IconButton("Compare with Original", systemImage: "square.split.2x1") {
                    openWindow(id: ComparisonWindow.id, value: entry.id)
                }
            }
            if actions.canUpscale {
                IconButton(Workflow.upscale.title, systemImage: Workflow.upscale.symbol) { actions.process(.upscale) }
                    .disabled(!actions.canStart)
            }
            if actions.canEstimateDepth {
                IconButton("Estimate Depth", systemImage: Workflow.depth.symbol) { actions.process(.depth) }
                    .disabled(!actions.canStart)
            }
            Spacer(minLength: 0)
            Menu {
                EntryMenu(entry: entry, conversation: conversation.id, project: conversation.projectID, preview: preview)
            } label: {
                Label("More", systemImage: "ellipsis")
            }
            .menuStyle(.button)
            .buttonStyle(.borderless)
            .menuIndicator(.hidden)
            .labelStyle(.iconOnly)
            .frame(width: 26, height: 22)
            .help("More")
        }
    }

    // MARK: - Request

    private func prompt(_ request: GenerationRequest) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Prompt").font(.headline)
                Spacer()
                IconButton("Copy Prompt", systemImage: "doc.on.doc") { actions.copy(request.prompt) }
            }
            Text(request.prompt)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            if request.model == .original, !request.negativePrompt.isEmpty {
                Text("Avoid: \(request.negativePrompt)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var input: some View {
        HStack(spacing: 10) {
            StoredImage(url: library.inputURL(entry.id), maxPixels: 160, placeholder: "photo")
                .id(generator.storedInput == entry.id)  // the input is written just after the entry appears
                .frame(width: 44, height: 44)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Label(entry.workflow.title, systemImage: entry.workflow.symbol)
                    .font(.headline)
                Text(entry.inputName ?? "image")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    /// Making the image again, the same or changed, is the next step after looking at one.
    @ViewBuilder private var primaryActions: some View {
        VStack(spacing: 8) {
            if let request = entry.request {
                Button {
                    workspace.reuse(request)
                } label: {
                    Text("Reuse Prompt and Settings").frame(maxWidth: .infinity)
                }
                .help("Put this prompt and its settings, seed included, in the prompt bar")
                if entry.isDone {
                    Button {
                        actions.runAgain(newSeed: false)
                    } label: {
                        Text("Generate Again").frame(maxWidth: .infinity)
                    }
                    .disabled(!actions.canRegenerate)
                    .help("Generate with the same model, prompt, seed and settings")
                    Button {
                        actions.runAgain(newSeed: true)
                    } label: {
                        Text("Generate with New Seed").frame(maxWidth: .infinity)
                    }
                    .disabled(!actions.canRegenerate)
                    .help("Generate this prompt again with the same settings and a new seed")
                }
            }
            if !entry.isDone, entry.outcome != .running, actions.canRunAgain {
                Button {
                    actions.runAgain(newSeed: false)
                } label: {
                    Text("Try Again").frame(maxWidth: .infinity)
                }
                .disabled(!actions.canRegenerate)
            }
        }
        .controlSize(.large)
    }

    // MARK: - Details

    private var details: some View {
        Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 8) {
            if let r = entry.request {
                let size = ImageSize(width: r.width, height: r.height)
                row("Model", r.model.fullTitle)
                row("Size", "\(size.label) (\(size.ratioLabel))")
                row("Steps", r.model == .turbo ? "4 · Fixed" : QualityPreset(steps: r.steps).map { "\(r.steps) · \($0.title)" } ?? "\(r.steps)")
                if r.model == .original {
                    row("Guidance", AdherencePreset(cfgScale: r.cfgScale).map { "\(r.cfgScale.formatted()) · \($0.title)" } ?? r.cfgScale.formatted())
                }
                GridRow {
                    label("Seed")
                    HStack(spacing: 4) {
                        Text(String(r.seed)).textSelection(.enabled)
                        IconButton("Copy Seed", systemImage: "doc.on.doc") { actions.copy(String(r.seed)) }
                    }
                }
            } else {
                if let input = sizes.input { row("Input", input.pixelLabel) }
                if let output = sizes.output { row("Result", output.pixelLabel) }
            }
            row("Created", entry.created.formatted(date: .abbreviated, time: .shortened))
            if let precision = entry.metadata?.precisionLabel { row("Precision", precision) }
            if case let .done(seconds) = entry.outcome {
                row("Time", Duration.seconds(seconds).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))
            }
            GridRow {
                label("Project")
                Picker("Project", selection: Binding(
                    get: { conversation.projectID },
                    set: { library.move(entry: entry.id, to: $0) }
                )) {
                    Text("None").tag(UUID?.none)
                    ForEach(library.projects) { Text($0.name).tag(UUID?.some($0.id)) }
                }
                .labelsHidden()
                .fixedSize()
                .disabled(!actions.canChange)
            }
        }
        .font(.callout)
        .monospacedDigit()
    }

    private func row(_ title: String, _ value: String) -> some View {
        GridRow {
            label(title)
            Text(value)
        }
    }

    private func label(_ title: String) -> some View {
        Text(title)
            .foregroundStyle(.secondary)
            .gridColumnAlignment(.trailing)
    }
}

/// A running or queued request, with Stop.
private struct RunningStatus: View {
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    let entry: Entry

    var body: some View {
        let isActive = generator.active?.entry == entry.id
        VStack(alignment: .leading, spacing: 8) {
            if isActive {
                let text = ProgressText(generator.progress, workflow: entry.workflow)
                ProgressView(value: generator.progress.fraction ?? 0)
                    .accessibilityValue(ProgressText.spoken(generator.progress, workflow: entry.workflow))
                Text(text.headline).font(.headline).monospacedDigit()
                if let detail = text.detail {
                    Text(detail).font(.callout).foregroundStyle(.secondary).monospacedDigit()
                }
            } else {
                Text(generator.isQueued(entry.id) ? "Queued" : "Waiting").font(.headline)
                Text("Starts when the image before it is done.").font(.callout).foregroundStyle(.secondary)
            }
            if generator.isBusy {
                Button("Stop") { generator.cancel(engine: engine) }
                    .help("Stop this request and the rest of its batch (⌘.)")
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: Theme.tileRadius, style: .continuous))
    }
}

struct IconButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    init(_ title: String, systemImage: String, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.iconOnly)
                .frame(width: 26, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(title)
    }
}

/// Pixel sizes of an entry's input and result, read from the files' headers.
struct EntrySizes: Equatable {
    var input: CGSize?
    var output: CGSize?

    /// The result's shape, or the input's while the result is being made.
    var aspectRatio: CGFloat? {
        (output ?? input).map { $0.width / $0.height }
    }

    @MainActor static func load(_ entry: Entry, library: Library) async -> EntrySizes {
        guard entry.request == nil else { return EntrySizes() }  // its request has the size
        let (input, output) = (library.inputURL(entry.id), library.imageURL(entry.id))
        let done = entry.isDone
        return await Task.detached(priority: .utility) {
            EntrySizes(input: ImageFiles.pixelSize(of: input), output: done ? ImageFiles.pixelSize(of: output) : nil)
        }.value
    }
}

private struct SizeKey: Equatable {
    let outcome: Entry.Outcome
    let storedInput: Bool
}
