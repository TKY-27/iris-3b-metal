import SwiftUI

/// One request (trailing, like a sent message) and its result, canvas or status (leading).
struct EntryView: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.openWindow) private var openWindow
    let entry: Entry
    let conversation: UUID
    /// The newest entry keeps its actions visible; older ones show them on hover.
    var isLatest = true
    let preview: (URL) -> Void
    @State private var hovering = false
    @State private var sizes = EntrySizes()
    @FocusState private var actionFocus: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            RequestBubble(entry: entry)
                .frame(maxWidth: .infinity, alignment: .trailing)
            result
                .transition(.opacity)
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.45), value: entry.outcome)
        .onHover { hovering = $0 }
        // the input is written just after a depth or upscaling entry appears, the result when it is done
        .task(id: SizeKey(outcome: entry.outcome, storedInput: generator.storedInput == entry.id)) {
            let loaded = await EntrySizes.load(entry, library: library)
            if !Task.isCancelled { sizes = loaded }  // not a slower read for an earlier state
        }
        .onChange(of: entry.outcome) { old, _ in
            guard old == .running else { return }
            AccessibilityNotification.Announcement(entry.isDone ? "Image ready" : "The request ended").post()
        }
    }

    @ViewBuilder private var result: some View {
        switch entry.outcome {
        case .running:
            GenerationCanvas(entry: entry, aspectRatio: aspectRatio)
        case let .done(seconds):
            VStack(alignment: .leading, spacing: 8) {
                ResultImage(
                    entry: entry, url: library.imageURL(entry.id), aspectRatio: aspectRatio,
                    open: { preview(library.imageURL(entry.id)) }
                )
                .contextMenu { menu }
                HStack(spacing: 2) {
                    Text(caption(seconds: seconds))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        .lineLimit(1)
                        .help(exactSettings)
                        // Not selectable: a selectable Text with its own accessibility label sends
                        // SwiftUI's label resolution into endless recursion when an assistive
                        // client reads it, which crashed the app.
                        .accessibilityLabel(caption(seconds: seconds).spokenSizes)
                    Spacer(minLength: 12)
                    actionBar
                        .opacity(isLatest || hovering || actionFocus != nil ? 1 : 0)  // shown on hover or keyboard focus
                }
                .frame(maxWidth: ResultImage.width(aspectRatio: aspectRatio))
            }
        case .failed, .cancelled, .interrupted:
            HStack(spacing: 12) {
                Label(statusText, systemImage: entry.outcome == .cancelled ? "stop.circle" : "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
                if canRunAgain {
                    Button("Try Again") { runAgain(newSeed: false) }
                        .disabled(!canStart)
                }
                Button("Delete", role: .destructive) { library.delete(entry: entry.id) }
            }
            .font(.callout)
            .padding(14)
            .frame(maxWidth: ResultImage.maxWidth)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
            .contextMenu { menu }
        }
    }

    // MARK: - Text

    private var statusText: String {
        switch entry.outcome {
        case let .failed(message): message
        case .cancelled: "Stopped."
        case .interrupted: "Interrupted before the result arrived."
        case .running, .done: ""
        }
    }

    /// The settings in the words the composer uses; the exact numbers are in the tooltip.
    private func caption(seconds: Double) -> String {
        let time = Duration.seconds(seconds).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated))
        guard let r = entry.request else {
            switch entry.workflow {
            case .depth:
                return "Relative depth, near is bright · \(time)"
            default:
                // the request above names the action; the sizes say what it did
                guard let input = sizes.input, let output = sizes.output else { return time }
                return "\(input.pixelLabel) → \(output.pixelLabel) · \(time)"
            }
        }
        let quality = QualityPreset(steps: r.steps)?.title ?? "\(r.steps) steps"
        let adherence = AdherencePreset(cfgScale: r.cfgScale)?.title ?? "guidance \(r.cfgScale.formatted())"
        return "\(ImageSize(width: r.width, height: r.height).ratioLabel) · \(quality) · \(adherence) · \(time)"
    }

    private var exactSettings: String {
        guard let r = entry.request else {
            if entry.workflow == .depth { return "A 16-bit grayscale PNG; compare it with the original to check edges" }
            guard let input = sizes.input else { return "Restored at 4×" }
            let working = Upscaling.workingSize(for: input)
            return working == input ? "Restored at 4×" : "Reduced to \(working.pixelLabel), then restored at 4×"
        }
        return "\(r.width) × \(r.height) · \(r.steps) steps · guidance \(r.cfgScale.formatted()) · seed \(r.seed)"
    }

    // MARK: - Actions

    private var canStart: Bool { !generator.isBusy && engine.state == .ready && !library.isReadOnly }

    private var canRunAgain: Bool {
        entry.request != nil || sizes.input != nil
    }

    /// A depth map or upscale next to the image it was made from.
    private var canCompare: Bool { entry.isDone && sizes.input != nil }

    /// A created image's shape is known before its file is read, so its frame never jumps.
    private var aspectRatio: CGFloat? {
        entry.request.map { CGFloat($0.width) / CGFloat($0.height) } ?? sizes.aspectRatio
    }

    /// Whether the restorer would make this result larger; an upscale of an upscale usually would not.
    private var canUpscale: Bool {
        let request = entry.request.map { CGSize(width: $0.width, height: $0.height) }
        guard entry.workflow != .depth, let size = request ?? sizes.output else { return false }
        return Upscaling.refusal(for: size) == nil
    }

    private var actionBar: some View {
        HStack(spacing: 2) {
            if canCompare {
                // the point of a depth map or upscale, so it is named, not only drawn
                Button(action: compare) {
                    Label("Compare", systemImage: "square.split.2x1")
                        .labelStyle(.titleAndIcon)
                        .padding(.horizontal, 4)
                        .frame(height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .help("Compare with Original")
                .focused($actionFocus, equals: "compare")
            }
            IconButton("Copy Image", systemImage: "doc.on.doc") { ImageActions.copy(library.imageURL(entry.id)) }
                .focused($actionFocus, equals: "copy")
            IconButton("Save As…", systemImage: "square.and.arrow.down") {
                ImageActions.saveAs(library.imageURL(entry.id), suggestedName: suggestedName)
            }
            .focused($actionFocus, equals: "save")
            if entry.workflow == .create {
                IconButton("Variation with a New Seed", systemImage: "arrow.clockwise") { runAgain(newSeed: true) }
                    .disabled(!canStart)
                    .focused($actionFocus, equals: "vary")
            }
            if canUpscale {
                IconButton(Workflow.upscale.title, systemImage: Workflow.upscale.symbol) { process(.upscale) }
                    .disabled(!canStart)
                    .focused($actionFocus, equals: "upscale")
            }
            if entry.workflow != .depth {
                IconButton("Estimate Depth", systemImage: Workflow.depth.symbol) { process(.depth) }
                    .disabled(!canStart)
                    .focused($actionFocus, equals: "depth")
            }
            Menu {
                menu
            } label: {
                Label("More", systemImage: "ellipsis")
            }
            .menuStyle(.button)
            .buttonStyle(.borderless)
            .menuIndicator(.hidden)
            .labelStyle(.iconOnly)
            .frame(width: 26, height: 22)
            .focused($actionFocus, equals: "more")
            .help("More")
        }
    }

    @ViewBuilder private var menu: some View {
        if entry.isDone {
            Button("Quick Look") { preview(library.imageURL(entry.id)) }
            Button("Copy Image") { ImageActions.copy(library.imageURL(entry.id)) }
            Button("Save As…") { ImageActions.saveAs(library.imageURL(entry.id), suggestedName: suggestedName) }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([library.imageURL(entry.id)]) }
            if canCompare {
                Button("Compare with Original") { compare() }
            }
            Divider()
        }
        if let request = entry.request {
            if entry.isDone {
                Button("Variation with a New Seed") { runAgain(newSeed: true) }.disabled(!canStart)
            }
            Button("Reuse Prompt and Settings") { workspace.reuse(request) }
        }
        if entry.isDone, entry.workflow != .depth {
            Divider()
            if canUpscale {
                Button(Workflow.upscale.title) { process(.upscale) }.disabled(!canStart)
            }
            Button("Estimate Depth") { process(.depth) }.disabled(!canStart)
        }
        Divider()
        Button("Delete", role: .destructive) { library.delete(entry: entry.id) }
            .disabled(entry.outcome == .running)
    }

    private var suggestedName: String { entry.fileName }

    private func compare() {
        openWindow(id: ComparisonWindow.id, value: entry.id)
    }

    private func runAgain(newSeed: Bool) {
        generator.rerun(entry, newSeed: newSeed, in: conversation, library: library, engine: engine)
    }

    /// Depth or upscaling of this result, as a new entry of the same conversation.
    private func process(_ workflow: Workflow) {
        generator.process(
            workflow, image: library.imageURL(entry.id), named: suggestedName,
            in: conversation, library: library, engine: engine
        )
    }
}

/// The prompt, or the input image and what is done with it.
private struct RequestBubble: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(Workspace.self) private var workspace
    let entry: Entry
    @State private var hovering = false
    @FocusState private var editFocused: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 6) {
            if let request = entry.request {
                // the core loop: refine the prompt and send it again as a new request
                Button {
                    workspace.reuse(request)
                } label: {
                    Image(systemName: "pencil").frame(width: 22, height: 22).contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .focused($editFocused)
                .opacity(hovering || editFocused ? 1 : 0)
                .help("Edit Prompt: put it and its settings back in the composer")
                .accessibilityLabel("Edit Prompt")
            }
            bubble
        }
        .onHover { hovering = $0 }
        .contextMenu {
            if let request = entry.request {
                Button("Edit Prompt") { workspace.reuse(request) }
                Button("Copy Prompt") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(request.prompt, forType: .string)
                }
            }
        }
    }

    private var bubble: some View {
        Group {
            if let request = entry.request {
                VStack(alignment: .trailing, spacing: 6) {
                    Text(request.prompt)
                        .textSelection(.enabled)
                    if !request.negativePrompt.isEmpty {
                        Text("Avoid: \(request.negativePrompt)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                HStack(spacing: 10) {
                    StoredImage(url: library.inputURL(entry.id), maxPixels: 160, placeholder: "photo")
                        .id(generator.storedInput == entry.id)  // the input is written just after the entry appears
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    VStack(alignment: .leading, spacing: 2) {
                        Label(entry.workflow.title, systemImage: entry.workflow.symbol)
                            .font(.callout.weight(.medium))
                        Text(entry.inputName ?? "image")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .frame(maxWidth: 560, alignment: .trailing)
    }
}

/// A result at display size; clicking opens Quick Look.
private struct ResultImage: View {
    let entry: Entry
    let url: URL
    let aspectRatio: CGFloat?
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            StoredImage(url: url, maxPixels: 1600, aspectRatio: aspectRatio)
                .frame(maxWidth: Self.width(aspectRatio: aspectRatio), maxHeight: Self.maxHeight, alignment: .leading)
                .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .draggable(url)
        .accessibilityLabel(entry.title)
        .accessibilityHint("Opens Quick Look")
        .help("Quick Look")
    }

    static let maxWidth: CGFloat = 640
    static let maxHeight: CGFloat = 520

    /// The width a result is shown at when its aspect ratio is known, so its caption lines up with it.
    static func width(aspectRatio: CGFloat?) -> CGFloat {
        guard let aspectRatio else { return maxWidth }
        return min(maxWidth, maxHeight * aspectRatio)
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


/// The running request: the iris at work in the frame its image will fill, with what the
/// engine reports. Stop is in the composer.
private struct GenerationCanvas: View {
    @Environment(Generator.self) private var generator
    let entry: Entry
    let aspectRatio: CGFloat?

    var body: some View {
        let progress = generator.active?.entry == entry.id ? generator.progress : Generator.Progress()
        let ratio = aspectRatio ?? 1
        let queued = generator.isQueued(entry.id)
        IrisCanvas(energy: queued ? 0.25 : 1, progress: progress.fraction, opaque: true) {
            Color.clear
                .aspectRatio(ratio, contentMode: .fit)
                .overlay { ProgressReadout(progress: progress, workflow: entry.workflow, queued: queued) }
        }
        .overlay(alignment: .bottom) {
            if let fraction = progress.fraction {
                GeometryReader { geometry in
                    Rectangle()
                        .fill(.white.opacity(0.75))
                        .frame(width: geometry.size.width * fraction, height: 2)
                        .animation(.easeOut(duration: 0.4), value: fraction)
                }
                .frame(height: 2)
                .accessibilityHidden(true)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
        .frame(maxWidth: ResultImage.width(aspectRatio: aspectRatio), maxHeight: ResultImage.maxHeight, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(entry.workflow == .create ? "Generating" : entry.workflow.title)
        .accessibilityValue(accessibilityValue(progress))
    }

    private func accessibilityValue(_ progress: Generator.Progress) -> String {
        switch progress.phase {
        case .sampling where entry.workflow == .upscale, .finishing where entry.workflow == .upscale:
            "\(progress.percent ?? 0) percent, tile \(min(progress.step + 1, progress.steps)) of \(progress.steps)"
        case .sampling, .finishing: "\(progress.percent ?? 0) percent, step \(progress.step) of \(progress.steps)"
        case .preparing: "Preparing"
        case .waiting where generator.isQueued(entry.id): "Queued"
        case .waiting: "Waiting for the engine"
        case .working: "Working"
        }
    }
}

/// "65%" over "13 / 20 steps · about 1 min left", or the phase before and after sampling.
private struct ProgressReadout: View {
    let progress: Generator.Progress
    let workflow: Workflow
    /// Later in a batch: it starts when the one above is done.
    var queued = false

    var body: some View {
        VStack(spacing: 6) {
            if queued {
                Text("Queued").font(.title3.weight(.medium))
                Text("Starts when the image above is done").font(.callout).opacity(0.7)
            } else {
                readout
            }
        }
        .foregroundStyle(.white)
        .shadow(color: .black.opacity(0.5), radius: 4)  // legible over the fibers
        .multilineTextAlignment(.center)
        .padding()
        .animation(.easeOut(duration: 0.25), value: progress.step)
    }

    @ViewBuilder private var readout: some View {
        switch progress.phase {
        case .sampling, .finishing:
            Text("\(progress.percent ?? 0)%")
                .font(.system(size: 32, weight: .medium))
                .monospacedDigit()
                .contentTransition(.numericText())
            Text(detail)
                .font(.callout)
                .monospacedDigit()
                .opacity(0.75)
        case .waiting:
            Text("Waiting for the engine").font(.title3.weight(.medium))
        case .preparing:
            Text("Preparing").font(.title3.weight(.medium))
            Text(workflow == .create ? "Loading the model and reading the prompt" : "Loading the model")
                .font(.callout)
                .opacity(0.7)
        case .working:
            Text("Estimating depth").font(.title3.weight(.medium))
        }
    }

    /// "4 / 20 steps", or for an upscale the tile under way, "tile 5 of 9".
    private var detail: String {
        if progress.phase == .finishing {
            let unit = workflow == .upscale ? "tiles" : "steps"
            return "\(progress.steps) / \(progress.steps) \(unit) · finishing the image"
        }
        let steps = workflow == .upscale
            ? "tile \(progress.step + 1) of \(progress.steps)"
            : "\(progress.step) / \(progress.steps) steps"
        guard let seconds = progress.secondsLeft else { return steps }
        let left = Duration.seconds(seconds).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated, maximumUnitCount: 1))
        return "\(steps) · about \(left) left"
    }
}

private struct IconButton: View {
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
