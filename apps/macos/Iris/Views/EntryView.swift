import SwiftUI

/// One request (trailing, like a sent message) and its result, canvas or status (leading).
struct EntryView: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let entry: Entry
    let conversation: UUID
    /// The newest entry keeps its actions visible; older ones show them on hover.
    var isLatest = true
    let preview: (URL) -> Void
    @State private var hovering = false
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
        .onChange(of: entry.outcome) { old, _ in
            guard old == .running else { return }
            AccessibilityNotification.Announcement(entry.isDone ? "Image ready" : "The request ended").post()
        }
    }

    @ViewBuilder private var result: some View {
        switch entry.outcome {
        case .running:
            GenerationCanvas(entry: entry)
        case let .done(seconds):
            VStack(alignment: .leading, spacing: 8) {
                ResultImage(entry: entry, url: library.imageURL(entry.id), open: { preview(library.imageURL(entry.id)) })
                    .contextMenu { menu }
                HStack(spacing: 2) {
                    Text(caption(seconds: seconds))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        .lineLimit(1)
                        .help(exactSettings)
                        .textSelection(.enabled)
                    Spacer(minLength: 12)
                    actionBar
                        .opacity(isLatest || hovering || actionFocus != nil ? 1 : 0)  // shown on hover or keyboard focus
                }
                .frame(maxWidth: ResultImage.width(for: entry))
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
            return "\(entry.workflow == .depth ? "Relative depth, near is bright" : "Restored at 4×") · \(time)"
        }
        let quality = QualityPreset(steps: r.steps)?.title ?? "\(r.steps) steps"
        let adherence = AdherencePreset(cfgScale: r.cfgScale)?.title ?? "guidance \(r.cfgScale.formatted())"
        return "\(ImageSize(width: r.width, height: r.height).ratioLabel) · \(quality) · \(adherence) · \(time)"
    }

    private var exactSettings: String {
        guard let r = entry.request else { return "" }
        return "\(r.width) × \(r.height) · \(r.steps) steps · guidance \(r.cfgScale.formatted()) · seed \(r.seed)"
    }

    // MARK: - Actions

    private var canStart: Bool { !generator.isBusy && engine.state == .ready && !library.isReadOnly }

    private var canRunAgain: Bool {
        entry.request != nil || FileManager.default.fileExists(atPath: library.inputURL(entry.id).path)
    }

    private var actionBar: some View {
        HStack(spacing: 2) {
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
            if entry.workflow != .depth {
                IconButton("Upscale ×4", systemImage: Workflow.upscale.symbol) { process(.upscale) }
                    .disabled(!canStart)
                    .focused($actionFocus, equals: "upscale")
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
            Button("Upscale ×4") { process(.upscale) }.disabled(!canStart)
            Button("Estimate Depth") { process(.depth) }.disabled(!canStart)
        }
        Divider()
        Button("Delete", role: .destructive) { library.delete(entry: entry.id) }
            .disabled(entry.outcome == .running)
    }

    private var suggestedName: String {
        let words = entry.title.lowercased().split { !$0.isLetter && !$0.isNumber }.prefix(6)
        let stem = words.isEmpty ? "iris" : words.joined(separator: "-")
        return stem + (entry.workflow == .depth ? "-depth" : entry.workflow == .upscale ? "-x4" : "") + ".png"
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
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            StoredImage(url: url, maxPixels: 1600, aspectRatio: aspectRatio)
                .frame(maxWidth: Self.width(for: entry), maxHeight: Self.maxHeight, alignment: .leading)
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
    static func width(for entry: Entry) -> CGFloat {
        guard let request = entry.request else { return maxWidth }
        return min(maxWidth, maxHeight * CGFloat(request.width) / CGFloat(request.height))
    }

    private var aspectRatio: CGFloat? {
        entry.request.map { CGFloat($0.width) / CGFloat($0.height) }
    }
}

/// The running request: the iris at work in the frame its image will fill, with what the
/// engine reports. Stop is in the composer.
private struct GenerationCanvas: View {
    @Environment(Generator.self) private var generator
    let entry: Entry

    var body: some View {
        let progress = generator.active?.entry == entry.id ? generator.progress : Generator.Progress()
        let ratio = entry.request.map { CGFloat($0.width) / CGFloat($0.height) } ?? 1
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
        .frame(maxWidth: ResultImage.width(for: entry), maxHeight: ResultImage.maxHeight, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(entry.workflow == .create ? "Generating" : entry.workflow.title)
        .accessibilityValue(accessibilityValue(progress))
    }

    private func accessibilityValue(_ progress: Generator.Progress) -> String {
        switch progress.phase {
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
            Text("Loading the model and reading the prompt").font(.callout).opacity(0.7)
        case .working:
            Text(workflow == .depth ? "Estimating depth" : "Upscaling ×4").font(.title3.weight(.medium))
        }
    }

    private var detail: String {
        if progress.phase == .finishing { return "\(progress.steps) / \(progress.steps) steps · finishing the image" }
        let steps = "\(progress.step) / \(progress.steps) steps"
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
