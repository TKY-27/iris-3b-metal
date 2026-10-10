import SwiftUI

/// One request (trailing, like a sent message) and its result or status (leading).
struct EntryView: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    let entry: Entry
    let conversation: UUID
    let preview: (URL) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            RequestBubble(entry: entry)
                .frame(maxWidth: .infinity, alignment: .trailing)
            result
        }
    }

    @ViewBuilder private var result: some View {
        switch entry.outcome {
        case .running:
            RunningCard()
        case let .done(seconds):
            VStack(alignment: .leading, spacing: 8) {
                ResultImage(entry: entry, url: library.imageURL(entry.id), open: { preview(library.imageURL(entry.id)) })
                    .contextMenu { menu }
                HStack(spacing: 2) {
                    Text(caption(seconds: seconds))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        .textSelection(.enabled)
                    Spacer(minLength: 12)
                    actionBar
                }
                .frame(maxWidth: ResultImage.width(for: entry))
            }
        case .failed, .cancelled, .interrupted:
            HStack(spacing: 12) {
                Label(statusText, systemImage: entry.outcome == .cancelled ? "stop.circle" : "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                if canRunAgain {
                    Button("Try Again") { runAgain(newSeed: false) }
                        .disabled(!canStart)
                }
                Button("Delete", role: .destructive) { library.delete(entry: entry.id) }
            }
            .font(.callout)
            .padding(12)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
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

    private func caption(seconds: Double) -> String {
        let time = Duration.seconds(seconds).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated))
        guard let r = entry.request else {
            return "\(entry.workflow == .depth ? "Relative depth, near is bright" : "Restored at 4×") · \(time)"
        }
        return "\(r.width) × \(r.height) · \(r.steps) steps · guidance \(r.cfgScale.formatted()) · seed \(r.seed) · \(time)"
    }

    // MARK: - Actions

    private var canStart: Bool { !generator.isBusy && engine.state == .ready }

    private var canRunAgain: Bool {
        entry.request != nil || FileManager.default.fileExists(atPath: library.inputURL(entry.id).path)
    }

    private var actionBar: some View {
        HStack(spacing: 2) {
            IconButton("Copy", systemImage: "doc.on.doc") { ImageActions.copy(library.imageURL(entry.id)) }
            IconButton("Save As…", systemImage: "square.and.arrow.down") {
                ImageActions.saveAs(library.imageURL(entry.id), suggestedName: suggestedName)
            }
            if entry.workflow == .create {
                IconButton("Generate Again with a New Seed", systemImage: "arrow.clockwise") { runAgain(newSeed: true) }
                    .disabled(!canStart)
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
            .fixedSize()
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
                Button("Generate Again with a New Seed") { runAgain(newSeed: true) }.disabled(!canStart)
            }
            Button("Reuse Prompt and Settings") {
                workspace.workflow = .create
                workspace.prompt = request.prompt
                workspace.settings.reuse(request)
            }
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
    let entry: Entry

    var body: some View {
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
                    Label(entry.workflow.title, systemImage: entry.workflow.symbol)
                        .font(.callout.weight(.medium))
                    InputThumbnail(url: library.inputURL(entry.id), name: entry.inputName ?? "image")
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 16))
        .frame(maxWidth: 560, alignment: .trailing)
    }
}

private struct InputThumbnail: View {
    let url: URL
    let name: String

    var body: some View {
        HStack(spacing: 8) {
            StoredImage(url: url, maxPixels: 160, placeholder: "photo")
                .frame(width: 48, height: 48)
                .clipShape(RoundedRectangle(cornerRadius: 6))
            Text(name)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
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
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .draggable(url)
        .accessibilityLabel(entry.title)
        .accessibilityHint("Opens Quick Look")
        .help("Quick Look")
    }

    static let maxWidth: CGFloat = 640
    static let maxHeight: CGFloat = 560

    /// The width a result is shown at when its aspect ratio is known, so its caption lines up with it.
    static func width(for entry: Entry) -> CGFloat {
        guard let request = entry.request else { return maxWidth }
        return min(maxWidth, maxHeight * CGFloat(request.width) / CGFloat(request.height))
    }

    private var aspectRatio: CGFloat? {
        entry.request.map { CGFloat($0.width) / CGFloat($0.height) }
    }
}

/// Progress of the running entry; Stop is in the composer.
private struct RunningCard: View {
    @Environment(Generator.self) private var generator

    var body: some View {
        HStack(spacing: 14) {
            let progress = generator.progress
            if progress.step > 0 {
                ProgressView(value: Double(progress.step), total: Double(max(progress.steps, 1)))
                    .frame(width: 220)
            } else {
                ProgressView().controlSize(.small)
            }
            status(progress)
                .font(.callout)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            Spacer(minLength: 0)
        }
        .padding(14)
        .frame(maxWidth: 640)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    // one string per state, so each reads naturally and stays translatable as a whole
    @ViewBuilder private func status(_ progress: Generator.Progress) -> some View {
        if progress.step == 0 {
            Text("Waiting for the engine…")
        } else if let secondsLeft = progress.secondsLeft {
            let remaining = Duration.seconds(secondsLeft)
            Text("Step \(progress.step) of \(progress.steps) · about \(remaining, format: .units(allowed: [.minutes, .seconds], width: .abbreviated)) left")
        } else {
            Text("Step \(progress.step) of \(progress.steps)")
        }
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
