import SwiftUI

/// A result in the gallery: the image, or its progress or status, over its prompt and its seed
/// and steps.
struct GalleryCard: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let entry: Entry
    let isSelected: Bool
    /// Whether the keyboard acts on the gallery, which the selection's color shows.
    let isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Color.clear
                .aspectRatio(1, contentMode: .fit)
                .overlay { tile }
                .background(.quaternary.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: Theme.tileRadius, style: .continuous))
                .overlay {
                    // around the image, not on it, so it shows on any picture
                    if isSelected {
                        RoundedRectangle(cornerRadius: Theme.tileRadius + 5, style: .continuous)
                            .strokeBorder(isFocused ? Color.accentColor : Color.secondary, lineWidth: 3)
                            .padding(-5)
                    }
                }
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.title)
                    .font(.callout)
                    .lineLimit(2, reservesSpace: true)
                Text(entry.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .lineLimit(1)
            }
            .padding(.horizontal, 2)
        }
        .contentShape(Rectangle())
        .help(entry.title)
        .animation(reduceMotion ? nil : .easeOut(duration: 0.3), value: entry.outcome)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(entry.title)
        .accessibilityValue(accessibilityValue)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .onChange(of: entry.outcome) { old, _ in
            guard old == .running else { return }
            AccessibilityNotification.Announcement(entry.isDone ? "Image ready" : "The request ended").post()
        }
    }

    private var accessibilityValue: String {
        switch entry.outcome {
        case .running where generator.active?.entry == entry.id:
            ProgressText.spoken(generator.progress, workflow: entry.workflow)
        case .running:
            generator.isQueued(entry.id) ? "Queued" : "Waiting"
        default:
            [entry.statusText, entry.summary.spokenSizes].compactMap { $0 }.joined(separator: ", ")
        }
    }

    @ViewBuilder private var tile: some View {
        switch entry.outcome {
        case .done:
            StoredImage(url: library.imageURL(entry.id), maxPixels: 640, aspectRatio: entry.aspectRatio)
                .transition(.opacity)
        case .running where generator.active?.entry == entry.id:
            RunningTile(workflow: entry.workflow)
        case .running:
            StatusTile(title: generator.isQueued(entry.id) ? "Queued" : "Waiting", systemImage: "clock")
        case .failed:
            StatusTile(title: "Failed", systemImage: "exclamationmark.triangle")
        case .interrupted:
            StatusTile(title: "Interrupted", systemImage: "exclamationmark.triangle")
        case .cancelled:
            StatusTile(title: "Stopped", systemImage: "stop.circle")
        }
    }
}

extension Entry {
    /// A card's second line: the seed and steps of a created image, or what was done to an input.
    var summary: String {
        if let request { return "\(request.model.title) · \(request.steps) steps · \(request.seed)" }
        return workflow == .depth ? "Depth map" : "Restored at 4×"
    }

    /// Known before the file is read for a created image, so its frame never jumps.
    var aspectRatio: CGFloat? {
        request.map { CGFloat($0.width) / CGFloat($0.height) }
    }

    /// What a card or the inspector says about an entry that has no image.
    var statusText: String? {
        switch outcome {
        case let .failed(message): message
        case .cancelled: "Stopped."
        case .interrupted: "Interrupted before the result arrived."
        case .running, .done: nil
        }
    }
}

/// A running request uses the engine's completed-step progress.
private struct RunningTile: View {
    @Environment(Generator.self) private var generator
    let workflow: Workflow

    var body: some View {
        let progress = generator.progress
        let text = ProgressText(progress, workflow: workflow)
        IrisCanvas(energy: 1, progress: progress.fraction, opaque: true, center: UnitPoint(x: 0.5, y: 0.37)) {
            VStack(spacing: 6) {
                Spacer(minLength: 0)
                Text(text.headline).font(.headline)
                if let detail = text.detail {
                    Text(detail).font(.caption).foregroundStyle(.white.opacity(0.85))
                }
                if let fraction = progress.fraction {
                    ProgressView(value: fraction).tint(.white)
                }
            }
            .foregroundStyle(.white)
            .monospacedDigit()
            .multilineTextAlignment(.center)
            .padding(16)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .accessibilityHidden(true)
    }
}

/// A request without an image: queued, failed or stopped.
private struct StatusTile: View {
    let title: String
    let systemImage: String

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage).font(.title2)
            Text(title).font(.callout)
        }
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// What the engine reports about a running request, in words: "65%" over
/// "13 / 20 steps · about 1 min left", or the phase before and after sampling.
struct ProgressText: Equatable {
    var headline: String
    var detail: String?

    init(_ progress: Generator.Progress, workflow: Workflow) {
        switch progress.phase {
        case .sampling, .finishing:
            headline = "\(progress.percent ?? 0)%"
            detail = Self.detail(progress, workflow: workflow)
        case .waiting:
            headline = "Waiting for the engine"
        case .preparing:
            headline = "Preparing"
            detail = workflow == .create ? "Loading the model and reading the prompt" : "Loading the model"
        case .working:
            headline = "Estimating depth"
        }
    }

    /// "4 / 20 steps", or for an upscale the tile under way, "tile 5 of 9".
    private static func detail(_ progress: Generator.Progress, workflow: Workflow) -> String {
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

    /// The progress as VoiceOver should say it.
    static func spoken(_ progress: Generator.Progress, workflow: Workflow) -> String {
        switch progress.phase {
        case .sampling where workflow == .upscale, .finishing where workflow == .upscale:
            "\(progress.percent ?? 0) percent, tile \(min(progress.step + 1, progress.steps)) of \(progress.steps)"
        case .sampling, .finishing: "\(progress.percent ?? 0) percent, step \(progress.step) of \(progress.steps)"
        case .preparing: "Preparing"
        case .waiting: "Waiting for the engine"
        case .working: "Working"
        }
    }
}
