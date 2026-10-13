import SwiftUI

/// A result with progress and settings.
struct GalleryCard: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let entry: Entry
    let isSelected: Bool
    /// Whether gallery keyboard navigation is active.
    let isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Color.clear
                .aspectRatio(1, contentMode: .fit)
                .overlay { tile }
                .background(.quaternary.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: Theme.tileRadius, style: .continuous))
                .overlay {
                    // Keep selection visible around any image.
                    if isSelected {
                        RoundedRectangle(cornerRadius: Theme.tileRadius + 5, style: .continuous)
                            .strokeBorder(isFocused ? Color.accentColor : Color.secondary, lineWidth: 3)
                            .padding(-5)
                    }
                }
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.displayTitle)
                    .font(.callout)
                    .foregroundStyle(.primary)
                    .lineLimit(2, reservesSpace: true)
                // Separate captions keep long seeds visible.
                Text(entry.summary)
                    .lineLimit(1, reservesSpace: true)
                Text(entry.seedLabel ?? "")
                    .lineLimit(1, reservesSpace: true)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .monospacedDigit()
            .padding(.horizontal, 2)
        }
        .contentShape(Rectangle())
        .help(entry.displayTitle)
        .animation(reduceMotion ? nil : .easeOut(duration: 0.3), value: entry.outcome)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(entry.displayTitle)
        .accessibilityValue(accessibilityValue)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .onChange(of: entry.outcome) { old, _ in
            guard old == .running else { return }
            AccessibilityNotification.Announcement(entry.isDone ? String(localized: "Image ready") : String(localized: "The request ended")).post()
        }
    }

    private var accessibilityValue: String {
        switch entry.outcome {
        case .running where generator.active?.entry == entry.id:
            ProgressText.spoken(generator.progress, workflow: entry.workflow)
        case .running:
            generator.isQueued(entry.id) ? String(localized: "Queued") : String(localized: "Waiting")
        default:
            [entry.statusText, entry.summary.spokenSizes, entry.seedLabel].compactMap { $0 }.joined(separator: ", ")
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
            StatusTile(title: generator.isQueued(entry.id) ? String(localized: "Queued") : String(localized: "Waiting"), systemImage: "clock")
        case .failed:
            StatusTile(title: String(localized: "Failed"), systemImage: "exclamationmark.triangle")
        case .interrupted:
            StatusTile(title: String(localized: "Interrupted"), systemImage: "exclamationmark.triangle")
        case .cancelled:
            StatusTile(title: String(localized: "Stopped"), systemImage: "stop.circle")
        }
    }
}

extension Entry {
    /// The model, steps, or processing action.
    var summary: String {
        if let request { return String(localized: "\(request.model.title) · \(request.steps) steps") }
        return workflow == .depth ? String(localized: "Depth map") : String(localized: "Restored at 4×")
    }

    /// The generation seed caption.
    var seedLabel: String? {
        request.map { String(localized: "Seed \(String($0.seed))") }
    }

    /// Reserve space before loading the image.
    var aspectRatio: CGFloat? {
        request.map { CGFloat($0.width) / CGFloat($0.height) }
    }

    /// The status of an unfinished entry.
    var statusText: String? {
        switch outcome {
        case let .failed(message): failureText(message)
        case .cancelled: String(localized: "Stopped.")
        case .interrupted: String(localized: "Interrupted before the result arrived.")
        case .running, .done: nil
        }
    }

    private func failureText(_ message: String) -> String {
        switch message {
        case "This entry has no input.":
            return String(localized: "This entry has no input.")
        case "The engine could not finish this request.":
            return String(localized: "The engine could not finish this request.")
        case "The engine is not running.":
            return String(localized: "The engine is not running.")
        case "The engine is busy; try again when the current image finishes.":
            return String(localized: "The engine is busy; try again when the current image finishes.")
        default:
            if let model = request?.model,
               message == "This engine does not support \(model.fullTitle). Update the engine in Settings." {
                return String(localized: "This engine does not support \(model.fullTitle). Update the engine in Settings.")
            }
            return String(localized: "Request failed: \(message)")
        }
    }
}

/// Progress from completed engine steps.
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

/// A queued, failed, or stopped request.
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

/// Human-readable engine progress and remaining time.
struct ProgressText: Equatable {
    var headline: String
    var detail: String?

    init(_ progress: Generator.Progress, workflow: Workflow) {
        switch progress.phase {
        case .sampling, .finishing:
            headline = "\(progress.percent ?? 0)%"
            detail = Self.detail(progress, workflow: workflow)
        case .waiting:
            headline = String(localized: "Waiting for the engine")
        case .preparing:
            headline = String(localized: "Preparing")
            detail = workflow == .create ? String(localized: "Loading the model and reading the prompt") : String(localized: "Loading the model")
        case .working:
            headline = String(localized: "Estimating depth")
        }
    }

    /// Step or tile progress, with remaining time.
    private static func detail(_ progress: Generator.Progress, workflow: Workflow) -> String {
        if progress.phase == .finishing {
            return workflow == .upscale
                ? String(localized: "\(progress.steps) / \(progress.steps) tiles · finishing the image")
                : String(localized: "\(progress.steps) / \(progress.steps) steps · finishing the image")
        }
        let steps = workflow == .upscale
            ? String(localized: "tile \(progress.step + 1) of \(progress.steps)")
            : String(localized: "\(progress.step) / \(progress.steps) steps")
        // Avoid rounding remaining time to zero.
        guard let seconds = progress.secondsLeft, seconds >= 1 else { return steps }
        let left = Duration.seconds(seconds).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated, maximumUnitCount: 1).locale(.iris))
        return String(localized: "\(steps) · about \(left) left")
    }

    /// Spoken progress for VoiceOver.
    static func spoken(_ progress: Generator.Progress, workflow: Workflow) -> String {
        switch progress.phase {
        case .sampling where workflow == .upscale, .finishing where workflow == .upscale:
            String(localized: "\(progress.percent ?? 0) percent, tile \(min(progress.step + 1, progress.steps)) of \(progress.steps)")
        case .sampling, .finishing: String(localized: "\(progress.percent ?? 0) percent, step \(progress.step) of \(progress.steps)")
        case .preparing: String(localized: "Preparing")
        case .waiting: String(localized: "Waiting for the engine")
        case .working: String(localized: "Working")
        }
    }
}
