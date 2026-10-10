import SwiftUI

struct Canvas: View {
    @Environment(EngineController.self) private var engine
    @Environment(Generator.self) private var generator
    let item: HistoryItem?

    var body: some View {
        Group {
            if case let .failed(message) = engine.state {
                EngineFailure(message: message)
            } else if engine.state == .starting {
                ContentUnavailableView {
                    Label("Starting the engine", systemImage: "hourglass")
                } description: {
                    Text(engine.logTail.last ?? "The first start downloads the Iris-3B and Qwen3-VL weights (about 21 GB).")
                        .monospacedDigit()
                }
            } else if case let .failed(message) = generator.phase {
                ContentUnavailableView {
                    Label("Generation failed", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(message)
                } actions: {
                    Button("Dismiss") { generator.dismissError() }
                }
            } else if generator.isBusy {
                GenerationProgress(phase: generator.phase)
            } else if let item {
                ResultImage(item: item)
            } else {
                ContentUnavailableView("No image yet", systemImage: emptySymbol, description: Text(emptyHint))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptySymbol: String {
        switch generator.workflow {
        case .create: "photo"
        case .depth: "square.3.layers.3d.down.right"
        case .upscale: "arrow.up.left.and.arrow.down.right"
        }
    }

    private var emptyHint: String {
        switch generator.workflow {
        case .create: "Describe an image below and press ⌘↩."
        case .depth: "Drop a photo here to estimate its relative depth."
        case .upscale: "Drop an image here to restore it at 4× its size."
        }
    }
}

private struct GenerationProgress: View {
    let phase: Generator.Phase

    var body: some View {
        VStack(spacing: 12) {
            switch phase {
            case let .sampling(step, total, secondsLeft):
                ProgressView(value: Double(step), total: Double(total))
                    .frame(maxWidth: 360)
                // one string, so it reads naturally and stays translatable as a whole
                Group {
                    if let secondsLeft {
                        let remaining = Duration.seconds(secondsLeft)
                        Text("Step \(step) of \(total) · \(remaining, format: .units(allowed: [.minutes, .seconds], width: .abbreviated)) left")
                    } else {
                        Text("Step \(step) of \(total)")
                    }
                }
                .font(.callout)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            default:
                ProgressView()
                Text("Preparing…")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
    }
}

private struct EngineFailure: View {
    @Environment(EngineController.self) private var engine
    @Environment(\.openSettings) private var openSettings
    let message: String

    var body: some View {
        ContentUnavailableView {
            Label("The engine is not running", systemImage: "exclamationmark.triangle")
        } description: {
            VStack(spacing: 10) {
                Text(message)
                if !engine.logTail.isEmpty {
                    Text(engine.logTail.suffix(6).joined(separator: "\n"))
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .frame(maxWidth: 520, alignment: .leading)
                }
            }
        } actions: {
            Button("Try Again") { Task { await engine.restart() } }
                .buttonStyle(.borderedProminent)
            Button("Settings…") { openSettings() }
        }
    }
}
