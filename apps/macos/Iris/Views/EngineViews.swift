import SwiftUI

/// The engine's state in the toolbar.
struct EngineStatus: View {
    @Environment(EngineController.self) private var engine

    var body: some View {
        Group {
            switch engine.state {
            case .ready:
                Label("Engine ready", systemImage: "circle.fill")
                    .foregroundStyle(.green)
            case .starting:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Starting engine")
                }
            case .stopped, .failed:
                Label("Engine not running", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
        }
        .labelStyle(.titleAndIcon)
        .font(.caption)
    }
}

/// Above the composer while the engine is starting or has failed, with what to do about it.
struct EngineBanner: View {
    @Environment(EngineController.self) private var engine
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        switch engine.state {
        case .ready:
            EmptyView()
        case .starting:
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Starting the engine")
                    Text(engine.logTail.last ?? "The first start downloads the Iris-3B and Qwen3-VL weights, about 21 GB.")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
            }
            .banner()
        case .stopped, .failed:
            VStack(alignment: .leading, spacing: 8) {
                Label(failure, systemImage: "exclamationmark.triangle")
                    .fixedSize(horizontal: false, vertical: true)
                if !engine.logTail.isEmpty {
                    Text(engine.logTail.suffix(6).joined(separator: "\n"))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(6)
                }
                HStack {
                    Button("Try Again") { Task { await engine.restart() } }
                    Button("Engine Settings…") { openSettings() }
                }
                .controlSize(.small)
            }
            .banner()
        }
    }

    private var failure: String {
        if case let .failed(message) = engine.state { message } else { "The engine is not running." }
    }
}

private extension View {
    func banner() -> some View {
        padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .font(.callout)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
    }
}
