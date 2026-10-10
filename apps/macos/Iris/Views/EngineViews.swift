import SwiftUI

/// The engine's state as a dot and a word, in the sidebar footer; when it is not running,
/// a click opens its settings.
struct EngineStatus: View {
    @Environment(EngineController.self) private var engine
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        switch engine.state {
        case .ready:
            status(Text("Engine ready"), dot: .green)
        case .starting:
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("Starting engine")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .accessibilityElement(children: .combine)
        case .stopped, .failed:
            Button { openSettings() } label: {
                status(Text("Engine offline"), dot: .orange)
            }
            .buttonStyle(.plain)
            .help("Open the engine settings")
        }
    }

    private func status(_ text: Text, dot: Color) -> some View {
        HStack(spacing: 6) {
            Circle().fill(dot).frame(width: 7, height: 7).accessibilityHidden(true)
            text
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
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
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Starting the engine")
                    Text("The first start downloads the Iris-3B and Qwen3-VL weights, about 21 GB.")
                        .foregroundStyle(.secondary)
                    if let line = engine.logTail.last {
                        Text(line)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Spacer(minLength: 0)
            }
            .banner()
        case .stopped, .failed:
            VStack(alignment: .leading, spacing: 10) {
                Label(failure, systemImage: "exclamationmark.triangle")
                    .fixedSize(horizontal: false, vertical: true)
                if !engine.logTail.isEmpty {
                    DisclosureGroup("Details") {
                        Text(engine.logTail.suffix(6).joined(separator: "\n"))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 4)
                    }
                }
                HStack {
                    Button("Try Again") { Task { await engine.restart() } }
                        .disabled(engine.isRestarting)
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
            .background(.fieldBackground, in: RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous).strokeBorder(.separator))
    }
}
