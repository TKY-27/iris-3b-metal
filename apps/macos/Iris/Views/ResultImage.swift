import AppKit
import SwiftUI

struct ResultImage: View {
    @Environment(History.self) private var history
    @Environment(Generator.self) private var generator
    let item: HistoryItem

    var body: some View {
        VStack(spacing: 10) {
            if let image = NSImage(contentsOf: item.imageURL) {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .draggable(item.imageURL)
                    .contextMenu { actions }
                    .accessibilityLabel(item.request.prompt)
            }
            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .textSelection(.enabled)
        }
        .padding(20)
    }

    private var caption: String {
        let r = item.request
        let time = Duration.seconds(item.seconds).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated))
        return "\(r.width) × \(r.height) · \(r.steps) steps · guidance \(r.cfgScale.formatted()) · seed \(r.seed) · \(time)"
    }

    @ViewBuilder private var actions: some View {
        Button("Copy Image") {
            NSPasteboard.general.clearContents()
            if let image = NSImage(contentsOf: item.imageURL) {
                NSPasteboard.general.writeObjects([image])
            }
        }
        Button("Save As…", action: saveAs)
        Button("Show in Finder") {
            NSWorkspace.shared.activateFileViewerSelecting([item.imageURL])
        }
        Divider()
        Button("Reuse Settings") { generator.reuse(item.request) }
        Divider()
        Button("Delete", role: .destructive) { history.remove(item) }
    }

    private func saveAs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = filename
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)  // the panel already confirmed replacing it
            }
            try FileManager.default.copyItem(at: item.imageURL, to: url)
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    private var filename: String {
        let words = item.request.prompt.lowercased().split { !$0.isLetter && !$0.isNumber }.prefix(6)
        return (words.isEmpty ? "iris" : words.joined(separator: "-")) + ".png"
    }
}
