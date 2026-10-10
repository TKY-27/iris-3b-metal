import ImageIO
import SwiftUI

struct HistorySidebar: View {
    @Environment(History.self) private var history
    @Environment(Generator.self) private var generator
    @Binding var selection: HistoryItem.ID?

    var body: some View {
        List(selection: $selection) {
            ForEach(history.items) { item in
                HistoryRow(item: item)
                    .tag(item.id)
                    .contextMenu {
                        if let request = item.request {
                            Button("Reuse Settings") { generator.reuse(request) }
                        }
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([item.imageURL])
                        }
                        Divider()
                        Button("Delete", role: .destructive) { delete(item) }
                    }
            }
        }
        .onDeleteCommand {
            if let item = history.items.first(where: { $0.id == selection }) { delete(item) }
        }
        .overlay {
            if history.items.isEmpty {
                Text("Generated images appear here.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding()
            }
        }
    }

    private func delete(_ item: HistoryItem) {
        if selection == item.id { selection = nil }
        history.remove(item)
    }
}

private struct HistoryRow: View {
    let item: HistoryItem
    @State private var thumbnail: CGImage?

    var body: some View {
        HStack(spacing: 10) {
            Group {
                if let thumbnail {
                    Image(decorative: thumbnail, scale: 2).resizable().scaledToFill()
                } else {
                    Rectangle().fill(.quaternary)
                }
            }
            .frame(width: 44, height: 44)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 2) {
                Text(item.title)
                    .lineLimit(2)
                Text(item.created, format: .relative(presentation: .named))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .task(id: item.id) {
            thumbnail = await Self.thumbnail(of: item.imageURL)
        }
    }

    private static func thumbnail(of url: URL) async -> CGImage? {
        await Task.detached(priority: .utility) {
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 128,
            ]
            return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }.value
    }
}
