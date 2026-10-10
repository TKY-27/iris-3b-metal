import QuickLook
import SwiftUI

/// Every finished result across conversations, newest first; opening one shows it in its conversation.
struct AllImagesView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @State private var previewURL: URL?

    var body: some View {
        let results = library.finishedEntries
        Group {
            if results.isEmpty {
                ContentUnavailableView(
                    "No Images Yet", systemImage: "photo.on.rectangle.angled",
                    description: Text("Results from every conversation appear here.")
                )
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 12)], spacing: 12) {
                        ForEach(results, id: \.entry.id) { result in
                            cell(result.entry, in: result.conversation)
                        }
                    }
                    .padding(20)
                }
            }
        }
        .navigationTitle("All Images")
        .navigationSubtitle("")
        .quickLookPreview($previewURL, in: results.map { library.imageURL($0.entry.id) })
    }

    private func cell(_ entry: Entry, in conversation: UUID) -> some View {
        let url = library.imageURL(entry.id)
        return Button {
            workspace.show(conversation, entry: entry.id)
        } label: {
            Color.clear
                .aspectRatio(1, contentMode: .fit)
                .overlay {
                    StoredImage(url: url, maxPixels: 480)
                        .scaledToFill()
                }
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .help(entry.title)
        .accessibilityLabel(entry.title)
        .accessibilityHint("Shows the image in its conversation")
        .draggable(url)
        .contextMenu {
            Button("Show in Conversation") { workspace.show(conversation, entry: entry.id) }
            Button("Quick Look") { previewURL = url }
            Button("Copy Image") { ImageActions.copy(url) }
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
    }
}
