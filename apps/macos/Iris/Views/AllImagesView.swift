import QuickLook
import SwiftUI

/// Every finished result across chats, newest first. Click selects, Space opens Quick Look,
/// and a double click or Return shows the image in its chat.
struct AllImagesView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @State private var previewURL: URL?
    @State private var selected: UUID?
    @FocusState private var gridFocused: Bool

    var body: some View {
        let results = library.finishedEntries
        Group {
            if results.isEmpty {
                ContentUnavailableView(
                    "No Images Yet", systemImage: "photo.on.rectangle",
                    description: Text("Results from every chat appear here.")
                )
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 168), spacing: 12)], spacing: 12) {
                        ForEach(results, id: \.entry.id) { result in
                            cell(result.entry, in: result.conversation)
                        }
                    }
                    .padding(24)
                }
                .focusable()
                .focused($gridFocused)
                .focusEffectDisabled()
                .onKeyPress(.space) {
                    guard let selected else { return .ignored }
                    previewURL = library.imageURL(selected)
                    return .handled
                }
                .onKeyPress(.return) {
                    guard let selected, let result = results.first(where: { $0.entry.id == selected }) else { return .ignored }
                    workspace.show(result.conversation, entry: selected)
                    return .handled
                }
            }
        }
        .navigationTitle("All Images")
        .navigationSubtitle(results.count == 1 ? "1 image" : "\(results.count) images")
        .quickLookPreview($previewURL, in: results.map { library.imageURL($0.entry.id) })
    }

    private func cell(_ entry: Entry, in conversation: UUID) -> some View {
        let url = library.imageURL(entry.id)
        let isSelected = selected == entry.id
        return Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                StoredImage(url: url, maxPixels: 480)
                    .scaledToFill()
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.accentColor, lineWidth: isSelected ? 3 : 0)
            }
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .onTapGesture(count: 2) { workspace.show(conversation, entry: entry.id) }
            .onTapGesture {
                selected = entry.id
                gridFocused = true
            }
            .help(entry.title)
            .accessibilityElement()
            .accessibilityLabel(entry.title)
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
            .accessibilityAction { workspace.show(conversation, entry: entry.id) }
            .accessibilityAction(named: "Quick Look") { previewURL = url }
            .draggable(url)
            .contextMenu {
                Button("Show in Chat") { workspace.show(conversation, entry: entry.id) }
                Button("Quick Look") { previewURL = url }
                Button("Copy Image") { ImageActions.copy(url) }
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
    }
}
