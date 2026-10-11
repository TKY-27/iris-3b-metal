import QuickLook
import SwiftUI

/// Every finished result across chats, newest first. Click selects, Space opens Quick Look,
/// and a double click or Return shows the image in its chat.
struct AllImagesView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(\.openWindow) private var openWindow
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
                if entry.request == nil, FileManager.default.fileExists(atPath: library.inputURL(entry.id).path) {
                    Button("Compare with Original") { openWindow(id: ComparisonWindow.id, value: entry.id) }
                }
                if entry.workflow != .depth {
                    Divider()
                    if canUpscale(entry) {
                        Button(Workflow.upscale.title) { process(.upscale, entry, in: conversation) }
                            .disabled(!canStart)
                    }
                    Button("Estimate Depth") { process(.depth, entry, in: conversation) }
                        .disabled(!canStart)
                }
            }
    }

    private var canStart: Bool { !generator.isBusy && engine.state == .ready && !library.isReadOnly }

    /// As in the chat: only when the result would come out larger.
    private func canUpscale(_ entry: Entry) -> Bool {
        let size = entry.request.map { CGSize(width: $0.width, height: $0.height) }
            ?? ImageFiles.pixelSize(of: library.imageURL(entry.id))
        return size.map { Upscaling.refusal(for: $0) == nil } ?? false
    }

    /// Runs on the result as a new entry of its chat, and shows it there.
    private func process(_ workflow: Workflow, _ entry: Entry, in conversation: UUID) {
        let started = generator.process(
            workflow, image: library.imageURL(entry.id), named: entry.fileName,
            in: conversation, library: library, engine: engine
        )
        if let started { workspace.show(started) }
    }
}
