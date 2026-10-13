import QuickLook
import SwiftUI

/// The gallery and its bottom prompt bar, beside project navigation and image details.
struct ContentView: View {
    @Environment(UpdateChecker.self) private var updates
    @Environment(\.openWindow) private var openWindow
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @State private var previewURL: URL?
    @SceneStorage("showsInspector") private var showsInspector = true

    var body: some View {
        @Bindable var workspace = workspace
        let scope = workspace.selection ?? .allImages
        let items = Gallery.items(of: library.conversations, in: scope, matching: workspace.searchText)
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 340)
        } detail: {
            VStack(spacing: 0) {
                IrisCanvas {
                    GalleryView(items: items, query: workspace.searchText, preview: { previewURL = $0 })
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .id(scope)
                }
                .clipped()
                Divider()
                Composer()
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                    .background(.bar)
            }
            // The column takes the height it is given. Its content's minimum height depends on its
            // width (wrapped notices), and reporting it sends the split view's layout into a loop
            // that AppKit ends by raising an exception.
            .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
            .acceptsImageDrops()
            .navigationTitle(title(scope))
            .navigationSubtitle(subtitle(items))
            .toolbarTitleMenu {
                if case let .project(id) = scope {
                    Button("Rename…") { workspace.naming = .project(id) }
                    Divider()
                    Button("Delete Project…", role: .destructive) { workspace.deleting = .project(id) }
                }
            }
            .inspector(isPresented: $showsInspector) {
                ImageInspector(preview: { previewURL = $0 })
                    .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                    .inspectorColumnWidth(min: 260, ideal: 300, max: 420)
            }
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    GallerySearchField(text: $workspace.searchText, focusRequests: workspace.searchFocusRequests)
                        .frame(minWidth: 160, idealWidth: 220, maxWidth: 320)
                }
                if let release = updates.available {
                    ToolbarItem(placement: .primaryAction) {
                        Button { openWindow(id: "updates") } label: {
                            Label("Iris \(release.displayVersion) available", systemImage: "arrow.down.circle")
                        }
                        .help("Iris \(release.displayVersion) is available")
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showsInspector.toggle()
                    } label: {
                        Label(showsInspector ? "Hide Details" : "Show Details", systemImage: "info.circle")
                    }
                    .keyboardShortcut("i", modifiers: .command)
                    .help(showsInspector ? "Hide Details (⌘I)" : "Show Details (⌘I)")
                }
            }
        }
        .quickLookPreview($previewURL, in: items.filter(\.entry.isDone).map { library.imageURL($0.id) })
        // what is selected was deleted, moved elsewhere or filtered out
        .onChange(of: items.map(\.id)) { _, ids in
            if let selected = workspace.selectedEntry, !ids.contains(selected) { workspace.selectedEntry = nil }
        }
        .onChange(of: library.projects.map(\.id)) { _, ids in
            if let project = workspace.project, !ids.contains(project) { workspace.selection = .allImages }
        }
        .libraryEditing()
        .alert(
            "Library",
            isPresented: Binding(get: { library.problem != nil }, set: { if !$0 { library.problem = nil } })
        ) {
            Button("OK") { library.problem = nil }
        } message: {
            Text(library.problem ?? "")
        }
    }

    private func title(_ scope: Workspace.Selection) -> String {
        switch scope {
        case .allImages: "All Images"
        case let .project(id): library.project(id)?.name ?? "Project"
        }
    }

    private func subtitle(_ items: [GalleryItem]) -> String {
        let images = items.filter(\.entry.isDone).count
        return images == 1 ? "1 image" : "\(images) images"
    }
}

/// Handles the native clear action even when the search field is no longer editing.
struct GallerySearchField: NSViewRepresentable {
    @Binding var text: String
    var focusRequests = 0

    func makeCoordinator() -> Coordinator { Coordinator(text: $text, focusRequests: focusRequests) }

    func makeNSView(context: Context) -> NSSearchField {
        let field = NSSearchField()
        field.placeholderString = "Search images"
        field.setAccessibilityLabel("Search images")
        field.sendsSearchStringImmediately = true
        field.sendsWholeSearchString = false
        field.delegate = context.coordinator
        field.target = context.coordinator
        field.action = #selector(Coordinator.searchChanged(_:))
        return field
    }

    func updateNSView(_ field: NSSearchField, context: Context) {
        context.coordinator.text = $text
        if field.stringValue != text { field.stringValue = text }
        if context.coordinator.focusRequests != focusRequests {
            context.coordinator.focusRequests = focusRequests
            field.window?.makeFirstResponder(field)
        }
    }

    @MainActor
    final class Coordinator: NSObject, NSSearchFieldDelegate {
        var text: Binding<String>
        var focusRequests: Int

        init(text: Binding<String>, focusRequests: Int) {
            self.text = text
            self.focusRequests = focusRequests
        }

        @objc func searchChanged(_ field: NSSearchField) {
            text.wrappedValue = field.stringValue
        }

        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSSearchField else { return }
            searchChanged(field)
        }
    }
}
