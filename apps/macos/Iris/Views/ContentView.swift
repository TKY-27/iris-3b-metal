import QuickLook
import SwiftUI

/// Gallery, composer, project navigation and inspector.
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
                if workspace.showsGallery {
                    IrisCanvas {
                        GalleryView(items: items, query: workspace.searchText, preview: { previewURL = $0 })
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .id(scope)
                    }
                    .clipped()
                    Divider()
                } else {
                    Spacer(minLength: 0)
                    GenerationStatus()
                }
                Composer()
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                    .background(.bar)
                if !workspace.showsGallery { Spacer(minLength: 0) }
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
                    Button(String(localized: "Delete Project…"), role: .destructive) { workspace.deleting = .project(id) }
                }
            }
            .inspector(isPresented: Binding(
                get: { showsInspector && workspace.showsGallery },
                set: { if workspace.showsGallery { showsInspector = $0 } }
            )) {
                ImageInspector(preview: { previewURL = $0 })
                    .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                    .inspectorColumnWidth(min: 260, ideal: 300, max: 420)
            }
            .toolbar {
                if workspace.showsGallery {
                    ToolbarItem(placement: .automatic) {
                        GallerySearchField(
                            text: $workspace.searchText, focusPending: workspace.searchFocusPending,
                            onFocused: workspace.didFocusSearch
                        )
                            .frame(minWidth: 160, idealWidth: 220, maxWidth: 320)
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    GalleryVisibilityButton()
                }
                if let release = updates.available {
                    ToolbarItem(placement: .primaryAction) {
                        Button { openWindow(id: "updates") } label: {
                            Label(String(localized: "Iris \(release.displayVersion) available"), systemImage: "arrow.down.circle")
                        }
                        .help(String(localized: "Iris \(release.displayVersion) is available"))
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        if workspace.showsGallery {
                            showsInspector.toggle()
                        } else {
                            workspace.showsGallery = true
                            showsInspector = true
                        }
                    } label: {
                        Label(showsInspector && workspace.showsGallery ? String(localized: "Hide Details") : String(localized: "Show Details"), systemImage: "info.circle")
                    }
                    .keyboardShortcut("i", modifiers: .command)
                    .help(showsInspector && workspace.showsGallery ? String(localized: "Hide Details (⌘I)") : String(localized: "Show Details (⌘I)"))
                }
            }
        }
        .quickLookPreview($previewURL, in: items.filter(\.entry.isDone).map { library.imageURL($0.id) })
        // Clear selections outside the visible results.
        .onChange(of: items.map(\.id)) { _, ids in
            if let selected = workspace.selectedEntry, !ids.contains(selected) { workspace.selectedEntry = nil }
        }
        .onChange(of: library.projects.map(\.id)) { _, ids in
            if let project = workspace.project, !ids.contains(project) { workspace.selection = .allImages }
        }
        .libraryEditing()
        .alert(
            String(localized: "Library"),
            isPresented: Binding(get: { library.problem != nil }, set: { if !$0 { library.problem = nil } })
        ) {
            Button(String(localized: "OK")) { library.problem = nil }
        } message: {
            Text(library.problem ?? "")
        }
    }

    private func title(_ scope: Workspace.Selection) -> String {
        switch scope {
        case .allImages: String(localized: "All Images")
        case let .project(id): library.project(id)?.name ?? String(localized: "Project")
        }
    }

    private func subtitle(_ items: [GalleryItem]) -> String {
        let images = items.filter(\.entry.isDone).count
        return String(localized: "\(images) images")
    }
}

/// Handle native search clearing after focus changes.
struct GallerySearchField: NSViewRepresentable {
    @Binding var text: String
    var focusPending = false
    var onFocused: () -> Void = {}

    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }

    func makeNSView(context: Context) -> FocusableSearchField {
        let field = FocusableSearchField()
        field.placeholderString = String(localized: "Search images")
        field.setAccessibilityLabel(String(localized: "Search images"))
        field.sendsSearchStringImmediately = true
        field.sendsWholeSearchString = false
        field.delegate = context.coordinator
        field.target = context.coordinator
        field.action = #selector(Coordinator.searchChanged(_:))
        return field
    }

    func updateNSView(_ field: FocusableSearchField, context: Context) {
        context.coordinator.text = $text
        if field.stringValue != text { field.stringValue = text }
        field.onFocused = onFocused
        if focusPending { field.requestFocus() }
    }

    @MainActor
    final class Coordinator: NSObject, NSSearchFieldDelegate {
        var text: Binding<String>
        init(text: Binding<String>) {
            self.text = text
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

@MainActor
final class FocusableSearchField: NSSearchField {
    private var focusPending = false
    var onFocused: () -> Void = {}

    func requestFocus() {
        focusPending = true
        applyPendingFocus()
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        applyPendingFocus()
    }

    private func applyPendingFocus() {
        guard focusPending, let window else { return }
        focusPending = false
        DispatchQueue.main.async { [weak self, weak window] in
            guard let self, let window, self.window === window, window.makeFirstResponder(self) else { return }
            self.onFocused()
        }
    }
}
