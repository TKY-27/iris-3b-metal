import SwiftUI

/// Daily results with selection and Quick Look.
struct GalleryView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let items: [GalleryItem]
    let query: String
    let preview: (URL) -> Void
    @State private var columns = 1
    @FocusState private var focused: Bool

    private static let minimumWidth: CGFloat = 176
    private static let spacing: CGFloat = 16

    var body: some View {
        if !items.isEmpty {
            grid(Gallery.sections(items))
        } else if !query.trimmingCharacters(in: .whitespaces).isEmpty {
            ContentUnavailableView.search(text: query)
        } else {
            EmptyGallery(project: library.project(workspace.project))
        }
    }

    private func grid(_ sections: [GallerySection]) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 28) {
                    ForEach(sections) { section in
                        VStack(alignment: .leading, spacing: 12) {
                            Text(section.title)
                                .font(.headline)
                                .accessibilityAddTraits(.isHeader)
                            LazyVGrid(
                                columns: [GridItem(
                                    .adaptive(minimum: Self.minimumWidth, maximum: Self.minimumWidth * 1.6),
                                    spacing: Self.spacing, alignment: .top
                                )],
                                alignment: .leading, spacing: 20
                            ) {
                                ForEach(section.items) { card($0).id($0.id) }
                            }
                        }
                    }
                }
                .onGeometryChange(for: CGFloat.self, of: \.size.width) { width in
                    // Match the adaptive grid's column count.
                    columns = max(1, Int((width + Self.spacing) / (Self.minimumWidth + Self.spacing)))
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
            }
            .focusable()
            .focused($focused)
            .focusEffectDisabled()  // Selection indicates keyboard focus.
            .onKeyPress(keys: [.leftArrow, .rightArrow, .upArrow, .downArrow]) { press in
                let move: Gallery.Move = switch press.key {
                case .leftArrow: .left
                case .rightArrow: .right
                case .upArrow: .up
                default: .down
                }
                workspace.selectedEntry = Gallery.neighbor(
                    of: workspace.selectedEntry, in: sections, columns: columns, moving: move
                )
                return .handled
            }
            .onKeyPress(.space) {
                guard let selected = workspace.selectedEntry, items.contains(where: { $0.id == selected && $0.entry.isDone })
                else { return .ignored }
                preview(library.imageURL(selected))
                return .handled
            }
            .onDeleteCommand {
                guard let item = items.first(where: { $0.id == workspace.selectedEntry }) else { return }
                let actions = actions(for: item)
                if actions.canChange { actions.delete() }
            }
            .onChange(of: workspace.selectedEntry, initial: true) { _, selected in
                guard let selected else { return }
                withAnimation(reduceMotion ? nil : Theme.spring) { proxy.scrollTo(selected) }
            }
        }
    }

    private func card(_ item: GalleryItem) -> some View {
        let url = library.imageURL(item.id)
        return GalleryCard(entry: item.entry, isSelected: workspace.selectedEntry == item.id, isFocused: focused)
            .onTapGesture(count: 2) { if item.entry.isDone { preview(url) } }
            // Select immediately while awaiting a double click.
            .simultaneousGesture(TapGesture().onEnded {
                workspace.selectedEntry = item.id
                focused = true
            })
            .accessibilityAction { workspace.selectedEntry = item.id; focused = true }
            .accessibilityAction(named: String(localized: "Quick Look")) { if item.entry.isDone { preview(url) } }
            .draggable(url, if: item.entry.isDone)
            .contextMenu {
                EntryMenu(entry: item.entry, conversation: item.conversation, project: item.project, preview: preview)
            }
    }

    private func actions(for item: GalleryItem) -> EntryActions {
        EntryActions(
            entry: item.entry, conversation: item.conversation, library: library,
            generator: generator, engine: engine, workspace: workspace
        )
    }
}

private struct EmptyGallery: View {
    let project: Project?

    var body: some View {
        ContentUnavailableView {
            Label(project.map { String(localized: "No Images in “\($0.name)”") } ?? String(localized: "No Images Yet"), systemImage: "photo.on.rectangle")
        } description: {
            Text(String(localized: "Describe an image below to generate it, or add a photo for restoration or depth."))
        }
    }
}

extension View {
    /// Drag completed results into apps or projects.
    @ViewBuilder
    func draggable(_ url: URL, if enabled: Bool) -> some View {
        if enabled {
            draggable(url)
        } else {
            self
        }
    }
}
