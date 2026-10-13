import SwiftUI

/// Projects accept images dragged from the gallery.
struct Sidebar: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator

    var body: some View {
        @Bindable var workspace = workspace
        List(selection: $workspace.selection) {
            Section {
                HStack(spacing: 4) {
                    Label(String(localized: "All Images"), systemImage: "photo.on.rectangle")
                    Spacer(minLength: 0)
                    if generator.isBusy { WorkingIndicator() }
                }
                .tag(Workspace.Selection.allImages)
            }
            Section("Projects") {
                Button {
                    workspace.naming = .newProject(moving: nil)
                } label: {
                    Label(String(localized: "New Project"), systemImage: "plus")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help(String(localized: "New Project (⇧⌘N)"))
                ForEach(library.projects) { project in
                    ProjectRow(project: project, isWorking: runningProject == project.id)
                }
            }
        }
        .listStyle(.sidebar)
        .hardScrollEdge(.bottom)
        .safeAreaInset(edge: .bottom, spacing: 0) { SidebarFooter() }
        .onDeleteCommand {
            if case let .project(id) = workspace.selection { workspace.deleting = .project(id) }
        }
    }

    /// Project containing the active request.
    private var runningProject: UUID? {
        generator.active.flatMap { library.entry($0.entry)?.conversation.projectID }
    }
}

/// Select a project or file dropped images.
private struct ProjectRow: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let project: Project
    let isWorking: Bool
    @State private var targeted = false

    var body: some View {
        HStack(spacing: 4) {
            Label(project.name, systemImage: targeted ? "folder.fill" : "folder")
                .lineLimit(1)
            Spacer(minLength: 0)
            if isWorking { WorkingIndicator() }
        }
        .help(project.name)
        .tag(Workspace.Selection.project(project.id))
        .dropDestination(for: URL.self) { urls, _ in
            let entries = urls.compactMap(library.entryID(forImage:))
            for entry in entries { library.move(entry: entry, to: project.id) }
            return !entries.isEmpty
        } isTargeted: { targeted = $0 }
        .contextMenu {
            Button("Rename…") { workspace.naming = .project(project.id) }
            Divider()
            Button(String(localized: "Delete Project…"), role: .destructive) { workspace.deleting = .project(project.id) }
        }
    }
}

private struct WorkingIndicator: View {
    var body: some View {
        ProgressView()
            .controlSize(.mini)
            .accessibilityLabel("Generating")
    }
}

/// The engine's state and Settings, always at the bottom of the sidebar, aligned with the rows.
private struct SidebarFooter: View {
    var body: some View {
        HStack(spacing: 8) {
            SettingsLink {
                HStack(spacing: 8) {
                    Image(systemName: "gearshape").frame(width: 18)
                    Text(String(localized: "Settings"))
                }
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help(String(localized: "Settings (⌘,)"))
            Spacer(minLength: 0)
            EngineStatus()
        }
        .padding(.leading, 20)
        .padding(.trailing, 14)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }
}

extension View {
    /// A crisp edge where content scrolls under chrome, on systems that have it.
    @ViewBuilder
    func hardScrollEdge(_ edges: Edge.Set) -> some View {
        if #available(macOS 26, *) {
            scrollEdgeEffectStyle(.hard, for: edges)
        } else {
            self
        }
    }
}
