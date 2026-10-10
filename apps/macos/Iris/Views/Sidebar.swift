import SwiftUI

/// New Chat and All Images, pinned chats, projects with their chats, and recent chats by date,
/// with search. Chats can be dragged onto a project to file them there.
struct Sidebar: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator
    @State private var query = ""
    /// Collapsed projects, kept with the window across launches.
    @SceneStorage("collapsedProjects") private var collapsedStorage = ""

    var body: some View {
        @Bindable var workspace = workspace
        List(selection: $workspace.selection) {
            if query.isEmpty {
                Section {
                    Label("New Chat", systemImage: "square.and.pencil")
                        .tag(Workspace.Selection.newChat)
                        .help("New Chat (⌘N)")
                    Label("All Images", systemImage: "photo.on.rectangle")
                        .tag(Workspace.Selection.allImages)
                }
                if !library.pinnedConversations.isEmpty {
                    Section("Pinned") {
                        ForEach(library.pinnedConversations) { ChatRow(conversation: $0) }
                    }
                }
                projectsSection
                ForEach(RecentGroup.group(library.recentConversations)) { group in
                    Section(group.title) {
                        ForEach(group.conversations) { ChatRow(conversation: $0) }
                    }
                }
            } else {
                Section("Results") {
                    ForEach(searchResults) { ChatRow(conversation: $0) }
                }
            }
        }
        .listStyle(.sidebar)
        .hardScrollEdge(.bottom)
        .searchable(text: $query, placement: .sidebar, prompt: "Search chats")
        .safeAreaInset(edge: .bottom, spacing: 0) { SidebarFooter() }
        .onDeleteCommand {
            if let id = workspace.conversationID, !isRunning(id) { workspace.deleting = .conversation(id) }
        }
        .overlay {
            if !query.isEmpty, searchResults.isEmpty {
                ContentUnavailableView.search(text: query)
            }
        }
    }

    // Plain rows, not DisclosureGroup: a sidebar drops a DisclosureGroup row whose label is
    // selectable, and a project must be selectable to open its page.
    private var projectsSection: some View {
        Section("Projects") {
            Button {
                workspace.naming = .newProject(moving: nil)
            } label: {
                Label("New Project", systemImage: "plus")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("New Project (⇧⌘N)")
            ForEach(library.projects) { project in
                let expanded = !collapsedProjects.contains(project.id)
                ProjectRow(project: project, expanded: expanded) { toggle(project.id, open: $0) }
                if expanded {
                    // a pinned chat is listed once, under Pinned; the project page lists them all
                    ForEach(library.conversations(in: project.id).filter { $0.pinned == nil }) {
                        ChatRow(conversation: $0, inProject: true)
                    }
                }
            }
        }
    }

    private var searchResults: [Conversation] {
        library.conversations.filter { $0.matches(query) }.sorted { $0.updated > $1.updated }
    }

    private var collapsedProjects: Set<UUID> {
        Set(collapsedStorage.split(separator: ",").compactMap { UUID(uuidString: String($0)) })
    }

    private func toggle(_ project: UUID, open: Bool) {
        var collapsed = collapsedProjects
        if open { collapsed.remove(project) } else { collapsed.insert(project) }
        collapsedStorage = collapsed.map(\.uuidString).sorted().joined(separator: ",")
    }

    private func isRunning(_ conversation: UUID) -> Bool {
        generator.isRunning(conversation)
    }
}

/// A project: selecting it opens its page, the chevron shows or hides its chats, and dropping
/// a chat on it files the chat there.
private struct ProjectRow: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let project: Project
    let expanded: Bool
    let setExpanded: (Bool) -> Void
    @State private var targeted = false
    @State private var hovering = false
    @FocusState private var chevronFocused: Bool

    var body: some View {
        HStack(spacing: 4) {
            Label(project.name, systemImage: targeted ? "folder.fill" : "folder")
                .lineLimit(1)
            Spacer(minLength: 0)
            Button {
                setExpanded(!expanded)
            } label: {
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .rotationEffect(.degrees(expanded ? 90 : 0))
                    .frame(width: 22, height: 22)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .focused($chevronFocused)
            // like Finder: shown on hover or focus, and always while collapsed so hidden chats stay findable
            .opacity(hovering || chevronFocused || !expanded ? 1 : 0)
            .help(expanded ? "Hide chats" : "Show chats")
            .accessibilityLabel("Chats in \(project.name)")
            .accessibilityValue(expanded ? "Expanded" : "Collapsed")
        }
        .onHover { hovering = $0 }
        .help(project.name)
        .tag(Workspace.Selection.project(project.id))
        .dropDestination(for: ChatReference.self) { chats, _ in
            for chat in chats { library.move(conversation: chat.id, to: project.id) }
            setExpanded(true)
            return !chats.isEmpty
        } isTargeted: { targeted = $0 }
        .contextMenu {
            Button("New Chat in Project") { workspace.newConversation(in: project.id) }
            Button("Rename…") { workspace.naming = .project(project.id) }
            Divider()
            Button("Delete Project…", role: .destructive) { workspace.deleting = .project(project.id) }
        }
    }
}

/// A chat in the sidebar, with its latest result as a thumbnail.
private struct ChatRow: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator
    let conversation: Conversation
    /// Indented under its project.
    var inProject = false

    var body: some View {
        // a Label, so the title shares the sidebar's text column with every other row
        HStack(spacing: 4) {
            Label {
                Text(conversation.title)
                    .lineLimit(1)
                    .truncationMode(.tail)
            } icon: {
                ChatThumbnail(conversation: conversation, side: 18)
            }
            Spacer(minLength: 0)
            if generator.isRunning(conversation.id) {
                ProgressView()
                    .controlSize(.mini)
                    .accessibilityLabel("Working")
            }
        }
        .padding(.leading, inProject ? 26 : 0)  // its thumbnail under the project name
        .help(conversation.title)
        .tag(Workspace.Selection.conversation(conversation.id))
        .draggable(ChatReference(id: conversation.id)) {
            Label(conversation.title, systemImage: "bubble.left").padding(6)
        }
        .contextMenu { ChatMenu(conversation: conversation) }
        .swipeActions(edge: .trailing) {
            Button(conversation.pinned == nil ? "Pin" : "Unpin", systemImage: conversation.pinned == nil ? "pin" : "pin.slash") {
                library.setPinned(conversation.pinned == nil, conversation: conversation.id)
            }
            .tint(.orange)
        }
    }
}

/// What can be done with a chat, in the sidebar and on a project page.
struct ChatMenu: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator
    let conversation: Conversation

    var body: some View {
        Button(conversation.pinned == nil ? "Pin" : "Unpin") {
            library.setPinned(conversation.pinned == nil, conversation: conversation.id)
        }
        Button("Rename…") { workspace.naming = .conversation(conversation.id) }
        Menu("Move to Project") {
            ForEach(library.projects) { project in
                Button(project.name) { library.move(conversation: conversation.id, to: project.id) }
                    .disabled(conversation.projectID == project.id)
            }
            if !library.projects.isEmpty { Divider() }
            Button("New Project…") { workspace.naming = .newProject(moving: conversation.id) }
            if conversation.projectID != nil {
                Button("Remove from Project") { library.move(conversation: conversation.id, to: nil) }
            }
        }
        Divider()
        Button("Delete Chat…", role: .destructive) { workspace.deleting = .conversation(conversation.id) }
            .disabled(generator.isRunning(conversation.id))
    }
}

/// A chat's newest result, or the symbol of its last request.
struct ChatThumbnail: View {
    @Environment(Library.self) private var library
    let conversation: Conversation
    let side: CGFloat

    var body: some View {
        Group {
            if let result = conversation.latestResult {
                StoredImage(url: library.imageURL(result.id), maxPixels: Int(side * 4))
                    .scaledToFill()
            } else {
                Image(systemName: conversation.entries.last?.workflow.symbol ?? "photo")
                    .font(.system(size: side * 0.5))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.quaternary.opacity(0.6))
            }
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: side * 0.22, style: .continuous))
        .accessibilityHidden(true)
    }
}

/// The engine's state and Settings, always at the bottom of the sidebar, aligned with the rows.
private struct SidebarFooter: View {
    var body: some View {
        HStack(spacing: 8) {
            SettingsLink {
                HStack(spacing: 8) {
                    Image(systemName: "gearshape").frame(width: 18)
                    Text("Settings")
                }
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Settings (⌘,)")
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
