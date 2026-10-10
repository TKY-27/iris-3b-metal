import SwiftUI

/// All Images, projects with their conversations, and unfiled conversations, with search.
struct Sidebar: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @Environment(Generator.self) private var generator
    @State private var query = ""
    @State private var collapsedProjects: Set<UUID> = []
    @State private var naming: Naming?
    @State private var name = ""
    @State private var deleting: Deletion?

    /// What the name alert renames or creates.
    private enum Naming: Equatable {
        case newProject(moving: UUID?)
        case project(UUID)
        case conversation(UUID)
    }

    private enum Deletion: Equatable {
        case project(UUID)
        case conversation(UUID)
    }

    var body: some View {
        @Bindable var workspace = workspace
        List(selection: $workspace.selection) {
            if query.isEmpty {
                Label("All Images", systemImage: "photo.on.rectangle.angled")
                    .tag(Workspace.Selection.allImages)
                projectsSection
                Section("Conversations") {
                    ForEach(library.conversations(in: nil)) { row($0) }
                }
            } else {
                Section("Results") {
                    ForEach(searchResults) { row($0) }
                }
            }
        }
        .searchable(text: $query, placement: .sidebar, prompt: "Search conversations")
        .onDeleteCommand {
            if let id = workspace.conversationID, !isRunning(id) { deleting = .conversation(id) }
        }
        .overlay {
            if !query.isEmpty, searchResults.isEmpty {
                ContentUnavailableView.search(text: query)
            }
        }
        .onChange(of: workspace.requestsNewProject) { _, requested in
            if requested {
                workspace.requestsNewProject = false
                beginNaming(.newProject(moving: nil), name: "")
            }
        }
        .alert(namingTitle, isPresented: isNaming) {
            TextField("Name", text: $name)
            Button(namingCreates ? "Create" : "Rename", action: commitName)
                .keyboardShortcut(.defaultAction)
            Button("Cancel", role: .cancel) { naming = nil }
        }
        .confirmationDialog(deletionTitle, isPresented: isDeleting, titleVisibility: .visible) {
            Button("Delete", role: .destructive, action: commitDeletion)
        } message: {
            Text(deletionMessage)
        }
    }

    // MARK: - Sections

    private var projectsSection: some View {
        Section {
            ForEach(library.projects) { project in
                DisclosureGroup(isExpanded: expansion(of: project.id)) {
                    let conversations = library.conversations(in: project.id)
                    ForEach(conversations) { row($0) }
                    if conversations.isEmpty {
                        Button {
                            workspace.newConversation(in: project.id)
                        } label: {
                            Label("New Conversation", systemImage: "square.and.pencil")
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                } label: {
                    Label(project.name, systemImage: "folder")
                        .contextMenu { projectMenu(project) }
                }
            }
            Button {
                beginNaming(.newProject(moving: nil), name: "")
            } label: {
                Label("New Project", systemImage: "folder.badge.plus")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        } header: {
            Text("Projects")
        }
    }

    private func row(_ conversation: Conversation) -> some View {
        ConversationRow(conversation: conversation)
            .tag(Workspace.Selection.conversation(conversation.id))
            .contextMenu { conversationMenu(conversation) }
    }

    private var searchResults: [Conversation] {
        library.conversations.filter { $0.matches(query) }.sorted { $0.updated > $1.updated }
    }

    private func expansion(of project: UUID) -> Binding<Bool> {
        Binding(
            get: { !collapsedProjects.contains(project) },
            set: { expanded in
                if expanded { collapsedProjects.remove(project) } else { collapsedProjects.insert(project) }
            }
        )
    }

    // MARK: - Menus

    @ViewBuilder private func projectMenu(_ project: Project) -> some View {
        Button("New Conversation in Project") { workspace.newConversation(in: project.id) }
        Button("Rename…") { beginNaming(.project(project.id), name: project.name) }
        Divider()
        Button("Delete Project…", role: .destructive) { deleting = .project(project.id) }
    }

    @ViewBuilder private func conversationMenu(_ conversation: Conversation) -> some View {
        Button("Rename…") { beginNaming(.conversation(conversation.id), name: conversation.title) }
        Menu("Move to Project") {
            ForEach(library.projects) { project in
                Button(project.name) { library.move(conversation: conversation.id, to: project.id) }
                    .disabled(conversation.projectID == project.id)
            }
            if !library.projects.isEmpty { Divider() }
            Button("New Project…") { beginNaming(.newProject(moving: conversation.id), name: "") }
            if conversation.projectID != nil {
                Button("Remove from Project") { library.move(conversation: conversation.id, to: nil) }
            }
        }
        Divider()
        Button("Delete Conversation…", role: .destructive) { deleting = .conversation(conversation.id) }
            .disabled(isRunning(conversation.id))
    }

    private func isRunning(_ conversation: UUID) -> Bool {
        generator.active?.conversation == conversation
    }

    // MARK: - Naming

    private var isNaming: Binding<Bool> {
        Binding(get: { naming != nil }, set: { if !$0 { naming = nil } })
    }

    private var namingCreates: Bool {
        if case .newProject = naming { true } else { false }
    }

    private var namingTitle: String {
        switch naming {
        case .newProject: "New Project"
        case .project: "Rename Project"
        case .conversation: "Rename Conversation"
        case nil: ""
        }
    }

    private func beginNaming(_ target: Naming, name: String) {
        self.name = name
        naming = target
    }

    private func commitName() {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        defer { naming = nil }
        guard !trimmed.isEmpty else { return }
        switch naming {
        case let .newProject(moving):
            guard let project = library.createProject(named: trimmed) else { return }
            if let moving { library.move(conversation: moving, to: project) }
        case let .project(id):
            library.rename(project: id, to: trimmed)
        case let .conversation(id):
            library.rename(conversation: id, to: trimmed)
        case nil:
            break
        }
    }

    // MARK: - Deletion

    private var isDeleting: Binding<Bool> {
        Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })
    }

    private var deletionTitle: String {
        switch deleting {
        case let .project(id): "Delete “\(library.projects.first { $0.id == id }?.name ?? "")”?"
        case let .conversation(id): "Delete “\(library.conversation(id)?.title ?? "")”?"
        case nil: ""
        }
    }

    private var deletionMessage: String {
        switch deleting {
        case .project: "Its conversations are kept and move to Conversations."
        case .conversation: "Its requests and images are deleted. This cannot be undone."
        case nil: ""
        }
    }

    private func commitDeletion() {
        switch deleting {
        case let .project(id): library.delete(project: id)
        case let .conversation(id): library.delete(conversation: id)
        case nil: break
        }
        deleting = nil
    }
}

private struct ConversationRow: View {
    let conversation: Conversation

    var body: some View {
        Text(conversation.title)
            .lineLimit(1)
            .truncationMode(.tail)
            .help(conversation.title)
    }
}
