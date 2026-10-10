import SwiftUI
import UniformTypeIdentifiers

/// A project's page: a composer that starts a chat in it, then its chats, newest first.
/// Chats dragged here from the sidebar are filed in it.
struct ProjectView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let projectID: UUID
    @State private var dropTargeted = false

    var body: some View {
        if let project = library.project(projectID) {
            let chats = library.conversations(in: project.id)
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    header(project, count: chats.count)
                    Composer()
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Chats")
                            .font(.headline)
                        if chats.isEmpty {
                            Text("No chats yet. Start one above, or drag chats here from the sidebar.")
                                .foregroundStyle(.secondary)
                                .padding(.vertical, 6)
                        } else {
                            VStack(spacing: 0) {
                                ForEach(chats) { chat in
                                    ProjectChatRow(conversation: chat)
                                    if chat.id != chats.last?.id { Divider().padding(.leading, 60) }
                                }
                            }
                            .padding(.horizontal, -8)  // the hover fill reaches past the column; the content aligns with it
                        }
                    }
                }
                .frame(maxWidth: Theme.readableWidth)
                .padding(.horizontal, 32)
                .padding(.vertical, 28)
                .frame(maxWidth: .infinity)
            }
            .navigationTitle(project.name)
            .navigationSubtitle(chats.count == 1 ? "1 chat" : "\(chats.count) chats")
            .toolbarTitleMenu {
                Button("New Chat in Project") { workspace.newConversation(in: project.id) }
                Button("Rename…") { workspace.naming = .project(project.id) }
                Divider()
                Button("Delete Project…", role: .destructive) { workspace.deleting = .project(project.id) }
            }
            // one handler for both kinds: stacked drop destinations can hide one another
            .onDrop(of: [.irisChat, .fileURL], isTargeted: $dropTargeted) { providers in
                accept(providers, into: project.id)
            }
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                        .strokeBorder(Color.accentColor, lineWidth: 2)
                        .padding(8)
                        .allowsHitTesting(false)
                }
            }
        } else {
            ContentUnavailableView("No Such Project", systemImage: "folder")
        }
    }

    /// Chats dragged from the sidebar are filed here; an image file goes to the composer.
    private func accept(_ providers: [NSItemProvider], into project: UUID) -> Bool {
        let chats = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.irisChat.identifier) }
        if !chats.isEmpty {
            for provider in chats {
                _ = provider.loadDataRepresentation(for: .irisChat) { data, _ in
                    guard let data, let chat = try? JSONDecoder().decode(ChatReference.self, from: data) else { return }
                    Task { @MainActor in library.move(conversation: chat.id, to: project) }
                }
            }
            return true
        }
        guard let file = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) else {
            return false
        }
        _ = file.loadObject(ofClass: URL.self) { url, _ in
            guard let url, (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType?.conforms(to: .image) == true
            else { return }
            Task { @MainActor in workspace.attach(url) }
        }
        return true
    }

    private func header(_ project: Project, count: Int) -> some View {
        Label {
            Text(project.name)
                .font(.largeTitle.weight(.semibold))
                .lineLimit(2)
        } icon: {
            Image(systemName: "folder").foregroundStyle(.secondary)
        }
        .font(.title)
        .accessibilityAddTraits(.isHeader)
    }
}

/// A chat on a project page: its latest result, title and when it was last used.
private struct ProjectChatRow: View {
    @Environment(Workspace.self) private var workspace
    let conversation: Conversation
    @State private var hovering = false

    var body: some View {
        Button {
            workspace.show(conversation.id)
        } label: {
            HStack(spacing: 12) {
                ChatThumbnail(conversation: conversation, side: 40)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(conversation.title).lineLimit(1)
                        if conversation.pinned != nil {
                            Image(systemName: "pin.fill")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                                .accessibilityLabel("Pinned")
                        }
                    }
                    Text(summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Text(conversation.updated, format: Date.RelativeFormatStyle(presentation: .named, capitalizationContext: .beginningOfSentence))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 8)
            .background(hovering ? AnyShapeStyle(.quaternary.opacity(0.6)) : AnyShapeStyle(.clear), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .draggable(ChatReference(id: conversation.id))
        .contextMenu { ChatMenu(conversation: conversation) }
    }

    private var summary: String {
        let images = conversation.entries.filter(\.isDone).count
        return images == 1 ? "1 image" : "\(images) images"
    }
}
