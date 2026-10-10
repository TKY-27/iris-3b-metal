import SwiftUI

struct ContentView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace

    var body: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 380)
        } detail: {
            switch workspace.selection {
            case .allImages:
                AllImagesView()
            case let .project(id):
                ProjectView(projectID: id)
                    .id(id)
            case let .conversation(id):
                ConversationView(conversationID: id)
                    .id(id)  // fresh scroll position and image state per conversation
            case .newChat, nil:
                ConversationView(conversationID: nil)
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    workspace.newConversationHere(library: library)
                } label: {
                    Label("New Chat", systemImage: "square.and.pencil")
                }
                .help("New Chat (⌘N)")
            }
        }
        // what is shown was deleted: start a new chat instead
        .onChange(of: library.conversations.map(\.id)) { _, ids in
            if let id = workspace.conversationID, !ids.contains(id) { workspace.newConversation() }
        }
        .onChange(of: library.projects.map(\.id)) { _, ids in
            if case let .project(id) = workspace.selection, !ids.contains(id) { workspace.newConversation() }
            if let draft = workspace.draftProject, !ids.contains(draft) { workspace.draftProject = nil }
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
}
