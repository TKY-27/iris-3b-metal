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
            case let .conversation(id):
                ConversationView(conversationID: id)
                    .id(id)  // fresh scroll position and image state per conversation
            case nil:
                ConversationView(conversationID: nil)
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    workspace.newConversation()
                } label: {
                    Label("New Conversation", systemImage: "square.and.pencil")
                }
                .help("New Conversation (⌘N)")
            }
            ToolbarItem(placement: .status) {
                EngineStatus()
            }
        }
        .onChange(of: library.conversations.map(\.id)) { _, ids in
            if let id = workspace.conversationID, !ids.contains(id) {
                workspace.newConversation()  // the shown conversation was deleted
            }
        }
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
