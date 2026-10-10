import QuickLook
import SwiftUI

/// A conversation's requests and results in order, with the composer below.
/// `conversationID` nil is a new conversation that exists once its first request is sent.
struct ConversationView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let conversationID: UUID?
    @State private var previewURL: URL?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var conversation: Conversation? { library.conversation(conversationID) }

    var body: some View {
        VStack(spacing: 0) {
            if let conversation, !conversation.entries.isEmpty {
                feed(conversation)
            } else {
                EmptyConversation()
            }
            Composer()
        }
        .navigationTitle(conversation?.title ?? "New Conversation")
        .navigationSubtitle(projectName ?? "")
        .quickLookPreview($previewURL, in: previewURLs)
        .dropDestination(for: URL.self) { urls, _ in
            guard workspace.workflow.needsImage, let url = urls.first else { return false }
            workspace.attachment = .init(url: url, name: url.lastPathComponent)
            return true
        }
    }

    private var projectName: String? {
        let project = conversation?.projectID ?? (conversationID == nil ? workspace.draftProject : nil)
        return library.projects.first { $0.id == project }?.name
    }

    private var previewURLs: [URL] {
        (conversation?.entries ?? []).filter(\.isDone).map { library.imageURL($0.id) }
    }

    private func feed(_ conversation: Conversation) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 36) {
                    ForEach(conversation.entries) { entry in
                        EntryView(entry: entry, conversation: conversation.id, preview: { previewURL = $0 })
                            .id(entry.id)
                    }
                }
                .frame(maxWidth: 760)
                .padding(.horizontal, 24)
                .padding(.vertical, 28)
                .frame(maxWidth: .infinity)
            }
            .defaultScrollAnchor(.bottom)
            .onAppear { scrollToFocus(proxy) }
            .onChange(of: workspace.focusedEntry) { scrollToFocus(proxy) }
            // follow a new request, and its result when it replaces the progress card
            .onChange(of: conversation.entries.last?.id) { scrollToEnd(conversation, proxy) }
            .onChange(of: conversation.entries.last?.outcome) { scrollToEnd(conversation, proxy) }
        }
    }

    private func scrollToEnd(_ conversation: Conversation, _ proxy: ScrollViewProxy) {
        guard let last = conversation.entries.last else { return }
        withAnimation(reduceMotion ? nil : .default) { proxy.scrollTo(last.id, anchor: .bottom) }
    }

    private func scrollToFocus(_ proxy: ScrollViewProxy) {
        guard let entry = workspace.focusedEntry else { return }
        proxy.scrollTo(entry, anchor: .center)
        workspace.focusedEntry = nil
    }
}

/// What a new conversation shows above the composer.
private struct EmptyConversation: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        ContentUnavailableView {
            Label(workspace.workflow.title, systemImage: workspace.workflow.symbol)
        } description: {
            Text(hint)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var hint: String {
        switch workspace.workflow {
        case .create:
            "Describe an image below. Each request is generated on its own: Iris-3B does not see earlier images or prompts in the conversation."
        case .depth:
            "Choose or drop a photo to estimate its relative depth. Near is bright; values are relative, not metric."
        case .upscale:
            "Choose or drop an image to restore it at four times its size, repairing blur and compression."
        }
    }
}
