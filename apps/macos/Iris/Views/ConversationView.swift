import QuickLook
import SwiftUI
import UniformTypeIdentifiers

/// A conversation's requests and results in order, with the composer below.
/// `conversationID` nil is a new chat that exists once its first request is sent.
struct ConversationView: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let conversationID: UUID?
    @State private var previewURL: URL?
    @State private var scrollerWidth = ConversationView.legacyScrollerWidth()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var conversation: Conversation? { library.conversation(conversationID) }

    var body: some View {
        Group {
            if let conversation, !conversation.entries.isEmpty {
                VStack(spacing: 0) {
                    feed(conversation)
                    Composer()
                        .padding(.horizontal, 24)
                        .padding(.top, 4)
                        .padding(.bottom, 16)
                        .padding(.trailing, scrollerWidth)  // centered on the feed, not under its scroller
                }
            } else {
                WelcomeView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle(conversation?.title ?? "New Chat")
        .navigationSubtitle(projectName ?? "")
        .quickLookPreview($previewURL, in: previewURLs)
        .acceptsImageDrops()
        .toolbarTitleMenu {
            if let conversation { ChatMenu(conversation: conversation) }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSScroller.preferredScrollerStyleDidChangeNotification)) { _ in
            scrollerWidth = Self.legacyScrollerWidth()
        }
    }

    /// Always-shown scroll bars take width from the feed; overlay ones take none.
    private static func legacyScrollerWidth() -> CGFloat {
        NSScroller.preferredScrollerStyle == .legacy ? NSScroller.scrollerWidth(for: .regular, scrollerStyle: .legacy) : 0
    }

    private var projectName: String? {
        library.project(conversation?.projectID ?? (conversationID == nil ? workspace.draftProject : nil))?.name
    }

    private var previewURLs: [URL] {
        (conversation?.entries ?? []).filter(\.isDone).map { library.imageURL($0.id) }
    }

    private func feed(_ conversation: Conversation) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 36) {
                    ForEach(conversation.entries) { entry in
                        EntryView(
                            entry: entry, conversation: conversation.id, isLatest: entry.id == conversation.entries.last?.id,
                            preview: { previewURL = $0 }
                        )
                            .id(entry.id)
                    }
                }
                .frame(maxWidth: Theme.readableWidth)
                .padding(.horizontal, 24)
                .padding(.vertical, 24)
                .frame(maxWidth: .infinity)
            }
            .defaultScrollAnchor(.bottom)
            .hardScrollEdge(.top)
            .onAppear {
                if workspace.focusedEntry == nil { scrollToEnd(conversation, proxy, animated: false) }
                scrollToFocus(proxy)
            }
            .onChange(of: workspace.focusedEntry) { scrollToFocus(proxy) }
            // follow a new request, and its result when it replaces the canvas
            .onChange(of: conversation.entries.last?.id) { scrollToEnd(conversation, proxy) }
            .onChange(of: conversation.entries.last?.outcome) { scrollToEnd(conversation, proxy) }
        }
    }

    /// Shows the latest entry from its request down, so a tall result never hides its prompt.
    private func scrollToEnd(_ conversation: Conversation, _ proxy: ScrollViewProxy, animated: Bool = true) {
        guard let last = conversation.entries.last else { return }
        withAnimation(animated && !reduceMotion ? Theme.spring : nil) { proxy.scrollTo(last.id, anchor: .top) }
    }

    private func scrollToFocus(_ proxy: ScrollViewProxy) {
        guard let entry = workspace.focusedEntry else { return }
        proxy.scrollTo(entry, anchor: .center)
        workspace.focusedEntry = nil
    }
}

/// A new chat: one question and the composer, over the faint field.
private struct WelcomeView: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        IrisCanvas(energy: 0, center: UnitPoint(x: 0.5, y: 0.45)) {
            // centered with the footnote at the bottom; scrolls only in a window too short for it
            GeometryReader { geometry in
                ScrollView {
                    content.frame(minHeight: geometry.size.height)
                }
                .scrollBounceBehavior(.basedOnSize)
            }
        }
    }

    private var content: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 32)
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 12) {
                    Image(nsImage: NSApp.applicationIconImage)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: 40, height: 40)
                        .accessibilityHidden(true)
                    Text(headline)
                        .font(.largeTitle.weight(.semibold))
                        .accessibilityAddTraits(.isHeader)
                }
                .frame(maxWidth: .infinity)
                Composer()
            }
            .frame(maxWidth: Theme.readableWidth)
            .padding(.horizontal, 32)
            Spacer(minLength: 32)
            Text("Each request starts fresh: Iris-3B does not see earlier prompts or images. Drop a photo to upscale it or estimate its depth.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)
                .padding(.bottom, 16)
        }
        .frame(maxWidth: .infinity)
    }

    private var headline: String {
        switch workspace.workflow {
        case .create: "What would you like to create?"
        case .depth, .upscale: "What should Iris do with this image?"
        }
    }
}

extension View {
    /// Dropping an image file attaches it to the composer, for upscaling or depth.
    func acceptsImageDrops() -> some View { modifier(ImageDrop()) }
}

private struct ImageDrop: ViewModifier {
    @Environment(Workspace.self) private var workspace
    @State private var targeted = false

    func body(content: Content) -> some View {
        content
            .dropDestination(for: URL.self) { urls, _ in
                // only images; anything else would fail later, at upload
                guard let url = urls.first,
                      (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType?.conforms(to: .image) == true
                else { return false }
                workspace.attach(url)
                return true
            } isTargeted: { targeted = $0 }
            .overlay {
                if targeted {
                    RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                        .strokeBorder(Color.accentColor, lineWidth: 2)
                        .overlay {
                            Text("Drop to upscale or estimate depth")
                                .font(.headline)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .background(.fieldBackground, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(.separator))
                        }
                        .padding(8)
                        .allowsHitTesting(false)
                }
            }
    }
}
