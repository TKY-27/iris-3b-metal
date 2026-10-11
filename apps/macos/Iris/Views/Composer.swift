import SwiftUI

/// The next request: a prompt, or an attached image and what to do with it, plus Send or Stop.
struct Composer: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    @FocusState private var promptFocused: Bool
    @State private var showsTuning = false

    var body: some View {
        VStack(spacing: 8) {
            notices
            VStack(alignment: .leading, spacing: 10) {
                if let attachment = workspace.attachment {
                    AttachmentRow(attachment: attachment)
                } else {
                    promptField
                }
                HStack(spacing: 16) {
                    Button(action: chooseImage) {
                        Label("Add Image…", systemImage: "photo.badge.plus")
                    }
                    .help("Restore and upscale an image, or estimate its depth (or drop one here)")
                    if workspace.attachment == nil {
                        AspectMenu()
                        Button {
                            showsTuning.toggle()
                        } label: {
                            Label(tuningTitle, systemImage: "slider.horizontal.3")
                        }
                        .help("Quality, prompt adherence, seed and negative prompt")
                        .accessibilityLabel("Generation settings: \(tuningTitle)")
                        .popover(isPresented: $showsTuning, arrowEdge: .top) { GenerationControls() }
                        CountMenu()
                    }
                    if workspace.selection == .newChat {
                        ProjectMenu()
                    }
                    Spacer(minLength: 0)
                    sendOrStop
                }
                .buttonStyle(.borderless)
                .menuStyle(.button)
                .menuIndicator(.hidden)
                .labelStyle(.titleAndIcon)
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 14)
            .padding(.top, 14)
            .padding(.bottom, 10)
            .background(.fieldBackground, in: RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous)
                    .strokeBorder(promptFocused ? Color.primary.opacity(0.28) : Color(nsColor: .separatorColor), lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous))
            .onTapGesture { promptFocused = workspace.attachment == nil }
        }
        .frame(maxWidth: Theme.readableWidth)
        .onAppear { promptFocused = workspace.attachment == nil }
    }

    // MARK: - Parts

    @ViewBuilder private var notices: some View {
        EngineBanner()
        if library.isReadOnly {
            Notice("This library is read-only; see the message shown at launch.", systemImage: "lock")
        }
        if let elsewhere = busyElsewhere {
            BusyElsewhere(conversation: elsewhere)
        }
    }

    private var promptField: some View {
        @Bindable var workspace = workspace
        return TextField(placeholder, text: $workspace.prompt, axis: .vertical)
            .textFieldStyle(.plain)
            .font(.system(size: 15))
            .lineSpacing(2)
            // two lines to start, then it grows with the text and scrolls past twelve
            .lineLimit(2...12)
            .focused($promptFocused)
            .onSubmit(submit)
            .padding(.horizontal, 2)
            .accessibilityLabel("Prompt")
    }

    /// Each request starts fresh, so a chat's composer asks for a new image, not a follow-up.
    private var placeholder: String {
        switch workspace.selection {
        case let .project(id): "Start a chat in \(library.project(id)?.name ?? "this project")"
        case .conversation: "Describe a new image"
        default: "Describe an image"
        }
    }

    private var tuningTitle: String {
        "\(workspace.settings.qualityTitle), \(workspace.settings.adherenceTitle)"
    }

    @ViewBuilder private var sendOrStop: some View {
        if isBusyHere {
            RoundAction(systemImage: "stop.fill", label: "Stop", enabled: true) {
                generator.cancel(engine: engine)
            }
            .keyboardShortcut(".", modifiers: .command)
            .help("Stop (⌘.)")
        } else {
            RoundAction(systemImage: "arrow.up", label: sendTitle, enabled: canSend, action: submit)
                .keyboardShortcut(.return, modifiers: .command)
                .help(sendHelp)
        }
    }

    // MARK: - State

    private var isBusyHere: Bool {
        guard let active = generator.active else { return false }
        return active.conversation == workspace.conversationID
    }

    private var busyElsewhere: UUID? {
        guard let active = generator.active, active.conversation != workspace.conversationID else { return nil }
        return active.conversation
    }

    private var canSend: Bool {
        workspace.canSubmit && !generator.isBusy && engine.state == .ready && !library.isReadOnly
    }

    private var sendTitle: String {
        switch workspace.workflow {
        case .create: workspace.imageCount == 1 ? "Generate" : "Generate \(workspace.imageCount) Images"
        case .depth: "Estimate Depth"
        case .upscale: "Restore and Upscale"
        }
    }

    private var sendHelp: String {
        if engine.state != .ready { return "The engine is not ready" }
        if generator.isBusy { return "Wait for the current request to finish" }
        if !workspace.canSubmit {
            switch workspace.workflow {
            case .create: return "Describe an image first"
            case .upscale where workspace.attachment != nil: return "This image cannot be upscaled"
            default: return "Add an image first"
            }
        }
        return workspace.workflow == .create ? "\(sendTitle) (↩ or ⌘↩)" : "\(sendTitle) (⌘↩)"
    }

    private func submit() {
        workspace.submit(generator: generator, library: library, engine: engine)
    }

    private func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url {
            workspace.attach(url)
        }
    }
}

/// The attached image and the choice of what to do with it.
private struct AttachmentRow: View {
    @Environment(Workspace.self) private var workspace
    let attachment: Workspace.Attachment

    var body: some View {
        @Bindable var workspace = workspace
        HStack(alignment: .center, spacing: 12) {
            StoredImage(url: attachment.url, maxPixels: 200, placeholder: "photo")
                .frame(width: 52, height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(.separator))
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(attachment.name)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button {
                        workspace.attachment = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                    }
                    .buttonStyle(.borderless)
                    .help("Remove Image")
                    .accessibilityLabel("Remove Image")
                }
                Picker("Action", selection: $workspace.imageWorkflow) {
                    ForEach(Workflow.imageWorkflows) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            Spacer(minLength: 8)
            Text(explanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .accessibilityLabel(explanation.spokenSizes)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 230, alignment: .trailing)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// What the chosen action will make of this image, with the sizes when known.
    private var explanation: String {
        if workspace.imageWorkflow == .depth { return "Relative depth: near is bright, not in meters." }
        guard let size = attachment.pixelSize else { return "Repairs blur, noise and compression while enlarging." }
        if let refusal = Upscaling.refusal(for: size) { return refusal }
        return "Repairs blur, noise and compression while enlarging: \(size.pixelLabel) → \(Upscaling.output(for: size).pixelLabel)."
    }
}

/// The size of the next image, by shape.
private struct AspectMenu: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        let size = workspace.settings.size
        Menu {
            item(.square, name: "Square")
            Section("Landscape") { ForEach(ImageSize.landscape) { item($0) } }
            Section("Portrait") { ForEach(ImageSize.portrait) { item($0) } }
        } label: {
            Label(size.ratioLabel, systemImage: size.aspectRatio > 1.05 ? "rectangle" : size.aspectRatio < 0.95 ? "rectangle.portrait" : "square")
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)  // drawn by SwiftUI, so it takes the row's secondary color like the buttons
        .fixedSize()
        .help("Shape: \(size.label) pixels")
        .accessibilityLabel("Shape, \(size.ratioLabel), \(size.label)")
    }

    private func item(_ size: ImageSize, name: String? = nil) -> some View {
        Toggle(isOn: Binding(get: { workspace.settings.size == size }, set: { if $0 { workspace.settings.size = size } })) {
            Text("\(name ?? size.ratioLabel)   \(size.label)")
        }
    }
}

/// How many images the next prompt makes, one after another.
private struct CountMenu: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        @Bindable var workspace = workspace
        Menu {
            Picker("Images", selection: $workspace.imageCount) {
                ForEach(Array(Workspace.imageCountRange), id: \.self) { count in
                    Text(count == 1 ? "1 Image" : "\(count) Images").tag(count)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        } label: {
            Label(title, systemImage: "square.stack")
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .fixedSize()
        .help("How many images to make from the prompt, one after another, each with its own seed")
        .accessibilityLabel("Number of images, \(workspace.imageCount)")
    }

    private var title: String {
        workspace.imageCount == 1 ? "1 image" : "\(workspace.imageCount) images"
    }
}

/// The project a new chat is saved in, with a way to make one on the spot.
private struct ProjectMenu: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace

    var body: some View {
        Menu {
            Picker("Project", selection: Binding(get: { workspace.draftProject }, set: { workspace.draftProject = $0 })) {
                Text("No Project").tag(UUID?.none)
                ForEach(library.projects) { Text($0.name).tag(UUID?.some($0.id)) }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            Button("New Project…") { workspace.naming = .newProjectForDraft }
        } label: {
            Label(library.project(workspace.draftProject)?.name ?? "No Project", systemImage: "folder")
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .fixedSize()
        .help("The project this chat is saved in")
        .accessibilityLabel("Project, \(library.project(workspace.draftProject)?.name ?? "none")")
    }
}

/// Points to the conversation whose request is running, since only one runs at a time.
private struct BusyElsewhere: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let conversation: UUID

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.mini)
            Text("Working in “\(library.conversation(conversation)?.title ?? "another chat")”")
                .lineLimit(1)
            Button("Show") { workspace.show(conversation) }
                .buttonStyle(.link)
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 4)
    }
}

struct Notice: View {
    let text: String
    let systemImage: String

    init(_ text: String, systemImage: String) {
        self.text = text
        self.systemImage = systemImage
    }

    var body: some View {
        Label(text, systemImage: systemImage)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 4)
    }
}

/// The round send (or stop) button at the composer's trailing edge: ink on paper, no color.
private struct RoundAction: View {
    let systemImage: String
    let label: String
    let enabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(enabled ? Color.fieldBackground : Color.secondary)
                .frame(width: 26, height: 26)
                .background(Circle().fill(enabled ? AnyShapeStyle(.primary) : AnyShapeStyle(.quaternary)))
                .contentShape(Circle())
        }
        .buttonStyle(PressScale())
        .disabled(!enabled)
        .accessibilityLabel(label)
    }
}

/// Feedback on press, not on release.
struct PressScale: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? 0.75 : 1)
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }
}
