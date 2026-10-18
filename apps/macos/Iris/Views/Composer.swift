import SwiftUI

/// Prompt, attachment and settings for independent requests.
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
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) {
                        settingsRow
                        Spacer(minLength: 8)
                        generateOrStop
                    }
                    VStack(alignment: .leading, spacing: 10) {
                        settingsRow
                        HStack { Spacer(); generateOrStop }
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, 10)
            .background(.fieldBackground, in: RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous)
                    .strokeBorder(promptFocused ? Color.primary.opacity(0.28) : Color(nsColor: .separatorColor), lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: Theme.composerRadius, style: .continuous))
        }
        .onAppear { promptFocused = workspace.attachment == nil }
        .onChange(of: workspace.promptFocusRequests) { promptFocused = workspace.attachment == nil }
    }

    // MARK: - Parts

    @ViewBuilder private var notices: some View {
        EngineBanner()
        if library.isReadOnly {
            Notice(String(localized: "This library is read-only; see the message shown at launch."), systemImage: "lock")
        }
        if let elsewhere = busyElsewhere {
            BusyElsewhere(entry: elsewhere.entry, project: library.project(elsewhere.project))
        }
    }

    private var promptField: some View {
        @Bindable var workspace = workspace
        return HStack(alignment: .firstTextBaseline, spacing: 6) {
            TextField(placeholder, text: $workspace.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .lineSpacing(2)
                // Grow to eight lines, then scroll.
                .lineLimit(2...8)
                .focused($promptFocused)
                .onSubmit(submit)
                .accessibilityLabel(String(localized: "Prompt"))
            if !workspace.prompt.isEmpty {
                // Generation retains prompts; clear explicitly here.
                Button {
                    workspace.prompt = ""
                    promptFocused = true
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                }
                .buttonStyle(.borderless)
                .help(String(localized: "Clear Prompt"))
                .accessibilityLabel(String(localized: "Clear Prompt"))
            }
        }
        .padding(.horizontal, 2)
    }

    private var placeholder: String {
        library.project(workspace.project).map { String(localized: "Describe a new image for “\($0.name)”") } ?? String(localized: "Describe a new image")
    }

    private var settingsRow: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                inputControls
                if workspace.attachment == nil { tuningControls }
            }
            VStack(alignment: .leading, spacing: 10) {
                inputControls
                if workspace.attachment == nil { tuningControls }
            }
        }
        .buttonStyle(.borderless)
        .menuStyle(.button)
        .menuIndicator(.hidden)
        .labelStyle(.titleAndIcon)
        .foregroundStyle(.secondary)
    }

    private var inputControls: some View {
        HStack(spacing: 10) {
            Button(action: chooseImage) { Label(String(localized: "Add Image…"), systemImage: "photo.badge.plus") }
                .labelStyle(.iconOnly)
                .accessibilityLabel(String(localized: "Add Image"))
                .help(String(localized: "Restore and upscale an image, or estimate its depth"))
            if workspace.attachment == nil {
                modelMenu
                AspectMenu()
            }
        }
    }

    private var tuningControls: some View {
        HStack(spacing: 10) {
            Button { showsTuning.toggle() } label: {
                Label(tuningTitle, systemImage: "slider.horizontal.3")
            }
            .help(workspace.settings.model == .turbo ? String(localized: "Seed settings; Turbo always uses four steps") : String(localized: "Steps, guidance, seed and negative prompt"))
            .accessibilityLabel(String(localized: "Generation settings: \(tuningTitle)"))
            .popover(isPresented: $showsTuning, arrowEdge: .bottom) { GenerationControls() }
            CountMenu().labelStyle(.titleOnly)
        }
    }

    private var modelMenu: some View {
        @Bindable var workspace = workspace
        return Menu {
            Picker(String(localized: "Model"), selection: $workspace.settings.model) {
                ForEach(GenerationModel.allCases) { model in
                    Text(model == .turbo ? String(localized: "Turbo · 4 steps") : String(localized: "Original")).tag(model)
                        .disabled(engine.health?.supports(model) != true)
                }
            }
            .pickerStyle(.inline)
        } label: {
            Text(workspace.settings.model.title)
        }
        .fixedSize()
        .accessibilityLabel(String(localized: "Model, \(workspace.settings.model.title)"))
        .help(String(localized: "Original: adjustable steps, prompt adherence and negative prompt. Turbo: the distilled checkpoint generates images in four fixed steps. Each model downloads when first used."))
    }

    private var tuningTitle: String {
        String(localized: "\(workspace.settings.model == .turbo ? 4 : workspace.settings.steps) steps")
    }

    /// Stop replaces Generate while any request runs.
    @ViewBuilder private var generateOrStop: some View {
        if generator.isBusy {
            Button(String(localized: "Stop")) { generator.cancel(engine: engine) }
                .keyboardShortcut(".", modifiers: .command)
                .help(String(localized: "Stop the current request and the rest of its batch (⌘.)"))
        } else {
            Button(sendTitle, action: submit)
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!canSend)
                .help(sendHelp)
        }
    }

    // MARK: - State

    /// Running work outside the current gallery scope.
    private var busyElsewhere: (entry: UUID, project: UUID?)? {
        guard let active = generator.active, let found = library.entry(active.entry),
              workspace.selection?.contains(found.conversation) == false
        else { return nil }
        return (active.entry, found.conversation.projectID)
    }

    private var canSend: Bool {
        workspace.canSubmit && !generator.isBusy && engine.state == .ready && !library.isReadOnly
            && (workspace.workflow != .create || engine.health?.supports(workspace.settings.model) == true)
    }

    private var sendTitle: String {
        switch workspace.workflow {
        case .create: workspace.imageCount == 1 ? String(localized: "Generate") : String(localized: "Generate \(workspace.imageCount) Images")
        case .depth: String(localized: "Estimate Depth")
        case .upscale: String(localized: "Restore and Upscale")
        }
    }

    private var sendHelp: String {
        if engine.state != .ready { return String(localized: "The engine is not ready") }
        if workspace.workflow == .create, engine.health?.supports(workspace.settings.model) != true {
            return String(localized: "Update the engine in Settings to use Turbo.")
        }
        if !workspace.canSubmit {
            switch workspace.workflow {
            case .create: return String(localized: "Describe an image first")
            case .upscale where workspace.attachment != nil: return String(localized: "This image cannot be upscaled")
            default: return String(localized: "Add an image first")
            }
        }
        return workspace.workflow == .create ? String(localized: "\(sendTitle) (↩ or ⌘↩)") : String(localized: "\(sendTitle) (⌘↩)")
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
            StoredImage(url: attachment.url, maxPixels: 200, placeholder: "photo", version: attachment.id.uuidString)
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
                    .help(String(localized: "Remove Image"))
                    .accessibilityLabel(String(localized: "Remove Image"))
                }
                Picker(String(localized: "Action"), selection: $workspace.imageWorkflow) {
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
        if workspace.imageWorkflow == .depth { return String(localized: "Relative depth: near is bright, not in meters.") }
        guard let size = attachment.pixelSize else { return String(localized: "Repairs blur, noise and compression while enlarging.") }
        if let refusal = Upscaling.refusal(for: size) { return refusal }
        return String(localized: "Repairs blur, noise and compression while enlarging: \(size.pixelLabel) → \(Upscaling.output(for: size).pixelLabel).")
    }
}

/// The size of the next image, by shape.
private struct AspectMenu: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        let size = workspace.settings.size
        Menu {
            item(.square, name: String(localized: "Square"))
            Section(String(localized: "Landscape")) { ForEach(ImageSize.landscape) { item($0) } }
            Section(String(localized: "Portrait")) { ForEach(ImageSize.portrait) { item($0) } }
        } label: {
            Label(size.ratioLabel, systemImage: size.aspectRatio > 1.05 ? "rectangle" : size.aspectRatio < 0.95 ? "rectangle.portrait" : "square")
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)  // drawn by SwiftUI, so it takes the row's secondary color like the buttons
        .fixedSize()
        .help(String(localized: "Shape: \(size.label) pixels"))
        .accessibilityLabel(String(localized: "Shape, \(size.ratioLabel), \(size.label)"))
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
            Picker(String(localized: "Images"), selection: $workspace.imageCount) {
                ForEach(Array(Workspace.imageCountRange), id: \.self) { count in
                    Text(String(localized: "\(count) images")).tag(count)
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
        .help(String(localized: "How many images to make from the prompt, one after another, each with its own seed"))
        .accessibilityLabel(String(localized: "Number of images, \(workspace.imageCount)"))
    }

    private var title: String {
        String(localized: "\(workspace.imageCount) images")
    }
}

/// Reveal running work outside this gallery scope.
private struct BusyElsewhere: View {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    let entry: UUID
    let project: Project?

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.mini)
            Text(project.map { String(localized: "Generating in “\($0.name)”") } ?? String(localized: "Generating outside this project"))
                .lineLimit(1)
            Button(String(localized: "Show")) { workspace.reveal(entry, library: library) }
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

extension View {
    /// Attach dropped images for processing.
    func acceptsImageDrops() -> some View { modifier(ImageDrop()) }
}

private struct ImageDrop: ViewModifier {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @State private var targeted = false

    func body(content: Content) -> some View {
        content
            .dropDestination(for: URL.self) { urls, _ in
                // Gallery drags move entries instead of attaching.
                guard let url = urls.first, library.entryID(forImage: url) == nil,
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
                            Text(String(localized: "Drop to restore and upscale, or to estimate depth"))
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
