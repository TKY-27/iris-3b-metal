import SwiftUI

/// The prompt or input image, the workflow and its settings, and Send/Stop; always at the bottom.
struct Composer: View {
    @Environment(Library.self) private var library
    @Environment(Generator.self) private var generator
    @Environment(EngineController.self) private var engine
    @Environment(Workspace.self) private var workspace
    @FocusState private var promptFocused: Bool
    @State private var showsSettings = false

    var body: some View {
        @Bindable var workspace = workspace
        VStack(spacing: 8) {
            EngineBanner()
            if library.isReadOnly {
                Label("This library is read-only; see the message shown at launch.", systemImage: "lock")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if let elsewhere = busyElsewhere {
                BusyElsewhere(conversation: elsewhere)
            }
            VStack(alignment: .leading, spacing: 10) {
                input
                HStack(spacing: 10) {
                    Picker("Workflow", selection: $workspace.workflow) {
                        ForEach(Workflow.allCases) { workflow in
                            Label(workflow.title, systemImage: workflow.symbol).tag(workflow)
                        }
                    }
                    .labelsHidden()
                    .fixedSize()
                    .help("What to do with the next request")
                    if workspace.workflow == .create {
                        Button {
                            showsSettings.toggle()
                        } label: {
                            Label(workspace.settings.summary, systemImage: "slider.horizontal.3")
                        }
                        .help("Size, steps, guidance, seed and negative prompt")
                        .popover(isPresented: $showsSettings, arrowEdge: .top) {
                            GenerationOptions()
                        }
                    }
                    Spacer(minLength: 0)
                    sendOrStop
                }
                .controlSize(.small)
            }
            .padding(12)
            .background(.background, in: RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.separator))
        }
        .frame(maxWidth: 760)
        .padding(.horizontal, 24)
        .padding(.bottom, 16)
        .padding(.top, 8)
        .frame(maxWidth: .infinity)
        .onAppear { promptFocused = true }
    }

    @ViewBuilder private var input: some View {
        @Bindable var workspace = workspace
        if workspace.workflow == .create {
            TextField("Describe a new image", text: $workspace.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...8)
                .focused($promptFocused)
                .onSubmit(submit)
                .accessibilityLabel("Prompt")
        } else if let attachment = workspace.attachment {
            HStack(spacing: 10) {
                StoredImage(url: attachment.url, maxPixels: 160, placeholder: "photo")
                    .frame(width: 44, height: 44)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                Text(attachment.name)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Button {
                    workspace.attachment = nil
                } label: {
                    Label("Remove Image", systemImage: "xmark.circle.fill")
                        .labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Remove Image")
            }
        } else {
            HStack(spacing: 8) {
                Button("Choose Image…", action: chooseImage)
                Text("or drop one onto the window")
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var sendOrStop: some View {
        if isBusyHere {
            Button {
                generator.cancel(engine: engine)
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }
            .keyboardShortcut(".", modifiers: .command)
            .help("Stop (⌘.)")
        } else {
            Button(action: submit) {
                Label(sendTitle, systemImage: "arrow.up")
            }
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(!canSend)
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
        case .create: "Generate"
        case .depth: "Estimate"
        case .upscale: "Upscale"
        }
    }

    private var sendHelp: String {
        if engine.state != .ready { return "The engine is not ready" }
        if generator.isBusy { return "Wait for the current request to finish" }
        return "\(sendTitle) (⌘↩)"
    }

    private func submit() {
        workspace.submit(generator: generator, library: library, engine: engine)
    }

    private func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url {
            workspace.attachment = .init(url: url, name: url.lastPathComponent)
        }
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
            Text("Working in “\(library.conversation(conversation)?.title ?? "another conversation")”")
                .lineLimit(1)
            Button("Show") { workspace.show(conversation) }
                .buttonStyle(.link)
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Text-to-image settings for the next request.
struct GenerationOptions: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        @Bindable var workspace = workspace
        Form {
            Picker("Size", selection: $workspace.settings.size) {
                Text(ImageSize.square.label).tag(ImageSize.square)
                Section("Landscape") {
                    ForEach(ImageSize.landscape) { Text($0.label).tag($0) }
                }
                Section("Portrait") {
                    ForEach(ImageSize.portrait) { Text($0.label).tag($0) }
                }
            }
            LabeledContent("Steps") {
                HStack(spacing: 4) {
                    TextField("Steps", value: $workspace.settings.steps, format: .number)
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 56)
                    Stepper("Steps", value: $workspace.settings.steps, in: GenerationSettings.stepRange, step: 10)
                        .labelsHidden()
                }
            }
            LabeledContent("Guidance") {
                HStack(spacing: 4) {
                    TextField("Guidance", value: $workspace.settings.cfgScale, format: .number.precision(.fractionLength(1)))
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 56)
                    Stepper("Guidance", value: $workspace.settings.cfgScale, in: GenerationSettings.guidanceRange, step: 0.5)
                        .labelsHidden()
                }
            }
            Toggle("New seed for every image", isOn: $workspace.settings.randomizeSeed)
            TextField("Seed", value: $workspace.settings.seed, format: .number.grouping(.never))
                .disabled(workspace.settings.randomizeSeed)
            TextField("Avoid", text: $workspace.settings.negativePrompt, prompt: Text("Negative prompt"), axis: .vertical)
                .lineLimit(1...3)
            if !workspace.settings.isDefaultQuality {
                Text("The model was evaluated at 100 steps and guidance 3; fewer steps are faster and change fine detail such as hands.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Restore Defaults") { workspace.settings = GenerationSettings() }
                    .disabled(workspace.settings == GenerationSettings())
            }
        }
        .formStyle(.grouped)
        .frame(width: 340)
    }
}
