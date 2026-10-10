import SwiftUI

struct ContentView: View {
    @Environment(EngineController.self) private var engine
    @Environment(History.self) private var history
    @Environment(Generator.self) private var generator
    @State private var selection: HistoryItem.ID?
    @State private var showsInspector = true

    private var selectedItem: HistoryItem? {
        history.items.first { $0.id == selection }
    }

    var body: some View {
        NavigationSplitView {
            HistorySidebar(selection: $selection)
                .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            VStack(spacing: 0) {
                Canvas(item: generator.isBusy ? nil : selectedItem)
                    .dropDestination(for: URL.self) { urls, _ in
                        guard generator.workflow != .create, !generator.isBusy, let url = urls.first else { return false }
                        generator.input = url
                        return true
                    }
                Divider()
                Composer(onRun: run)
            }
            .inspector(isPresented: $showsInspector) {
                SettingsInspector()
                    .inspectorColumnWidth(min: 240, ideal: 260)
            }
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                @Bindable var generator = generator
                Picker("Workflow", selection: $generator.workflow) {
                    ForEach(Workflow.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(generator.isBusy)
            }
            ToolbarItem(placement: .status) {
                EngineStatus()
            }
            ToolbarItem {
                Button {
                    showsInspector.toggle()
                } label: {
                    Label("Generation Settings", systemImage: "sidebar.right")
                }
                .help("Show or hide generation settings")
            }
        }
        .navigationTitle(selectedItem.map { String($0.title.prefix(60)) } ?? "Iris")
    }

    private func run() {
        Task {
            if let item = await generator.run(engine: engine, history: history) {
                selection = item.id
            }
        }
    }
}

/// The input and the primary action, always at the bottom of the window.
private struct Composer: View {
    @Environment(EngineController.self) private var engine
    @Environment(Generator.self) private var generator
    let onRun: () -> Void

    var body: some View {
        @Bindable var generator = generator
        HStack(alignment: .bottom, spacing: 12) {
            if generator.workflow == .create {
                TextField("Describe an image", text: $generator.prompt, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .onSubmit(onRun)
            } else {
                Button("Choose Image…", action: chooseImage)
                    .disabled(generator.isBusy)
                Text(generator.input?.lastPathComponent ?? "or drop one above")
                    .layoutPriority(1)
                    .foregroundStyle(generator.input == nil ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
            }
            if generator.isBusy {
                Button("Cancel", role: .cancel) {
                    generator.cancel(engine: engine)
                }
                .keyboardShortcut(".", modifiers: .command)
            } else {
                Button(actionTitle, action: onRun)
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .disabled(!generator.canRun || engine.state != .ready)
            }
        }
        .padding(14)
    }

    private var actionTitle: String {
        switch generator.workflow {
        case .create: "Generate"
        case .depth: "Estimate Depth"
        case .upscale: "Upscale"
        }
    }

    private func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK { generator.input = panel.url }
    }
}

private struct EngineStatus: View {
    @Environment(EngineController.self) private var engine

    var body: some View {
        switch engine.state {
        case .ready:
            Label("Engine ready", systemImage: "circle.fill")
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
                .font(.caption)
        case .starting:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Starting engine").font(.caption)
            }
        case .stopped, .failed:
            Label("Engine not running", systemImage: "exclamationmark.triangle.fill")
                .labelStyle(.titleAndIcon)
                .foregroundStyle(.orange)
                .font(.caption)
        }
    }
}
