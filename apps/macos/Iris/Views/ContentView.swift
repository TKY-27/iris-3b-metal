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
                Divider()
                Composer(onGenerate: generate)
            }
            .inspector(isPresented: $showsInspector) {
                SettingsInspector()
                    .inspectorColumnWidth(min: 240, ideal: 260)
            }
        }
        .toolbar {
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
        .navigationTitle(selectedItem.map { String($0.request.prompt.prefix(60)) } ?? "Iris")
    }

    private func generate() {
        Task {
            if let item = await generator.generate(engine: engine, history: history) {
                selection = item.id
            }
        }
    }
}

/// The prompt field and the primary action, always at the bottom of the window.
private struct Composer: View {
    @Environment(EngineController.self) private var engine
    @Environment(Generator.self) private var generator
    let onGenerate: () -> Void

    var body: some View {
        @Bindable var generator = generator
        HStack(alignment: .bottom, spacing: 12) {
            TextField("Describe an image", text: $generator.prompt, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .font(.body)
                .onSubmit(onGenerate)
            if generator.isBusy {
                Button("Cancel", role: .cancel) {
                    generator.cancel(engine: engine)
                }
                .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Generate", action: onGenerate)
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .disabled(!generator.canGenerate || engine.state != .ready)
            }
        }
        .padding(14)
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
