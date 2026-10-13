import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            Tab("General", systemImage: "gearshape") { GeneralSettings() }
            Tab("Engine", systemImage: "cpu") { EngineSettings() }
            Tab("Storage", systemImage: "internaldrive") { StorageSettings() }
        }
        .frame(width: 520)
    }
}

private struct GeneralSettings: View {
    @Environment(UpdateChecker.self) private var updates
    @AppStorage(AppAppearance.key) private var appearance = AppAppearance.system
    @AppStorage(FieldSettings.animatesKey) private var animates = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        @Bindable var updates = updates
        Form {
            Section {
                Picker("Appearance", selection: $appearance) {
                    ForEach(AppAppearance.allCases) { Text($0.title).tag($0) }
                }
                Toggle("Animate iris flowers", isOn: $animates)
                    .disabled(reduceMotion)
                    .help(reduceMotion ? "Reduce Motion is enabled in macOS Accessibility settings." : "Flowers respond to the pointer and animate while an image is being made.")
            }
            Section("Updates") {
                LabeledContent("App version", value: "\(updates.appVersion) (\(UpdateChecker.installedBuild))")
                Toggle("Check automatically", isOn: $updates.automatic)
                CheckForUpdatesButton()
            }

        }
        .formStyle(.grouped)
    }
}

private struct EngineSettings: View {
    @Environment(EngineController.self) private var engine
    @Environment(Generator.self) private var generator
    @State private var path = ""
    @State private var port = 8765

    var body: some View {
        Form {
            Section {
                LabeledContent("Status") { Text(status) }
                LabeledContent("Engine version", value: engine.health?.engineVersion ?? "Not reported")
                if let notice = engine.health?.compatibilityNotice {
                    Text(notice).foregroundStyle(.secondary)
                }
                DisclosureGroup("Engine update instructions") {
                    Text("Use iris-metal 0.3.x with this app. Quit Iris before changing the engine. Install the chosen release into a new Python environment, then select its bin/iris-metal here. Keep the old environment until the new one works; select the old path again to roll back.")
                    Text("For pipx or another package manager, use the tool that owns the installation. App updates do not update Python packages. Checkpoints and the Hugging Face cache can stay where they are.")
                    Link("Official installation guide", destination: EngineHealth.instructionsURL)
                }
                TextField("iris-metal location", text: $path, prompt: Text(engine.resolvedExecutable()?.path ?? "Not found"))
                Button("Choose…", action: choose)
                TextField("Port", value: $port, format: .number.grouping(.never))
                    .help("\(EngineController.portRange.lowerBound)–\(EngineController.portRange.upperBound)")
            } footer: {
                Text("The Python engine is installed separately. Iris never changes your Python environment. Leave the location empty to search ~/.local/bin and Homebrew.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("Apply and Restart Engine") {
                    engine.executablePath = path
                    engine.port = port
                    Task { await engine.restart() }
                }
                .disabled(generator.isBusy || engine.state == .starting)
                .help(generator.isBusy ? "Wait for the current request to finish" : "")
            }
        }
        .formStyle(.grouped)
        .onAppear {
            path = engine.executablePath
            port = engine.port
        }
    }

    private var status: String {
        switch engine.state {
        case .ready: "Running"
        case .starting: "Starting"
        case .stopped: "Stopped"
        case let .failed(message): message
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.showsHiddenFiles = true
        if panel.runModal() == .OK, let url = panel.url { path = url.path }
    }
}

private struct StorageSettings: View {
    @Environment(Library.self) private var library
    @State private var models: [ModelCache] = []
    @State private var libraryBytes: Int64?
    @State private var confirmsDelete = false

    var body: some View {
        Form {
            Section {
                if models.isEmpty {
                    Text("No model weights downloaded yet.").foregroundStyle(.secondary)
                }
                ForEach(models) { model in
                    LabeledContent(model.name) {
                        Text(model.bytes, format: .byteCount(style: .file))
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([model.url])
                        }
                    }
                }
            } header: {
                Text("Models")
            } footer: {
                // the Hugging Face cache can share files between repositories, so
                // removal is left to its own tool, which knows the layout
                Text("Models live in the Hugging Face cache. Remove one with `hf cache rm model/speridlabs/iris-3b`; the engine downloads it again when needed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Section("Library") {
                LabeledContent("^[\(library.imageCount) image](inflect: true)") {
                    if let libraryBytes {
                        Text(libraryBytes, format: .byteCount(style: .file))
                    }
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([library.store.root])
                    }
                }
                HStack {
                    Spacer()
                    Button("Delete All Images…", role: .destructive) { confirmsDelete = true }
                        .disabled(library.conversations.isEmpty || library.isReadOnly)
                }
            }
        }
        .formStyle(.grouped)
        .task { models = await ModelCache.scan() }
        .task(id: library.entryCount) { libraryBytes = await library.byteCount() }
        .confirmationDialog("Delete all images?", isPresented: $confirmsDelete) {
            Button("Delete All", role: .destructive) {
                library.deleteAllImages()
            }
        } message: {
            Text("Their prompts and settings are deleted too. Projects are kept, and so are images still being made. This cannot be undone.")
        }
    }
}

/// A model downloaded into the Hugging Face cache by the engine.
struct ModelCache: Identifiable, Sendable {
    let name: String
    let url: URL
    let bytes: Int64

    var id: URL { url }

    static let known = [
        ("Iris-3B", "models--speridlabs--iris-3b"),
        ("Qwen3-VL-4B text encoder", "models--Qwen--Qwen3-VL-4B-Instruct"),
    ]

    static func scan() async -> [ModelCache] {
        await Task.detached(priority: .utility) {
            let environment = ProcessInfo.processInfo.environment
            let hub = environment["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
                ?? URL(fileURLWithPath: environment["HF_HOME"] ?? NSString("~/.cache/huggingface").expandingTildeInPath)
                    .appending(path: "hub")
            return known.compactMap { name, folder in
                let url = hub.appending(path: folder)
                guard FileManager.default.fileExists(atPath: url.path) else { return nil }
                return ModelCache(name: name, url: url, bytes: allocatedSize(of: url))
            }
        }.value
    }

    /// Bytes of the distinct files reachable from the folder; the cache links
    /// snapshots to blobs, and blobs may live in a store shared by all models.
    private static func allocatedSize(of directory: URL) -> Int64 {
        let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        var seen = Set<URL>()
        var total: Int64 = 0
        while let file = files?.nextObject() as? URL {
            let target = file.resolvingSymlinksInPath()
            let values = try? target.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey])
            guard values?.isRegularFile == true, seen.insert(target).inserted else { continue }
            total += Int64(values?.totalFileAllocatedSize ?? 0)
        }
        return total
    }
}

/// Light, dark, or as the system is set.
enum AppAppearance: String, CaseIterable, Identifiable {
    case system, light, dark

    static let key = "appearance"

    var id: Self { self }

    var title: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    var nsAppearance: NSAppearance? {
        switch self {
        case .system: nil
        case .light: NSAppearance(named: .aqua)
        case .dark: NSAppearance(named: .darkAqua)
        }
    }
}
