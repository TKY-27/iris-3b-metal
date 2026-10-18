import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            Tab(String(localized: "General"), systemImage: "gearshape") { GeneralSettings() }
            Tab(String(localized: "Engine"), systemImage: "cpu") { EngineSettings() }
            Tab(String(localized: "Storage"), systemImage: "internaldrive") { StorageSettings() }
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
                Picker(String(localized: "Appearance"), selection: $appearance) {
                    ForEach(AppAppearance.allCases) { Text($0.title).tag($0) }
                }
                Toggle(String(localized: "Animate iris flowers"), isOn: $animates)
                    .disabled(reduceMotion)
                    .help(reduceMotion ? String(localized: "Reduce Motion is enabled in macOS Accessibility settings.") : String(localized: "Flowers respond to the pointer and animate while an image is being made."))
            }
            Section("Updates") {
                LabeledContent(String(localized: "App version"), value: "\(updates.appVersion) (\(UpdateChecker.installedBuild))")
                Toggle(String(localized: "Check automatically"), isOn: $updates.automatic)
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
                LabeledContent(String(localized: "Status")) { Text(status) }
                LabeledContent(String(localized: "Engine version"), value: engine.health?.engineVersion ?? String(localized: "Not reported"))
                if let notice = engine.health?.compatibilityNotice {
                    Text(notice).foregroundStyle(.secondary)
                }
                DisclosureGroup(String(localized: "Engine update instructions")) {
                    Text(String(localized: "Use iris-metal 0.3.x, 0.5.x or 0.6.x with this app. Quit Iris before changing the engine. Install the chosen release into a new Python environment, then select its bin/iris-metal here. Keep the old environment until the new one works; select the old path again to roll back."))
                    Text(String(localized: "For pipx or another package manager, use the tool that owns the installation. App updates do not update Python packages. Checkpoints and the Hugging Face cache can stay where they are."))
                    Link(String(localized: "Official installation guide"), destination: EngineHealth.instructionsURL)
                }
                TextField(String(localized: "iris-metal location"), text: $path, prompt: Text(engine.resolvedExecutable()?.path ?? String(localized: "Not found")))
                Button("Choose…", action: choose)
                TextField("Port", value: $port, format: .number.grouping(.never))
                    .help("\(EngineController.portRange.lowerBound)–\(EngineController.portRange.upperBound)")
            } footer: {
                Text(String(localized: "The Python engine is installed separately. Iris never changes your Python environment. Leave the location empty to search ~/.local/bin and Homebrew."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button(String(localized: "Apply and Restart Engine")) {
                    engine.executablePath = path
                    engine.port = port
                    Task { await engine.restart() }
                }
                .disabled(generator.isBusy || engine.state == .starting)
                .help(generator.isBusy ? String(localized: "Wait for the current request to finish") : "")
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
        case .ready: String(localized: "Running")
        case .starting: String(localized: "Starting")
        case .stopped: String(localized: "Stopped")
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
                    Text(String(localized: "No model weights downloaded yet.")).foregroundStyle(.secondary)
                }
                ForEach(models) { model in
                    LabeledContent(model.name) {
                        Text(model.bytes, format: .byteCount(style: .file))
                        Button(String(localized: "Show in Finder")) {
                            NSWorkspace.shared.activateFileViewerSelecting([model.url])
                        }
                    }
                }
            } header: {
                Text(String(localized: "Models"))
            } footer: {
                // the Hugging Face cache can share files between repositories, so
                // removal is left to its own tool, which knows the layout
                Text(String(localized: "Models live in the Hugging Face cache. Remove one with `hf cache rm model/speridlabs/iris-3b`; the engine downloads it again when needed."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Section(String(localized: "Library")) {
                LabeledContent(String(localized: "\(library.imageCount) images")) {
                    if let libraryBytes {
                        Text(libraryBytes, format: .byteCount(style: .file))
                    }
                    Button(String(localized: "Show in Finder")) {
                        NSWorkspace.shared.activateFileViewerSelecting([library.store.root])
                    }
                }
                HStack {
                    Spacer()
                    Button(String(localized: "Delete All Images…"), role: .destructive) { confirmsDelete = true }
                        .disabled(library.conversations.isEmpty || library.isReadOnly)
                }
            }
        }
        .formStyle(.grouped)
        .task { models = await ModelCache.scan() }
        .task(id: library.fileRevision) {
            let bytes = await library.byteCount()
            // a newer scan has taken over
            guard !Task.isCancelled else { return }
            libraryBytes = bytes
        }
        .confirmationDialog(String(localized: "Delete all images?"), isPresented: $confirmsDelete) {
            Button(String(localized: "Delete All"), role: .destructive) {
                library.deleteAllImages()
            }
        } message: {
            Text(String(localized: "Their prompts and settings are deleted too. Projects are kept, and so are images still being made. This cannot be undone."))
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
        (String(localized: "Qwen3-VL-4B text encoder"), "models--Qwen--Qwen3-VL-4B-Instruct"),
    ]

    static func scan() async -> [ModelCache] {
        await Task.detached(priority: .utility) {
            let hub = hubCache(environment: ProcessInfo.processInfo.environment)
            return known.compactMap { name, folder in
                let url = hub.appending(path: folder)
                guard FileManager.default.fileExists(atPath: url.path) else { return nil }
                return ModelCache(name: name, url: url, bytes: allocatedSize(of: url))
            }
        }.value
    }

    /// `HF_HUB_CACHE` as huggingface_hub 2.2 resolves it.
    static func hubCache(environment: [String: String]) -> URL {
        let home = environment["HOME"] ?? NSHomeDirectory()
        func expand(_ path: String) -> String {
            // Python's expanduser + expandvars: unknown variables stay
            // shortcut: no ~user form, add if reported
            var path = path
            if path == "~" || path.hasPrefix("~/") { path = home + path.dropFirst() }
            return path.replacing(/\$(\w+|\{[^}]*\})/) { match in
                var name = match.output.1
                if name.hasPrefix("{") { name = name.dropFirst().dropLast() }
                return environment[String(name)] ?? String(match.output.0)
            }
        }
        let hfHome = expand(environment["HF_HOME"] ?? ((environment["XDG_CACHE_HOME"] ?? home + "/.cache") as NSString).appendingPathComponent("huggingface"))
        let legacy = environment["HUGGINGFACE_HUB_CACHE"] ?? (hfHome as NSString).appendingPathComponent("hub")
        return URL(fileURLWithPath: expand(environment["HF_HUB_CACHE"] ?? legacy), isDirectory: true)
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
        case .system: String(localized: "System")
        case .light: String(localized: "Light")
        case .dark: String(localized: "Dark")
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
