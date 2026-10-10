import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            Tab("Engine", systemImage: "cpu") { EngineSettings() }
            Tab("Storage", systemImage: "internaldrive") { StorageSettings() }
        }
        .frame(width: 520)
    }
}

private struct EngineSettings: View {
    @Environment(EngineController.self) private var engine
    @State private var path = ""
    @State private var port = 8765

    var body: some View {
        Form {
            Section {
                LabeledContent("Status") { Text(status) }
                TextField("iris-metal location", text: $path, prompt: Text(engine.resolvedExecutable()?.path ?? "Not found"))
                Button("Choose…", action: choose)
                TextField("Port", value: $port, format: .number.grouping(.never))
                    .help("\(EngineController.portRange.lowerBound)–\(EngineController.portRange.upperBound)")
            } footer: {
                Text("Iris runs the iris-metal engine (see the README to install it) as a local server on 127.0.0.1. Leave the location empty to search ~/.local/bin and Homebrew.")
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
    @Environment(Generator.self) private var generator
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
                LabeledContent("^[\(library.conversations.count) conversation](inflect: true)") {
                    if let libraryBytes {
                        Text(libraryBytes, format: .byteCount(style: .file))
                    }
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([library.store.root])
                    }
                }
                HStack {
                    Spacer()
                    Button("Delete All Conversations…", role: .destructive) { confirmsDelete = true }
                        .disabled(library.conversations.isEmpty)
                }
            }
        }
        .formStyle(.grouped)
        .task { models = await ModelCache.scan() }
        .task(id: library.entryCount) { libraryBytes = await library.byteCount() }
        .confirmationDialog("Delete all conversations and their images?", isPresented: $confirmsDelete) {
            Button("Delete All", role: .destructive) {
                library.deleteAllConversations(except: generator.active?.conversation)
            }
        } message: {
            Text("Projects are kept. A request that is running is not deleted. This cannot be undone.")
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
