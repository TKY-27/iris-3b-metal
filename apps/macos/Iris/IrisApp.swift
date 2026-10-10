import SwiftUI

@main
struct IrisApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @AppStorage(AppAppearance.key) private var appearance = AppAppearance.system

    var body: some Scene {
        Window("Iris", id: "main") {
            ContentView()
                .environment(delegate.engine)
                .environment(delegate.library)
                .environment(delegate.generator)
                .environment(delegate.workspace)
                .frame(minWidth: 760, minHeight: 520)
                .task { if !AppDelegate.isTestHost { await delegate.engine.start() } }
                .onChange(of: appearance, initial: true) { NSApp.appearance = appearance.nsAppearance }
        }
        .defaultSize(width: 1180, height: 820)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { delegate.workspace.newConversationHere(library: delegate.library) }
                    .keyboardShortcut("n")
                Button("New Project…") { delegate.workspace.naming = .newProject(moving: nil) }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
            }
        }

        Settings {
            SettingsView()
                .environment(delegate.engine)
                .environment(delegate.library)
                .environment(delegate.generator)
        }
    }
}

/// Owns the app's state, and stops a running request and a launched engine on quit.
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Unit tests run inside the app; they must not touch the user's library or start an engine.
    static let isTestHost = ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil

    @MainActor let engine = EngineController()
    @MainActor let library = Library(store: LibraryStore(
        root: isTestHost ? .temporaryDirectory.appending(path: "IrisTestHost-\(UUID().uuidString)") : LibraryStore.defaultRoot
    ))
    @MainActor let generator = Generator()
    @MainActor lazy var workspace = Workspace(
        library: library, defaults: Self.isTestHost ? UserDefaults(suiteName: "IrisTestHost") ?? .standard : .standard
    )

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true  // a launched engine holds several GB of memory
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated {
            guard generator.isBusy else { return .terminateNow }
            let alert = NSAlert()
            alert.messageText = "Stop the current request and quit?"
            alert.informativeText = generator.queued.isEmpty
                ? "The image in progress will not be finished."
                : "The image in progress will not be finished, and the \(generator.queued.count) queued after it will not be made."
            alert.addButton(withTitle: "Quit")
            alert.addButton(withTitle: "Cancel")
            guard alert.runModal() == .alertFirstButtonReturn else { return .terminateCancel }
            Task {
                await generator.cancelAndWait(engine: engine)
                sender.reply(toApplicationShouldTerminate: true)
            }
            return .terminateLater
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            engine.stop()
            library.flush()
        }
    }
}
