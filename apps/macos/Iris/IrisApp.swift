import SwiftUI

@main
struct IrisApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @State private var history = History()
    @State private var generator = Generator()

    var body: some Scene {
        Window("Iris", id: "main") {
            ContentView()
                .environment(delegate.engine)
                .environment(history)
                .environment(generator)
                .frame(minWidth: 820, minHeight: 560)
                .task { await delegate.engine.start() }
        }
        .defaultSize(width: 1180, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }

        Settings {
            SettingsView()
                .environment(delegate.engine)
                .environment(history)
        }
    }
}

/// Stops an engine this app launched when the app quits.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor let engine = EngineController()

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { engine.stop() }
    }
}
