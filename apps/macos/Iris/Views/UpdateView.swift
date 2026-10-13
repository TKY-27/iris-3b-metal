import SwiftUI

struct CheckForUpdatesButton: View {
    @Environment(UpdateChecker.self) private var updates
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button(String(localized: "Check for Updates…")) {
            openWindow(id: "updates")
            Task { await updates.check(manual: true) }
        }
        .disabled(updates.isChecking)
    }
}

struct UpdateView: View {
    @Environment(UpdateChecker.self) private var updates
    @Environment(Generator.self) private var generator

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title).font(.title2).fontWeight(.semibold)
            Text(String(localized: "Installed: Iris \(updates.appVersion) (\(UpdateChecker.installedBuild))"))
                .foregroundStyle(.secondary)
            if updates.isChecking {
                ProgressView(String(localized: "Checking GitHub Releases…")).controlSize(.small)
            }
            if let release = updates.available {
                if !release.notes.isEmpty {
                    ScrollView {
                        Text(verbatim: release.notes)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 160)
                }
                Text(String(localized: "This build supports release checking. Download and installation are managed separately."))
                if generator.isBusy {
                    Text(String(localized: "Wait for generation to finish before replacing the app."))
                        .foregroundStyle(.secondary)
                }
                Link(String(localized: "View Release"), destination: release.url)
            } else if !updates.isChecking, updates.message == nil {
                Text(updates.lastChecked == nil ? String(localized: "Check for published macOS releases.") : String(localized: "No newer compatible macOS release was found."))
            }
            if let message = updates.message {
                Text(message).foregroundStyle(.secondary)
            }
            if let checked = updates.lastChecked {
                Text(String(localized: "Last checked: \(checked.formatted(Date.FormatStyle(date: .abbreviated, time: .shortened).locale(.iris)))"))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Divider()
            HStack {
                SettingsLink { Text("Settings…") }
                Spacer()
                CheckForUpdatesButton()
            }
        }
        .padding(24)
        .frame(width: 480, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var title: String {
        updates.available.map { String(localized: "Iris \($0.displayVersion) is available") } ?? String(localized: "Software Update")
    }
}
