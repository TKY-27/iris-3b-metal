import SwiftUI

struct GalleryVisibilityButton: View {
    @Environment(Workspace.self) private var workspace

    var body: some View {
        Button(action: workspace.toggleGallery) {
            Label(
                workspace.showsGallery ? String(localized: "Hide Gallery") : String(localized: "Show Gallery"),
                systemImage: "photo.on.rectangle"
            )
        }
        .help(workspace.showsGallery ? String(localized: "Hide Gallery (⇧⌘G)") : String(localized: "Show Gallery (⇧⌘G)"))
        .accessibilityValue(workspace.showsGallery ? String(localized: "Visible") : String(localized: "Hidden"))
    }
}

struct GenerationStatus: View {
    @Environment(Generator.self) private var generator
    @Environment(Library.self) private var library

    var body: some View {
        Group {
            if let active = generator.active, let entry = library.entry(active.entry)?.entry {
                let text = ProgressText(generator.progress, workflow: entry.workflow)
                VStack(spacing: 6) {
                    Text(text.headline).font(.headline)
                    if let detail = text.detail { Text(detail).foregroundStyle(.secondary) }
                    ProgressView(value: generator.progress.fraction)
                        .frame(maxWidth: 280)
                        .accessibilityValue(ProgressText.spoken(generator.progress, workflow: entry.workflow))
                }
                .padding()
            }
        }
        .onChange(of: generator.isBusy) { wasBusy, isBusy in
            if wasBusy && !isBusy {
                AccessibilityNotification.Announcement(String(localized: "The request ended")).post()
            }
        }
    }
}
