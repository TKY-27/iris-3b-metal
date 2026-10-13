import AppKit
import SwiftUI

/// An image file decoded off the main thread at no more than `maxPixels` on its long side.
struct StoredImage: View {
    let url: URL
    let maxPixels: Int
    /// Reserves the final shape while loading, so the layout does not jump.
    var aspectRatio: CGFloat?
    var placeholder = "photo"
    @State private var image: CGImage?
    @State private var missing = false

    var body: some View {
        Group {
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(CGFloat(image.width) / CGFloat(image.height), contentMode: .fit)
            } else {
                Rectangle()
                    .fill(.quaternary)
                    .aspectRatio(aspectRatio ?? 1, contentMode: .fit)
                    .overlay {
                        if missing {
                            Image(systemName: placeholder).foregroundStyle(.secondary)
                        }
                    }
            }
        }
        .task(id: url) {
            image = nil
            missing = false
            let loaded = await ImageFiles.thumbnail(of: url, maxPixels: maxPixels)
            guard !Task.isCancelled else { return }
            image = loaded
            missing = loaded == nil
        }
    }
}

/// Copy and Save As for result files.
@MainActor
enum ImageActions {
    static func copy(_ url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setData(data, forType: .png)
    }

    static func saveAs(_ url: URL, suggestedName: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = suggestedName
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            // the panel already confirmed replacing an existing file
            if FileManager.default.fileExists(atPath: target.path) {
                try replace(target, with: url)
            } else {
                try FileManager.default.copyItem(at: url, to: target)
            }
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    /// `replaceItemAt` moves its source, so it gets a copy on the target's volume.
    private static func replace(_ target: URL, with url: URL) throws {
        let fileManager = FileManager.default
        let directory = try fileManager.url(
            for: .itemReplacementDirectory, in: .userDomainMask, appropriateFor: target, create: true
        )
        defer { try? fileManager.removeItem(at: directory) }
        let copy = directory.appending(path: target.lastPathComponent)
        try fileManager.copyItem(at: url, to: copy)
        _ = try fileManager.replaceItemAt(target, withItemAt: copy)
    }
}
