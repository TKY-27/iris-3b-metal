import SwiftUI

/// A depth map or upscale next to the image it was made from, in a window of its own so both
/// can be zoomed to the pixel.
struct ComparisonWindow: View {
    static let id = "compare"

    @Environment(Library.self) private var library
    let entryID: UUID

    var body: some View {
        if let found = library.entry(entryID), found.entry.isDone, found.entry.request == nil {
            ComparisonViewer(
                entry: found.entry, before: library.inputURL(entryID), after: library.imageURL(entryID)
            )
            // Distinguish comparisons in the Window menu.
            .navigationTitle("\(Conversation.abbreviatedTitle(found.entry.displayTitle)) — \(ComparisonViewer.afterTitle(found.entry.workflow))")
        } else {
            ContentUnavailableView(
                String(localized: "Nothing to Compare"), systemImage: "square.split.2x1",
                description: Text(String(localized: "This result is no longer in the library."))
            )
            .navigationTitle(String(localized: "Compare"))
        }
    }
}

private struct ComparisonViewer: View {
    @Environment(\.displayScale) private var displayScale
    let entry: Entry
    let before: URL
    let after: URL
    @State private var images: (before: CGImage, after: CGImage)?
    @State private var missing = false
    @State private var split = 0.5
    /// Screen pixels per pixel of the result; nil fits the window.
    @State private var zoom: CGFloat?
    @State private var pinchStart: CGFloat?
    @State private var viewport = CGSize.zero
    @State private var visible = CGRect.zero
    @State private var position = ScrollPosition()

    private static let steps: [CGFloat] = [0.25, 0.5, 1, 2, 4, 8]
    /// Room around the image, with its two labels above it.
    private static let margin: CGFloat = 32

    var body: some View {
        Group {
            if let images {
                viewer(images)
            } else if missing {
                ContentUnavailableView(
                    String(localized: "Original Not Available"), systemImage: "photo",
                    description: Text(String(localized: "Iris keeps the original of results made since version 0.2."))
                )
            } else {
                ProgressView()
            }
        }
        .frame(minWidth: 480, minHeight: 360)
        .navigationSubtitle(subtitle)
        .toolbar { zoomControls }
        .task(id: after) {
            async let original = ImageFiles.thumbnail(of: before, maxPixels: 8192)
            async let result = ImageFiles.thumbnail(of: after, maxPixels: 8192)
            if let original = await original, let result = await result {
                images = (original, result)
            } else {
                missing = true
            }
        }
    }

    private func viewer(_ images: (before: CGImage, after: CGImage)) -> some View {
        GeometryReader { geometry in
            let size = displaySize(images.after, scale: scale(images.after))
            ScrollView([.horizontal, .vertical]) {
                ComparisonView(
                    before: images.before, after: images.after,
                    beforeTitle: String(localized: "comparison.original", defaultValue: "Original"), afterTitle: afterTitle, split: $split
                )
                .frame(width: size.width, height: size.height)
                .overlay(alignment: .topLeading) { SideLabel(String(localized: "comparison.original", defaultValue: "Original")) }
                .overlay(alignment: .topTrailing) { SideLabel(afterTitle) }
                .padding(Self.margin)
                .frame(minWidth: geometry.size.width, minHeight: geometry.size.height)
            }
            .scrollPosition($position)
            .onScrollGeometryChange(for: CGRect.self, of: \.visibleRect) { visible = $1 }
            .onChange(of: geometry.size, initial: true) { viewport = $1 }
        }
        .simultaneousGesture(
            MagnifyGesture()
                .onChanged { value in
                    let start = pinchStart ?? scale(images.after)
                    pinchStart = start
                    setZoom(start * value.magnification, of: images.after)
                }
                .onEnded { _ in pinchStart = nil }
        )
    }

    // MARK: - Zoom

    @ToolbarContentBuilder private var zoomControls: some ToolbarContent {
        let current = images.map { scale($0.after) }
        ToolbarItem {
            ControlGroup {
                Button(String(localized: "Zoom Out"), systemImage: "minus.magnifyingglass") { step(-1) }
                    .keyboardShortcut("-")
                    .help(String(localized: "Zoom Out (⌘-)"))
                    .disabled(current.map { $0 <= minimumScale * 1.01 } ?? true)
                Button(String(localized: "Zoom In"), systemImage: "plus.magnifyingglass") { step(1) }
                    .keyboardShortcut("=")
                    .help(String(localized: "Zoom In (⌘+)"))
                    .disabled(current.map { $0 >= Self.steps.last! * 0.99 } ?? true)
            }
        }
        // words, not two more magnifying glasses; each is off while it is what is shown
        ToolbarItem {
            ControlGroup {
                Button(String(localized: "Fit")) { setZoom(nil) }
                    .keyboardShortcut("9")
                    .help(String(localized: "Zoom to Fit (⌘9)"))
                    .accessibilityLabel(String(localized: "Zoom to Fit"))
                    .disabled(images == nil || zoom == nil)
                Button("100%") { setZoom(1) }
                    .keyboardShortcut("0")
                    .help(String(localized: "Actual Size (⌘0): one pixel of the result per screen pixel"))
                    .accessibilityLabel(String(localized: "Actual Size"))
                    .disabled(images == nil || zoom == 1)
            }
            .labelStyle(.titleOnly)
        }
    }

    private var minimumScale: CGFloat {
        images.map { min(Self.steps.first!, fitScale($0.after)) } ?? Self.steps.first!
    }

    /// The fitting scale: the whole result in the window, enlarged at most to twice its pixels.
    private func fitScale(_ image: CGImage) -> CGFloat {
        let points = CGSize(width: CGFloat(image.width) / displayScale, height: CGFloat(image.height) / displayScale)
        let room = CGSize(width: viewport.width - 2 * Self.margin, height: viewport.height - 2 * Self.margin)
        guard points.width > 0, points.height > 0, room.width > 0, room.height > 0 else { return 1 }
        return min(room.width / points.width, room.height / points.height, 2)
    }

    private func scale(_ image: CGImage) -> CGFloat { zoom ?? fitScale(image) }

    private func displaySize(_ image: CGImage, scale: CGFloat) -> CGSize {
        CGSize(width: CGFloat(image.width) / displayScale * scale, height: CGFloat(image.height) / displayScale * scale)
    }

    private func step(_ direction: Int) {
        guard let image = images?.after else { return }
        let current = scale(image)
        let next = direction > 0
            ? Self.steps.first { $0 > current * 1.01 } ?? Self.steps.last!
            : Self.steps.last { $0 < current * 0.99 } ?? Self.steps.first!
        setZoom(next)
    }

    private func setZoom(_ new: CGFloat?) {
        guard let image = images?.after else { return }
        setZoom(new, of: image)
    }

    /// Changes the zoom, keeping the point at the middle of the window where it is.
    private func setZoom(_ new: CGFloat?, of image: CGImage) {
        let old = displaySize(image, scale: scale(image))
        zoom = new.map { $0.clamped(to: minimumScale...Self.steps.last!) }
        let size = displaySize(image, scale: scale(image))
        guard size != old, old.width > 0, old.height > 0 else { return }
        // where the image starts in the scrolled content: after the margin, or centered when smaller
        func origin(_ size: CGSize) -> CGPoint {
            CGPoint(x: max(Self.margin, (viewport.width - size.width) / 2), y: max(Self.margin, (viewport.height - size.height) / 2))
        }
        let (from, to) = (origin(old), origin(size))
        // within the content, which is the image and its margins, or the window when smaller
        let limit = CGPoint(
            x: max(0, size.width + 2 * Self.margin - viewport.width),
            y: max(0, size.height + 2 * Self.margin - viewport.height)
        )
        let target = CGPoint(
            x: ((visible.midX - from.x) * size.width / old.width + to.x - viewport.width / 2).clamped(to: 0...limit.x),
            y: ((visible.midY - from.y) * size.height / old.height + to.y - viewport.height / 2).clamped(to: 0...limit.y)
        )
        // after the content has its new size
        DispatchQueue.main.async { position.scrollTo(point: target) }
    }

    // MARK: - Text

    private var afterTitle: String { Self.afterTitle(entry.workflow) }

    static func afterTitle(_ workflow: Workflow) -> String {
        workflow == .depth ? String(localized: "Depth Map") : String(localized: "Restored and Upscaled")
    }

    private var subtitle: String {
        guard let images else { return "" }
        let original = "\(images.before.width) × \(images.before.height)"
        let result = "\(images.after.width) × \(images.after.height)"
        let percent = Int((scale(images.after) * 100).rounded())
        return entry.workflow == .depth
            ? String(localized: "Depth map of the \(original) original · \(percent)%")
            : "\(original) → \(result) · \(percent)%"
    }
}

/// Names a side of the comparison, just above that corner of the image.
private struct SideLabel: View {
    let title: String

    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title)
            .font(.callout)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .alignmentGuide(.top) { $0[.bottom] + 6 }
            .allowsHitTesting(false)
            .accessibilityHidden(true)  // the comparison names both sides
    }
}
