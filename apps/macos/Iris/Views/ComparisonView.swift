import SwiftUI

/// Two versions of one picture in a single frame, split by a divider: `before` on the leading
/// side, `after` on the trailing side. Click or drag anywhere to move the divider; with focus,
/// the arrow keys move it in steps.
struct ComparisonView: View {
    let before: CGImage
    let after: CGImage
    let beforeTitle: String
    let afterTitle: String
    /// The divider's position, 0 at the leading edge.
    @Binding var split: Double
    @FocusState private var focused: Bool

    var body: some View {
        GeometryReader { geometry in
            let x = geometry.size.width * split
            ZStack(alignment: .topLeading) {
                picture(after)
                picture(before)
                    .mask(alignment: .leading) { Rectangle().frame(width: x) }
                DividerHandle(focused: focused)
                    .frame(height: geometry.size.height)
                    .position(x: x, y: geometry.size.height / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onChanged { value in
                    split = (value.location.x / max(geometry.size.width, 1)).clamped(to: 0...1)
                }
            )
            .pointerStyle(.columnResize)
        }
        // the handle shows focus; the window gives it focus, so the arrow keys work at once
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .defaultFocus($focused, true)
        .onKeyPress(keys: [.leftArrow, .rightArrow]) { press in
            nudge((press.key == .leftArrow ? -1 : 1) * (press.modifiers.contains(.shift) ? 0.2 : 1))
            return .handled
        }
        .accessibilityElement()
        .accessibilityLabel(String(localized: "Comparison of \(beforeTitle) and \(afterTitle)"))
        .accessibilityValue(
            String(localized: "\(beforeTitle) \(Int((split * 100).rounded())) percent, \(afterTitle) \(100 - Int((split * 100).rounded())) percent")
        )
        .accessibilityAdjustableAction { direction in
            nudge(direction == .increment ? 1 : -1)
        }
    }

    private func picture(_ image: CGImage) -> some View {
        // the same smooth resampling for both, so a smaller original is not shown as blocks
        Image(decorative: image, scale: 1)
            .resizable()
            .interpolation(.high)
    }

    /// 5% a press, 1% with Shift.
    private func nudge(_ direction: Double) {
        split = (split + direction * 0.05).clamped(to: 0...1)
    }
}

/// A thin line that stays visible on light and dark images, with a grip in the middle that
/// takes the focus ring.
private struct DividerHandle: View {
    let focused: Bool

    var body: some View {
        ZStack {
            Rectangle()
                .fill(.white)
                .frame(width: 2)
                .overlay { Rectangle().stroke(.black.opacity(0.25), lineWidth: 0.5) }
            Image(systemName: "arrow.left.and.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.black.opacity(0.75))
                .frame(width: 28, height: 28)
                .background(Circle().fill(.white))
                .overlay { Circle().strokeBorder(.black.opacity(0.2), lineWidth: 0.5) }
                .overlay {
                    if focused { Circle().strokeBorder(Color.accentColor, lineWidth: 2).padding(-3) }
                }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}
