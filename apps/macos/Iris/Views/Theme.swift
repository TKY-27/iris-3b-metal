import SwiftUI

/// The few measures the app's own surfaces share. Color comes from the system (accent, labels,
/// separators): no brand gradients, glows or shadows.
enum Theme {
    /// The column the conversation and composer share.
    static let readableWidth: CGFloat = 760
    static let composerRadius: CGFloat = 16
    static let cardRadius: CGFloat = 10

    /// Critically damped, as UI that nobody flicked should be.
    static let spring = Animation.spring(response: 0.3, dampingFraction: 1)
}

extension ShapeStyle where Self == Color {
    /// The text field background: white in light appearance, raised grey in dark.
    static var fieldBackground: Color { Color(nsColor: .textBackgroundColor) }
}
