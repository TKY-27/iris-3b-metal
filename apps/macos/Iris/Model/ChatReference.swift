import CoreTransferable
import UniformTypeIdentifiers

/// A chat being dragged, e.g. onto a project in the sidebar. Its own type, so dropping it
/// into a text field never pastes an identifier.
struct ChatReference: Codable, Transferable {
    let id: UUID

    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .irisChat)
    }
}

extension UTType {
    /// Declared in the app's Info.plist (project.yml).
    static let irisChat = UTType(exportedAs: "io.github.tky27.iris.chat")
}
