import SwiftUI

extension View {
    /// Shared naming and deletion dialogs.
    func libraryEditing() -> some View { modifier(LibraryEditing()) }
}

private struct LibraryEditing: ViewModifier {
    @Environment(Library.self) private var library
    @Environment(Workspace.self) private var workspace
    @State private var name = ""

    func body(content: Content) -> some View {
        content
            .onChange(of: workspace.naming) { _, naming in
                name = switch naming {
                case let .project(id): library.project(id)?.name ?? ""
                case .newProject, nil: ""
                }
            }
            .alert(namingTitle, isPresented: isNaming) {
                TextField(String(localized: "Name"), text: $name)
                Button(creates ? String(localized: "Create") : String(localized: "Rename"), action: commitName)
                    .keyboardShortcut(.defaultAction)
                Button(String(localized: "Cancel"), role: .cancel) { workspace.naming = nil }
            }
            .confirmationDialog(deletionTitle, isPresented: isDeleting, titleVisibility: .visible) {
                Button(String(localized: "Delete"), role: .destructive, action: commitDeletion)
            } message: {
                Text(deletionMessage)
            }
    }

    // MARK: - Naming

    private var isNaming: Binding<Bool> {
        Binding(get: { workspace.naming != nil }, set: { if !$0 { workspace.naming = nil } })
    }

    private var creates: Bool {
        if case .newProject = workspace.naming { true } else { false }
    }

    private var namingTitle: String {
        switch workspace.naming {
        case .newProject: String(localized: "New Project")
        case .project: String(localized: "Rename Project")
        case nil: ""
        }
    }

    private func commitName() {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        defer { workspace.naming = nil }
        guard !trimmed.isEmpty else { return }
        switch workspace.naming {
        case let .newProject(moving):
            guard let project = library.createProject(named: trimmed) else { return }
            if let moving {
                library.move(entry: moving, to: project)
            } else {
                workspace.selection = .project(project)
            }
        case let .project(id):
            library.rename(project: id, to: trimmed)
        case nil:
            break
        }
    }

    // MARK: - Deletion

    private var isDeleting: Binding<Bool> {
        Binding(get: { workspace.deleting != nil }, set: { if !$0 { workspace.deleting = nil } })
    }

    private var deletionTitle: String {
        switch workspace.deleting {
        case let .project(id): String(localized: "Delete “\(library.project(id)?.name ?? "")”?")
        case .image: String(localized: "Delete this image?")
        case nil: ""
        }
    }

    private var deletionMessage: String {
        switch workspace.deleting {
        case .project: String(localized: "Its images are kept and stay in All Images.")
        case .image: String(localized: "The image and the record of its prompt and settings are deleted. This cannot be undone.")
        case nil: ""
        }
    }

    private func commitDeletion() {
        switch workspace.deleting {
        case let .project(id): library.delete(project: id)
        case let .image(id): library.delete(entry: id)
        case nil: break
        }
        workspace.deleting = nil
    }
}
