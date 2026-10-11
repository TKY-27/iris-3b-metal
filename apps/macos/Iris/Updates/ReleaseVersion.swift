import Foundation

struct ReleaseVersion: Comparable, Sendable {
    let major: Int
    let minor: Int
    let patch: Int

    init?(_ value: String) {
        let value = value.hasPrefix("v") ? String(value.dropFirst()) : value
        guard value.range(of: #"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?\z"#,
                          options: .regularExpression) != nil else { return nil }
        let parts = value.split(separator: "+")[0].split(separator: ".")
        guard let major = Int(parts[0]), let minor = Int(parts[1]), let patch = Int(parts[2]) else { return nil }
        self.major = major
        self.minor = minor
        self.patch = patch
    }

    static func < (lhs: Self, rhs: Self) -> Bool {
        (lhs.major, lhs.minor, lhs.patch) < (rhs.major, rhs.minor, rhs.patch)
    }
}

struct AppRelease: Codable, Equatable, Sendable {
    let tag: String
    let notes: String
    let minimumSystemVersion: String
    let architectures: [String]

    var version: ReleaseVersion? { ReleaseVersion(tag) }
    var displayVersion: String { tag.hasPrefix("v") ? String(tag.dropFirst()) : tag }
    var url: URL { URL(string: "https://github.com/TKY-27/iris-3b-metal/releases/tag/\(tag)")! }

    func isCompatible(os: OperatingSystemVersion, architecture: String) -> Bool {
        let parts = minimumSystemVersion.split(separator: ".", omittingEmptySubsequences: false)
        guard (2...3).contains(parts.count),
              parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy({ $0.isASCII && $0.isNumber }) }),
              let major = Int(parts[0]), let minor = Int(parts[1]),
              let patch = Int(parts.count == 3 ? parts[2] : "0") else { return false }
        return architectures.contains(architecture)
            && (os.majorVersion, os.minorVersion, os.patchVersion) >= (major, minor, patch)
    }

    static func parse(_ data: Data) throws -> [Self] {
        guard let objects = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw UpdateFailure.invalidMetadata
        }
        return objects.compactMap { object in
            guard object["draft"] as? Bool == false, object["prerelease"] as? Bool == false,
                  let tag = object["tag_name"] as? String, ReleaseVersion(tag) != nil,
                  let body = object["body"] as? String else { return nil }
            let prefix = "<!-- iris-macos:"
            let compatibility: Compatibility
            var notes = body
            if let start = body.range(of: prefix) {
                guard let end = body.range(of: "-->", range: start.upperBound..<body.endIndex),
                      let metadata = body[start.upperBound..<end.lowerBound].data(using: .utf8),
                      let decoded = try? JSONDecoder().decode(Compatibility.self, from: metadata) else { return nil }
                compatibility = decoded
                notes.removeSubrange(start.lowerBound..<end.upperBound)
            } else {
                // The initial release predates compatibility metadata.
                guard tag == "v0.3.0" else { return nil }
                compatibility = Compatibility(minimumSystemVersion: "15.0", architectures: ["arm64"])
            }
            return Self(tag: tag, notes: String(notes.trimmingCharacters(in: .whitespacesAndNewlines).prefix(800)),
                        minimumSystemVersion: compatibility.minimumSystemVersion,
                        architectures: compatibility.architectures)
        }
    }

    private struct Compatibility: Decodable {
        let minimumSystemVersion: String
        let architectures: [String]
    }
}
