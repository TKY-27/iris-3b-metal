import AppKit
import XCTest

@testable import Iris

final class LocalizationTests: XCTestCase {
    private func languageBundle(_ language: String) throws -> Bundle {
        let app = Bundle(for: AppDelegate.self)
        let path = try XCTUnwrap(app.path(forResource: language, ofType: "lproj"))
        return try XCTUnwrap(Bundle(path: path))
    }

    func testJapaneseStringsAndInterpolation() throws {
        let japanese = try languageBundle("ja")
        XCTAssertEqual(String(localized: "Hide Gallery", bundle: japanese), "ギャラリーを非表示")
        XCTAssertEqual(String(localized: "Search images", bundle: japanese), "画像を検索")
        let name = "旅の写真"
        XCTAssertEqual(
            String(localized: "Describe a new image for “\(name)”", bundle: japanese),
            "「旅の写真」で生成したい画像を説明してください"
        )
        let seed = 2147483647
        XCTAssertEqual(String(localized: "Seed \(String(seed))", bundle: japanese), "シード 2147483647")
    }

    func testNativeImageAndStepPlurals() throws {
        let english = try languageBundle("en")
        let japanese = try languageBundle("ja")
        for count in [0, 1, 2, 4] {
            XCTAssertEqual(
                String(localized: "\(count) images", bundle: english, locale: Locale(identifier: "en")),
                count == 1 ? "1 image" : "\(count) images"
            )
            XCTAssertEqual(
                String(localized: "\(count) images", bundle: japanese, locale: Locale(identifier: "ja")),
                "\(count)枚"
            )
        }
        for count in [1, 4, 20] {
            XCTAssertEqual(
                String(localized: "\(count) steps", bundle: english, locale: Locale(identifier: "en")),
                count == 1 ? "1 step" : "\(count) steps"
            )
            let model = "Turbo"
            XCTAssertEqual(
                String(localized: "\(model) · \(count) steps", bundle: english, locale: Locale(identifier: "en")),
                "Turbo · \(count) \(count == 1 ? "step" : "steps")"
            )
        }
    }

    func testDisplayLanguageKeepsRegionalFormatting() {
        let japanese = Locale.displayLocale(language: "ja", regional: Locale(identifier: "en_US"))
        XCTAssertEqual(japanese.language.languageCode, "ja")
        XCTAssertEqual(japanese.region, "US")
        XCTAssertEqual(Duration.seconds(65).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated).locale(japanese)), "1分 5秒")
        let date = Date(timeIntervalSince1970: 0)
        let style = Date.FormatStyle(locale: japanese, timeZone: TimeZone(secondsFromGMT: 0)!).month(.wide)
        XCTAssertEqual(date.formatted(style), "1月")
    }

    func testCompatibilityIsBoundedToSupportedEngineVersions() {
        for version in ["0.3.0", "0.3.9", "0.5.0", "0.5.7", "0.6.0", "0.6.9"] {
            XCTAssertNil(EngineHealth(resident: nil, queued: 0, engineVersion: version, apiVersion: 1).compatibilityIssue)
        }
        for version in ["0.4.0", "0.7.0", "1.0.0"] {
            XCTAssertNotNil(EngineHealth(resident: nil, queued: 0, engineVersion: version, apiVersion: 1).compatibilityIssue)
        }
        XCTAssertNotNil(EngineHealth(resident: nil, queued: 0, engineVersion: "0.5.0", apiVersion: 2).compatibilityIssue)
    }

    func testEnglishFallbackAndDiagnosticPreservation() throws {
        let app = Bundle(for: AppDelegate.self)
        XCTAssertEqual(
            Bundle.preferredLocalizations(from: app.localizations, forPreferences: ["fr"]), ["en"]
        )
        XCTAssertEqual(String(localized: "Show Gallery", bundle: try languageBundle("en")), "Show Gallery")
        XCTAssertEqual(EngineError.unreachable.diagnosticDescription, "The engine is not running.")
        XCTAssertEqual(GenerationModel.original.fullTitle, "Iris-3B Original")
        let entry = Entry(workflow: .upscale, inputName: "風景.png", outcome: .failed("raw diagnostic"))
        let decoded = try JSONDecoder().decode(Entry.self, from: JSONEncoder().encode(entry))
        XCTAssertEqual(decoded.title, "風景.png, restored and upscaled")
        XCTAssertEqual(decoded.inputName, "風景.png")
        XCTAssertEqual(decoded.outcome, .failed("raw diagnostic"))
    }
}

@MainActor
final class GalleryVisibilityTests: XCTestCase {
    func testVisibilityPersistsAndHidingRequestsPromptFocus() {
        let name = "IrisGalleryVisibilityTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let workspace = Workspace(defaults: defaults)
        XCTAssertTrue(workspace.showsGallery)
        let selection = UUID()
        workspace.selectedEntry = selection
        workspace.prompt = "同じプロンプト"
        workspace.searchText = "fox"
        workspace.toggleGallery()
        XCTAssertFalse(workspace.showsGallery)
        XCTAssertEqual(workspace.promptFocusRequests, 1)
        XCTAssertEqual(workspace.selectedEntry, selection)
        XCTAssertEqual(workspace.prompt, "同じプロンプト")
        XCTAssertEqual(workspace.searchText, "fox")
        XCTAssertFalse(Workspace(defaults: defaults).showsGallery)
        workspace.toggleGallery()
        XCTAssertTrue(workspace.showsGallery)
        XCTAssertEqual(workspace.promptFocusRequests, 1)
    }

    func testFindRestoresGalleryAndRequestsSearchFocus() {
        let name = "IrisGallerySearchTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let workspace = Workspace(defaults: defaults)
        workspace.showsGallery = false
        workspace.focusSearch()
        XCTAssertTrue(workspace.showsGallery)
        XCTAssertEqual(workspace.searchFocusRequests, 1)
        XCTAssertTrue(workspace.searchFocusPending)
        workspace.didFocusSearch()
        XCTAssertFalse(workspace.searchFocusPending)
        XCTAssertTrue(Workspace(defaults: defaults).showsGallery)
    }

    func testSearchFocusWaitsForWindowConnection() async throws {
        let field = FocusableSearchField(frame: NSRect(x: 0, y: 0, width: 200, height: 24))
        var applied = false
        field.onFocused = { applied = true }
        field.requestFocus()
        XCTAssertNil(field.window)
        XCTAssertFalse(applied)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 300, height: 100),
            styleMask: [.titled], backing: .buffered, defer: false
        )
        window.contentView?.addSubview(field)
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertTrue(applied)
        XCTAssertNotNil(field.currentEditor())
        field.removeFromSuperview()
    }

    func testRevealRestoresGalleryAndPreservesTheEntry() throws {
        let name = "IrisGalleryRevealTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        let root = URL.temporaryDirectory.appending(path: name)
        defer {
            defaults.removePersistentDomain(forName: name)
            try? FileManager.default.removeItem(at: root)
        }
        let library = Library(store: LibraryStore(root: root))
        let workspace = Workspace(defaults: defaults)
        workspace.showsGallery = false
        let entry = UUID()
        workspace.reveal(entry, library: library)
        XCTAssertTrue(workspace.showsGallery)
        XCTAssertEqual(workspace.selectedEntry, entry)
    }
}
