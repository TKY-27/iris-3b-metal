import XCTest

@testable import Iris

@MainActor
final class ComposerModelTests: XCTestCase {
    /// One reused, emptied domain: the user's preferences are never touched and no files pile up.
    private func scratchDefaults() throws -> UserDefaults {
        let suite = "io.github.tky27.iris.tests"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        addTeardownBlock { defaults.removePersistentDomain(forName: suite) }
        return defaults
    }

    func testPresetsNameTheirNumbersAndFindTheNearest() {
        XCTAssertEqual(QualityPreset.allCases.map(\.steps), [10, 15, 20, 30, 50])
        XCTAssertEqual(GenerationSettings().steps, QualityPreset.standard.steps)
        XCTAssertEqual(GenerationSettings().cfgScale, 3)
        XCTAssertNil(QualityPreset(steps: 100))
        XCTAssertEqual(QualityPreset.nearest(to: 100), .maximum)
        XCTAssertEqual(QualityPreset.nearest(to: 17), .fast)
        XCTAssertEqual(AdherencePreset.nearest(to: 3.4), .balanced)

        var settings = GenerationSettings()
        settings.steps = 48
        settings.cfgScale = 3.25
        XCTAssertEqual(settings.qualityTitle, "48 steps")
        XCTAssertEqual(settings.adherenceTitle, "Guidance 3.25")
    }

    func testABatchHasOneRequestPerImageWithItsOwnSeed() {
        var settings = GenerationSettings()
        settings.randomizeSeed = false
        settings.seed = 41
        let fixed = settings.requests(for: "fox", count: 3)
        XCTAssertEqual(fixed.map(\.seed), [41, 42, 43])  // made again exactly with the same settings
        XCTAssertEqual(Set(fixed.map(\.prompt)), ["fox"])
        settings.seed = GenerationSettings.seedRange.upperBound
        XCTAssertEqual(settings.requests(for: "fox", count: 2).map(\.seed), [GenerationSettings.seedRange.upperBound, 0])
        settings.randomizeSeed = true
        XCTAssertEqual(settings.requests(for: "fox", count: 4).count, 4)
        XCTAssertEqual(settings.requests(for: "fox", count: 0).count, 1)
    }

    func testRestoringDefaultsKeepsTheSizeAndIgnoresADrawnSeed() {
        var settings = GenerationSettings()
        settings.size = ImageSize(width: 1344, height: 768)
        _ = settings.request(for: "fox")  // draws a seed
        XCTAssertTrue(settings.hasDefaultTuning)
        settings.steps = 50
        settings.negativePrompt = "blur"
        XCTAssertFalse(settings.hasDefaultTuning)
        settings.restoreDefaults()
        XCTAssertEqual(settings.size, ImageSize(width: 1344, height: 768))
        XCTAssertEqual([settings.steps], [20])
        XCTAssertEqual(settings.negativePrompt, "")
    }

    func testRatioLabelsAreShortAndHonest() {
        XCTAssertEqual(ImageSize.square.ratioLabel, "1:1")
        XCTAssertEqual(ImageSize(width: 1152, height: 896).ratioLabel, "9:7")
        XCTAssertEqual(ImageSize(width: 1344, height: 768).ratioLabel, "7:4")
        XCTAssertEqual(ImageSize(width: 1280, height: 832).ratioLabel, "≈3:2")
        XCTAssertEqual(ImageSize(width: 832, height: 1280).ratioLabel, "≈2:3")
        XCTAssertEqual(ImageSize(width: 2048, height: 512).ratioLabel, "4:1")
    }

    func testSavedSettingsOfIris02DropOnlyTheOldStepDefault() throws {
        let defaults = try scratchDefaults()
        var old = GenerationSettings()
        old.steps = 100
        old.cfgScale = 4
        defaults.set(try JSONEncoder().encode(old), forKey: "generationSettings")
        let migrated = GenerationSettings.load(from: defaults)
        XCTAssertEqual(migrated.steps, 20)
        XCTAssertEqual(migrated.cfgScale, 4)

        old.steps = 40  // a deliberate choice survives
        defaults.set(try JSONEncoder().encode(old), forKey: "generationSettings")
        XCTAssertEqual(GenerationSettings.load(from: defaults).steps, 40)

        var current = GenerationSettings()
        current.steps = 100  // chosen in this version: kept
        current.save(to: defaults)
        XCTAssertEqual(GenerationSettings.load(from: defaults).steps, 100)
    }

    func testProgressReportsOnlyWhatTheEngineSays() {
        typealias P = Generator.Progress
        XCTAssertEqual(P(.queued, step: 0, steps: 0, workflow: .create).phase, .waiting)
        XCTAssertEqual(P(.running, step: 0, steps: 0, workflow: .create).phase, .preparing)
        XCTAssertEqual(P(.running, step: 0, steps: 20, workflow: .create).phase, .preparing)
        let sampling = P(.running, step: 13, steps: 20, workflow: .create)
        XCTAssertEqual(sampling.phase, .sampling)
        XCTAssertEqual(sampling.percent, 65)
        XCTAssertEqual(P(.running, step: 20, steps: 20, workflow: .create).phase, .finishing)
        let upscale = P(.running, step: 0, steps: 0, workflow: .upscale)
        XCTAssertEqual(upscale.phase, .working)
        XCTAssertNil(upscale.fraction)
        XCTAssertNil(P(.running, step: 0, steps: 20, workflow: .create).fraction)
    }

    func testComposerSwitchesToTheImageWorkflowWhenAnImageIsAttached() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let defaults = try scratchDefaults()
        let workspace = Workspace(library: Library(store: LibraryStore(root: root)), defaults: defaults)
        XCTAssertEqual(workspace.workflow, .create)
        workspace.attach(URL(fileURLWithPath: "/tmp/photo.jpg"))
        XCTAssertEqual(workspace.workflow, .upscale)
        XCTAssertTrue(workspace.canSubmit)
        workspace.imageWorkflow = .depth
        XCTAssertEqual(workspace.workflow, .depth)
        workspace.reuse(GenerationRequest(prompt: "fox", width: 1024, height: 1024, steps: 30, cfgScale: 3, seed: 1, negativePrompt: ""))
        XCTAssertEqual(workspace.workflow, .create)
        XCTAssertEqual(workspace.prompt, "fox")
    }

    func testRecentChatsAreGroupedByDay() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "UTC"))
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 10, day: 11, hour: 9)))
        func chat(_ daysAgo: Int) -> Conversation {
            var conversation = Conversation(title: "\(daysAgo)", projectID: nil)
            conversation.updated = now.addingTimeInterval(-Double(daysAgo) * 86_400)
            return conversation
        }
        let groups = RecentGroup.group([chat(40), chat(0), chat(1), chat(3), chat(12), chat(0)], now: now, calendar: calendar)
        XCTAssertEqual(groups.map(\.title), ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "September"])
        XCTAssertEqual(groups.map(\.conversations.count), [2, 1, 1, 1, 1])
    }
}
