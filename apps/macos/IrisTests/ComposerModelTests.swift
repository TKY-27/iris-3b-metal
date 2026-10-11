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

    /// The sizes upstream `fit_budget` then 4x gives (computed with it in Python).
    func testUpscalingPredictsTheEnginesOutputSize() {
        let cases: [(CGSize, CGSize)] = [
            (CGSize(width: 1024, height: 1024), CGSize(width: 2048, height: 2048)),
            (CGSize(width: 1280, height: 768), CGSize(width: 3412, height: 2048)),
            (CGSize(width: 256, height: 256), CGSize(width: 1024, height: 1024)),
            (CGSize(width: 4032, height: 3024), CGSize(width: 2732, height: 2048)),
            (CGSize(width: 853, height: 1280), CGSize(width: 2048, height: 3072)),
            (CGSize(width: 1023, height: 511), CGSize(width: 4092, height: 2044)),
            (CGSize(width: 1025, height: 1024), CGSize(width: 2048, height: 2048)),  // 512.5 rounds to even
        ]
        for (input, output) in cases {
            XCTAssertEqual(Upscaling.output(for: input), output, "\(input)")
        }
        XCTAssertTrue(Upscaling.accepts(CGSize(width: 1024, height: 256)))
        XCTAssertFalse(Upscaling.accepts(CGSize(width: 1025, height: 256)))
        XCTAssertFalse(Upscaling.accepts(.zero))
    }

    /// Offered only when the engine takes the image and the result comes out larger.
    func testUpscalingIsRefusedWhenItWouldNotEnlarge() {
        XCTAssertNil(Upscaling.refusal(for: CGSize(width: 1024, height: 1024)))
        XCTAssertNil(Upscaling.refusal(for: CGSize(width: 384, height: 384)))
        XCTAssertNotNil(Upscaling.refusal(for: CGSize(width: 2048, height: 2048)))  // an upscale
        XCTAssertNotNil(Upscaling.refusal(for: CGSize(width: 4032, height: 3024)))  // a 12 MP photo
        XCTAssertNotNil(Upscaling.refusal(for: CGSize(width: 2000, height: 400)))  // 5:1
    }

    func testFileNamesSayWhatWasDoneOnce() {
        let created = Entry(workflow: .create, request: GenerationRequest(
            prompt: "A red fox in the snow", width: 1024, height: 1024, steps: 20, cfgScale: 3, seed: 1, negativePrompt: ""
        ))
        XCTAssertEqual(created.fileName, "a-red-fox-in-the-snow.png")
        let upscale = Entry(workflow: .upscale, inputName: created.fileName)
        XCTAssertEqual(upscale.fileName, "a-red-fox-in-the-snow-upscaled.png")
        XCTAssertEqual(Entry(workflow: .upscale, inputName: upscale.fileName).fileName, "a-red-fox-in-the-snow-upscaled.png")
        XCTAssertEqual(Entry(workflow: .depth, inputName: upscale.fileName).fileName, "a-red-fox-in-the-snow-depth.png")
        XCTAssertEqual(Entry(workflow: .depth, inputName: "IMG_0042.HEIC").fileName, "img-0042-depth.png")
        let prompt = Entry(workflow: .create, request: GenerationRequest(
            prompt: "Ocean depth", width: 1024, height: 1024, steps: 20, cfgScale: 3, seed: 1, negativePrompt: ""
        ))
        XCTAssertEqual(prompt.fileName, "ocean-depth.png")
    }

    func testPresetsNameTheirNumbersAndFindTheNearest() {
        XCTAssertEqual(QualityPreset.allCases.map(\.steps), [4, 10, 20, 30, 50])
        XCTAssertEqual(GenerationSettings().steps, QualityPreset.standard.steps)
        XCTAssertEqual(GenerationSettings().cfgScale, 3)
        XCTAssertNil(QualityPreset(steps: 100))
        XCTAssertEqual(QualityPreset.nearest(to: 100), .maximum)
        XCTAssertEqual(QualityPreset.nearest(to: 12), .fast)
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
        // an upscale counts tiles, reported before each one
        XCTAssertEqual(P(.running, step: 0, steps: 0, workflow: .upscale).phase, .preparing)
        let tiles = P(.running, step: 4, steps: 9, workflow: .upscale)
        XCTAssertEqual(tiles.phase, .sampling)
        XCTAssertEqual(tiles.percent, 44)
        XCTAssertEqual(P(.running, step: 0, steps: 9, workflow: .upscale).fraction, 0)
        let depth = P(.running, step: 0, steps: 0, workflow: .depth)
        XCTAssertEqual(depth.phase, .working)
        XCTAssertNil(depth.fraction)
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

    /// The engine refuses to upscale an image wider than 4:1, so the composer does not send one;
    /// depth takes any shape.
    func testATooNarrowImageCannotBeUpscaled() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = Workspace(library: Library(store: LibraryStore(root: root)), defaults: try scratchDefaults())
        workspace.attachment = Workspace.Attachment(
            url: URL(fileURLWithPath: "/tmp/strip.png"), name: "strip.png", pixelSize: CGSize(width: 2000, height: 400)
        )
        workspace.imageWorkflow = .upscale
        XCTAssertFalse(workspace.canSubmit)
        workspace.imageWorkflow = .depth
        XCTAssertTrue(workspace.canSubmit)
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
