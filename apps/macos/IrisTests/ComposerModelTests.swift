import XCTest

@testable import Iris

@MainActor
final class ComposerModelTests: XCTestCase {
    func testTurboKeepsOriginalTuningAndUsesItsOwnFixedRequest() throws {
        var settings = GenerationSettings()
        settings.steps = 48
        settings.cfgScale = 4.25
        settings.negativePrompt = "blur"
        settings.randomizeSeed = false
        settings.seed = 42
        settings.model = .turbo
        let turbo = settings.request(for: "a fox")
        XCTAssertEqual(turbo.model, .turbo)
        XCTAssertEqual(turbo.steps, 4)
        XCTAssertEqual(turbo.cfgScale, 1)
        XCTAssertEqual(turbo.negativePrompt, "")
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: turbo.submissionData()) as? [String: Any])
        XCTAssertEqual(payload["model"] as? String, "turbo")
        XCTAssertNil(payload["steps"])
        XCTAssertNil(payload["cfg_scale"])
        XCTAssertNil(payload["negative_prompt"])
        settings.reuse(turbo)
        settings.model = .original
        let original = settings.request(for: "a fox")
        XCTAssertEqual(original.steps, 48)
        XCTAssertEqual(original.cfgScale, 4.25)
        XCTAssertEqual(original.negativePrompt, "blur")
        XCTAssertEqual(original.seed, 42)
        settings.model = .turbo
        settings.restoreDefaults()
        XCTAssertTrue(settings.randomizeSeed)
        XCTAssertEqual(settings.steps, 48)
        XCTAssertEqual(settings.cfgScale, 4.25)
        XCTAssertEqual(settings.negativePrompt, "blur")
    }

    func testExactAspectPresetsKeepTheirDimensionsAcrossRequestsAndReuse() throws {
        let defaults = try scratchDefaults()
        for (size, width, height) in [(ImageSize.landscape16x9, 1280, 720), (.portrait9x16, 720, 1280)] {
            for model in [GenerationModel.original, .turbo] {
                var settings = GenerationSettings(size: size, model: model)
                let request = settings.request(for: "a fox")
                XCTAssertEqual([request.width, request.height], [width, height])

                let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: request.submissionData()) as? [String: Any])
                XCTAssertEqual(payload["width"] as? Int, width)
                XCTAssertEqual(payload["height"] as? Int, height)

                var reused = GenerationSettings()
                reused.reuse(request)
                XCTAssertEqual(reused.size, size)
                settings.save(to: defaults)
                XCTAssertEqual(GenerationSettings.load(from: defaults).size, size)
            }
        }
    }

    func testOldRequestsAndPreferencesDefaultToOriginal() throws {
        let request = Data(#"{"prompt":"fox","width":1024,"height":1024,"steps":50,"cfg_scale":3,"seed":7,"negative_prompt":""}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(GenerationRequest.self, from: request).model, .original)
        let oldSettings = Data(#"{"size":{"width":1024,"height":1024},"steps":50,"cfgScale":3,"seed":7,"randomizeSeed":false,"negativePrompt":"blur"}"#.utf8)
        let settings = try JSONDecoder().decode(GenerationSettings.self, from: oldSettings)
        XCTAssertEqual(settings.model, .original)
        XCTAssertEqual(settings.steps, 50)
        XCTAssertEqual(settings.negativePrompt, "blur")
        let legacyHealth = try JSONDecoder().decode(EngineHealth.self, from: Data(#"{"resident":null,"queued":0}"#.utf8))
        XCTAssertTrue(legacyHealth.supports(.original))
        XCTAssertFalse(legacyHealth.supports(.turbo))
    }

    func testTurboHistoryRoundTripPreservesModelAndSettings() throws {
        var settings = GenerationSettings(model: .turbo)
        settings.seed = 123
        settings.randomizeSeed = false
        let request = settings.request(for: "a lighthouse")
        var entry = Entry(workflow: .create, request: request, outcome: .done(seconds: 2))
        entry.metadata = try JSONDecoder().decode(GenerationMetadata.self, from: Data(#"{"sampler":"flow-euler","precision":"bf16-linear-fp32-parameter-bf16-autocast","sigmas":[1,0.938,0.833,0.625],"checkpoint_sha256":"verified-checkpoint-hash"}"#.utf8))
        let decoded = try JSONDecoder().decode(Entry.self, from: JSONEncoder().encode(entry))
        XCTAssertEqual(decoded, entry)
        XCTAssertEqual(decoded.request?.model, .turbo)
        XCTAssertEqual(decoded.metadata?.checkpointSHA256, "verified-checkpoint-hash")
        XCTAssertEqual(decoded.metadata?.precisionLabel, "BF16 / FP32")
        XCTAssertTrue(decoded.matches("Turbo"))
        XCTAssertFalse(decoded.matches("Original"))
    }

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
        XCTAssertEqual(settings.qualityTitle, String(localized: "\(48) steps"))
        XCTAssertEqual(settings.adherenceTitle, String(localized: "Guidance \("3.25")"))
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
        XCTAssertEqual(ImageSize.landscape16x9.ratioLabel, "16:9")
        XCTAssertEqual(ImageSize.portrait9x16.ratioLabel, "9:16")
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
        let workspace = Workspace(defaults: try scratchDefaults())
        XCTAssertEqual(workspace.workflow, .create)
        workspace.attach(URL(fileURLWithPath: "/tmp/photo.jpg"))
        XCTAssertEqual(workspace.workflow, .upscale)
        XCTAssertTrue(workspace.canSubmit)
        workspace.imageWorkflow = .depth
        XCTAssertEqual(workspace.workflow, .depth)
        workspace.reuse(GenerationRequest(prompt: "fox", width: 1024, height: 1024, steps: 30, cfgScale: 3, seed: 1, negativePrompt: ""))
        XCTAssertEqual(workspace.workflow, .create)
        XCTAssertEqual(workspace.prompt, "fox")
        XCTAssertEqual(workspace.settings.seed, 1)
        XCTAssertFalse(workspace.settings.randomizeSeed)  // the same seed, to make it again
        XCTAssertEqual(workspace.promptFocusRequests, 1)
    }

    /// The engine refuses to upscale an image wider than 4:1, so the composer does not send one;
    /// depth takes any shape.
    func testATooNarrowImageCannotBeUpscaled() throws {
        let workspace = Workspace(defaults: try scratchDefaults())
        workspace.attachment = Workspace.Attachment(
            url: URL(fileURLWithPath: "/tmp/strip.png"), name: "strip.png", pixelSize: CGSize(width: 2000, height: 400)
        )
        workspace.imageWorkflow = .upscale
        XCTAssertFalse(workspace.canSubmit)
        workspace.imageWorkflow = .depth
        XCTAssertTrue(workspace.canSubmit)
    }
}
