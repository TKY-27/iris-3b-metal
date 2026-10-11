import XCTest
@testable import Iris

private actor ReleaseFeed {
    var requests: [URLRequest] = []
    var responses: [(Data, HTTPURLResponse)]

    init(_ responses: [(Data, HTTPURLResponse)]) { self.responses = responses }

    func fetch(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        requests.append(request)
        return responses.removeFirst()
    }
}

@MainActor
final class UpdateTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suite: String!
    private let os = OperatingSystemVersion(majorVersion: 15, minorVersion: 0, patchVersion: 0)

    override func setUp() async throws {
        suite = "IrisUpdateTests-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)!
    }

    override func tearDown() async throws { defaults.removePersistentDomain(forName: suite) }

    private func release(_ tag: String, draft: Bool = false, prerelease: Bool = false,
                         minimum: String = "15.0", architecture: String = "arm64") -> [String: Any] {
        ["tag_name": tag, "draft": draft, "prerelease": prerelease,
         "html_url": "https://attacker.invalid/download",
         "body": "A smaller, safer Iris.\n<!-- iris-macos: {\"minimumSystemVersion\":\"\(minimum)\",\"architectures\":[\"\(architecture)\"]} -->"]
    }

    private func response(_ objects: [[String: Any]] = [], status: Int = 200,
                          headers: [String: String] = [:], url: URL = UpdateChecker.endpoint) throws -> (Data, HTTPURLResponse) {
        (try JSONSerialization.data(withJSONObject: objects),
         try XCTUnwrap(HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: headers)))
    }

    private func checker(_ feed: ReleaseFeed, version: String = "0.3.0") -> UpdateChecker {
        UpdateChecker(defaults: defaults, appVersion: version, os: os, fetch: { try await feed.fetch($0) })
    }

    func testStableSemanticVersions() {
        XCTAssertLessThan(ReleaseVersion("0.3.9")!, ReleaseVersion("v0.3.10")!)
        XCTAssertLessThan(ReleaseVersion("0.99.99")!, ReleaseVersion("1.0.0")!)
        XCTAssertEqual(ReleaseVersion("v1.2.3+build.4"), ReleaseVersion("1.2.3"))
        for invalid in ["", "1.2", "01.2.3", "1.2.3-beta", "1.2.3+", "v1.2.3/path", "1.2.3\n", "1.2.3+build\n", "999999999999999999999.0.0"] {
            XCTAssertNil(ReleaseVersion(invalid), invalid)
        }
    }

    func testSelectsNewestCompatibleStableRelease() async throws {
        let feed = ReleaseFeed([try response([
            release("v0.3.2"), release("v0.3.10"), release("v9.0.0", draft: true),
            release("v8.0.0", prerelease: true), release("v7.0.0-beta"),
            release("v6.0.0", minimum: "28.0"), release("v5.0.0", architecture: "x86_64"),
            release("v4.0.0", minimum: "garbage"), ["tag_name": "v3.0.0"],
        ])])
        let updates = checker(feed)
        await updates.check(manual: true)
        XCTAssertEqual(updates.available?.tag, "v0.3.10")
        XCTAssertEqual(updates.available?.notes, "A smaller, safer Iris.")
        XCTAssertEqual(updates.available?.url.absoluteString, "https://github.com/TKY-27/iris-3b-metal/releases/tag/v0.3.10")
        XCTAssertFalse(updates.isChecking)
        XCTAssertNil(updates.message)
    }

    func testMissingCompatibilityDoesNotAdvertiseUpdate() async throws {
        let feed = ReleaseFeed([try response([["tag_name": "v1.0.0", "draft": false, "prerelease": false, "body": "Source only"]])])
        let updates = checker(feed)
        await updates.check(manual: true)
        XCTAssertNil(updates.available)
        XCTAssertNotNil(updates.lastChecked)
    }

    func testInitialSourceReleaseSupportsOlderInstalledApp() async throws {
        let feed = ReleaseFeed([try response([["tag_name": "v0.3.0", "draft": false, "prerelease": false, "body": "Initial source release"]])])
        let updates = checker(feed, version: "0.2.0")
        await updates.check(manual: true)
        XCTAssertEqual(updates.available?.tag, "v0.3.0")
        XCTAssertEqual(updates.available?.minimumSystemVersion, "15.0")
    }

    func testNoUpdateAndOlderRelease() async throws {
        let updates = checker(ReleaseFeed([try response([release("v0.3.0"), release("v0.2.9")])]))
        await updates.check(manual: true)
        XCTAssertNil(updates.available)
        XCTAssertNil(updates.message)
    }

    func testCacheConditionalRequestAndNewAppVersion() async throws {
        let feed = ReleaseFeed([try response([release("v0.3.1")], headers: ["ETag": "test-etag"]), try response(status: 304)])
        let updates = checker(feed)
        let now = Date()
        await updates.check(now: now)
        await updates.check(now: now.addingTimeInterval(100))
        let firstRequests = await feed.requests
        XCTAssertEqual(firstRequests.count, 1)
        await updates.check(now: now.addingTimeInterval(UpdateChecker.day))
        let requests = await feed.requests
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests.last?.value(forHTTPHeaderField: "If-None-Match"), "test-etag")
        XCTAssertEqual(updates.available?.tag, "v0.3.1")
        let upgraded = checker(ReleaseFeed([]), version: "0.3.1")
        XCTAssertNil(upgraded.available)
    }

    func testOfflineRetainsRecentCacheWithoutFalseNoUpdate() async throws {
        let updates = checker(ReleaseFeed([try response([release("v0.3.1")])]))
        await updates.check(manual: true)
        let offline = UpdateChecker(defaults: defaults, appVersion: "0.3.0", os: os,
                                    fetch: { _ in throw URLError(.notConnectedToInternet) })
        await offline.check(manual: true)
        XCTAssertEqual(offline.available?.tag, "v0.3.1")
        XCTAssertNotNil(offline.message)
        XCTAssertFalse(offline.isChecking)
    }

    func testInvalidJSONDoesNotEraseCache() async throws {
        let bad = (Data("not JSON".utf8), try response().1)
        let feed = ReleaseFeed([try response([release("v0.3.1")]), bad])
        let updates = checker(feed)
        await updates.check(manual: true)
        await updates.check(manual: true)
        XCTAssertNotNil(updates.message)
        XCTAssertEqual(updates.available?.tag, "v0.3.1")
    }

    func testRateLimitSurvivesRestartAndManualChecks() async throws {
        let now = Date()
        let feed = ReleaseFeed([try response(status: 429, headers: ["Retry-After": "120"])])
        let updates = checker(feed)
        await updates.check(manual: true, now: now)
        XCTAssertEqual(updates.nextAttempt, now.addingTimeInterval(120))
        let restarted = checker(feed)
        await restarted.check(manual: true, now: now.addingTimeInterval(20))
        let requests = await feed.requests
        XCTAssertEqual(requests.count, 1)
        XCTAssertNotNil(restarted.message)
    }

    func testAutomaticOptOutStillAllowsManualCheck() async throws {
        let feed = ReleaseFeed([try response()])
        let updates = checker(feed)
        updates.automatic = false
        await updates.check()
        let before = await feed.requests
        XCTAssertTrue(before.isEmpty)
        await updates.check(manual: true)
        let after = await feed.requests
        XCTAssertEqual(after.count, 1)
        XCTAssertFalse(checker(ReleaseFeed([])).automatic)
    }

    func testHTTPFailureAndUnexpectedOrigin() async throws {
        for response in [try response(status: 500), try response([release("v0.3.1")], url: URL(string: "https://attacker.invalid/")!)] {
            let isolatedSuite = "\(suite!)-\(UUID().uuidString)"
            let isolated = UserDefaults(suiteName: isolatedSuite)!
            defer { isolated.removePersistentDomain(forName: isolatedSuite) }
            let feed = ReleaseFeed([response])
            let updates = UpdateChecker(defaults: isolated, appVersion: "0.3.0", os: os, fetch: { try await feed.fetch($0) })
            await updates.check(manual: true)
            XCTAssertNil(updates.available)
            XCTAssertNotNil(updates.message)
        }
    }

    func testCoalescesChecksAndCancellationIsQuiet() async throws {
        let reply = try response()
        let updates = UpdateChecker(defaults: defaults, appVersion: "0.3.0", fetch: { _ in
            try await Task.sleep(for: .seconds(10))
            return reply
        })
        let task = Task { await updates.check(manual: true) }
        await Task.yield()
        XCTAssertTrue(updates.isChecking)
        await updates.check(manual: true)
        task.cancel()
        await task.value
        XCTAssertFalse(updates.isChecking)
        XCTAssertNil(updates.message)
        XCTAssertNil(updates.lastChecked)
    }

    func testUpdateCheckPreservesUnrelatedSettings() async throws {
        defaults.set("/user/managed/iris-metal", forKey: "enginePath")
        defaults.set(42, forKey: "seed")
        let updates = checker(ReleaseFeed([try response([release("v0.3.1")])]))
        await updates.check(manual: true)
        XCTAssertEqual(defaults.string(forKey: "enginePath"), "/user/managed/iris-metal")
        XCTAssertEqual(defaults.integer(forKey: "seed"), 42)
    }

    func testExpiredCacheIsHiddenDuringRateLimit() async throws {
        let feed = ReleaseFeed([try response([release("v0.3.1")]), try response(status: 429)])
        let updates = checker(feed)
        let now = Date()
        await updates.check(manual: true, now: now)
        XCTAssertNotNil(updates.available)
        await updates.check(manual: true, now: now.addingTimeInterval(8 * UpdateChecker.day))
        XCTAssertNil(updates.available)
        await updates.check(manual: true, now: now.addingTimeInterval(8 * UpdateChecker.day + 10))
        XCTAssertNil(updates.available)
    }

    func testNotModifiedWithoutCacheIsAnError() async throws {
        let updates = checker(ReleaseFeed([try response(status: 304)]))
        await updates.check(manual: true)
        XCTAssertNotNil(updates.message)
        XCTAssertNil(updates.lastChecked)
    }

    func testInvalidAppVersionNeverChecksNetwork() async {
        let updates = checker(ReleaseFeed([]), version: "Unknown")
        await updates.check(manual: true)
        XCTAssertNotNil(updates.message)
        XCTAssertNil(updates.available)
    }

    func testEngineCompatibilityAndLegacyRelease() async throws {
        func health(_ metadata: String) throws -> EngineHealth {
            try JSONDecoder().decode(EngineHealth.self, from: Data("{\"resident\":null,\"queued\":0\(metadata)}".utf8))
        }
        let legacy = try health("")
        XCTAssertNil(legacy.compatibilityIssue)
        XCTAssertNotNil(legacy.compatibilityNotice)
        let supported = try health(",\"engine_version\":\"0.3.0\",\"api_version\":1")
        XCTAssertNil(supported.compatibilityNotice)
        for metadata in [",\"engine_version\":\"0.4.0\",\"api_version\":1", ",\"engine_version\":\"0.3.0\",\"api_version\":2", ",\"engine_version\":\"unknown\""] {
            let incompatible = try health(metadata)
            let engine = EngineController(defaults: defaults, healthCheck: { _ in incompatible })
            await engine.start()
            guard case .failed = engine.state else { return XCTFail("Incompatible engine accepted") }
            XCTAssertNotNil(engine.health?.compatibilityIssue)
        }
        let engine = EngineController(defaults: defaults, healthCheck: { _ in supported })
        await engine.start()
        XCTAssertEqual(engine.state, .ready)
        engine.stop()
        XCTAssertNil(engine.health)
    }
}
