import Foundation
import Observation

enum UpdateFailure: LocalizedError {
    case invalidMetadata
    case http(Int)

    var errorDescription: String? {
        switch self {
        case .invalidMetadata: String(localized: "GitHub returned invalid release information. Try again later.")
        case let .http(code): String(localized: "GitHub could not complete the check (HTTP \(code)). Try again later.")
        }
    }
}

@MainActor
@Observable
final class UpdateChecker {
    static let endpoint = URL(string: "https://api.github.com/repos/TKY-27/iris-3b-metal/releases?per_page=100")!
    static let installedVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? String(localized: "Unknown")
    static let installedBuild = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? String(localized: "Unknown")
    static let cacheKey = "releaseCheckCache"
    static let automaticKey = "automaticallyCheckForUpdates"
    static let day: TimeInterval = 86_400

    typealias Fetch = @Sendable (URLRequest) async throws -> (Data, HTTPURLResponse)

    let appVersion: String
    private(set) var available: AppRelease?
    private(set) var isChecking = false
    private(set) var lastChecked: Date?
    private(set) var message: String?
    private(set) var nextAttempt = Date.distantPast
    var automatic: Bool {
        didSet { defaults.set(automatic, forKey: Self.automaticKey) }
    }

    private let defaults: UserDefaults
    private let fetch: Fetch
    private let os: OperatingSystemVersion
    private let architecture: String
    private var cache: Cache?
    private var lastFailure: Date?

    private struct Cache: Codable {
        let checked: Date
        let releases: [AppRelease]
        let etag: String?
    }

    init(defaults: UserDefaults = .standard, appVersion: String = installedVersion,
         os: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion,
         architecture: String = "arm64", fetch: @escaping Fetch = UpdateChecker.fetchOfficial) {
        self.defaults = defaults
        self.appVersion = appVersion
        self.os = os
        self.architecture = architecture
        self.fetch = fetch
        automatic = defaults.object(forKey: Self.automaticKey) as? Bool ?? true
        if let data = defaults.data(forKey: Self.cacheKey), let saved = try? JSONDecoder().decode(Cache.self, from: data),
           saved.checked <= Date(), Date().timeIntervalSince(saved.checked) < 7 * Self.day {
            cache = saved
            lastChecked = saved.checked
            available = select(saved.releases)
        }
        if let retry = defaults.object(forKey: "releaseCheckRetryAfter") as? Date {
            nextAttempt = min(retry, Date().addingTimeInterval(Self.day))
        }
    }

    func check(manual: Bool = false, now: Date = Date()) async {
        if let lastChecked, now.timeIntervalSince(lastChecked) >= 7 * Self.day { available = nil }
        guard !isChecking, manual || automatic else { return }
        if now < nextAttempt {
            if manual { message = String(localized: "Please try again after \(nextAttempt.formatted(Date.FormatStyle(date: .omitted, time: .shortened).locale(.iris))).") }
            return
        }
        guard manual || lastChecked.map({ now.timeIntervalSince($0) >= Self.day }) ?? true else { return }
        guard manual || lastFailure.map({ now.timeIntervalSince($0) >= 3600 }) ?? true else { return }
        guard ReleaseVersion(appVersion) != nil else {
            message = String(localized: "This app's version cannot be compared with published releases.")
            return
        }
        isChecking = true
        message = nil
        defer { isChecking = false }
        var request = URLRequest(url: Self.endpoint, timeoutInterval: 15)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        request.setValue("Iris/\(appVersion)", forHTTPHeaderField: "User-Agent")
        if let etag = cache?.etag { request.setValue(etag, forHTTPHeaderField: "If-None-Match") }
        do {
            let (data, response) = try await fetch(request)
            try Task.checkCancellation()
            guard response.url == Self.endpoint else { throw UpdateFailure.invalidMetadata }
            if response.statusCode == 403 || response.statusCode == 429 {
                let retry = Double(response.value(forHTTPHeaderField: "Retry-After") ?? "")
                let reset = Double(response.value(forHTTPHeaderField: "X-RateLimit-Reset") ?? "")
                let proposed = retry ?? reset.map { $0 - now.timeIntervalSince1970 } ?? 3600
                let delay = proposed.isFinite ? proposed : 3600
                deferAttempt(until: now.addingTimeInterval(max(60, min(delay, Self.day))))
                message = String(localized: "GitHub's request limit was reached. Try again after \(nextAttempt.formatted(Date.FormatStyle(date: .omitted, time: .shortened).locale(.iris))).")
                return
            }
            let releases: [AppRelease]
            let etag: String?
            if response.statusCode == 304, let cache {
                releases = cache.releases
                etag = cache.etag
            } else {
                guard response.statusCode == 200 else { throw UpdateFailure.http(response.statusCode) }
                guard data.count <= 2_000_000 else { throw UpdateFailure.invalidMetadata }
                releases = try AppRelease.parse(data)
                etag = response.value(forHTTPHeaderField: "ETag")
            }
            let saved = Cache(checked: now, releases: releases, etag: etag)
            cache = saved
            defaults.set(try JSONEncoder().encode(saved), forKey: Self.cacheKey)
            lastChecked = now
            lastFailure = nil
            available = select(releases)
            deferAttempt(until: .distantPast)
        } catch is CancellationError {
            return
        } catch let error as URLError where error.code == .cancelled {
            return
        } catch {
            lastFailure = now
            message = error is URLError ? String(localized: "Could not reach GitHub. Check your connection and try again later.") : error.localizedDescription
        }
    }

    func monitor() async {
        while !Task.isCancelled {
            await check()
            do { try await Task.sleep(for: .seconds(3600)) } catch { return }
        }
    }

    private func select(_ releases: [AppRelease]) -> AppRelease? {
        guard let installed = ReleaseVersion(appVersion) else { return nil }
        return releases.filter {
            guard let version = $0.version else { return false }
            return version > installed && $0.isCompatible(os: os, architecture: architecture)
        }.max { $0.version! < $1.version! }
    }

    private func deferAttempt(until date: Date) {
        nextAttempt = date
        defaults.set(date, forKey: "releaseCheckRetryAfter")
    }

    nonisolated static func fetchOfficial(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let session = URLSession(configuration: .ephemeral, delegate: ReleaseTransport(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let (bytes, response) = try await session.bytes(for: request)
        guard let response = response as? HTTPURLResponse else { throw UpdateFailure.invalidMetadata }
        var data = Data()
        for try await byte in bytes {
            guard data.count < 2_000_000 else { throw UpdateFailure.invalidMetadata }
            data.append(byte)
        }
        return (data, response)
    }
}

private final class ReleaseTransport: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
