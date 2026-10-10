import Foundation

/// Parameters of one text-to-image request, as `iris-metal serve` accepts them.
struct GenerationRequest: Codable, Hashable, Sendable {
    var prompt: String
    var width: Int
    var height: Int
    var steps: Int
    var cfgScale: Double
    var seed: Int
    var negativePrompt: String

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed
        case cfgScale = "cfg_scale"
        case negativePrompt = "negative_prompt"
    }
}

struct JobStatus: Decodable, Sendable {
    enum State: String, Decodable, Sendable {
        case queued, running, done, failed, cancelled
    }

    let id: String
    let status: State
    let step: Int
    let steps: Int
    let seconds: Double?
    let error: String?
}

struct EngineHealth: Decodable, Sendable {
    /// "text-to-image", "depth", "restoration" or nil
    let resident: String?
    let queued: Int
}

enum EngineError: LocalizedError {
    case unreachable
    case rejected(status: Int, message: String)

    var errorDescription: String? {
        switch self {
        case .unreachable:
            "The engine is not running."
        case let .rejected(status, message):
            status == 429 ? "The engine is busy; try again when the current image finishes." : message
        }
    }
}

/// Thin async client for the local HTTP API in `src/iris_metal/server.py`.
struct EngineClient: Sendable {
    let baseURL: URL

    init(port: Int) {
        baseURL = URL(string: "http://127.0.0.1:\(port)/v1/")!
    }

    func health() async throws -> EngineHealth {
        try await decode(get("health"))
    }

    func submit(_ request: GenerationRequest) async throws -> String {
        var urlRequest = URLRequest(url: baseURL.appending(path: "images"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        struct Submitted: Decodable { let id: String }
        let submitted: Submitted = try await decode(send(urlRequest))
        return submitted.id
    }

    /// Depth or upscaling of an image, sent as PNG.
    func submit(_ workflow: Workflow, png: Data) async throws -> String {
        let endpoint = workflow == .depth ? "depth" : "upscale"
        var request = URLRequest(url: baseURL.appending(path: endpoint))
        request.httpMethod = "POST"
        request.httpBody = png
        struct Submitted: Decodable { let id: String }
        let submitted: Submitted = try decode(await send(request))
        return submitted.id
    }

    func status(of job: String) async throws -> JobStatus {
        try await decode(get("jobs/\(job)"))
    }

    func image(of job: String) async throws -> Data {
        try await get("jobs/\(job)/image")
    }

    func cancel(_ job: String) async throws {
        var request = URLRequest(url: baseURL.appending(path: "jobs/\(job)"))
        request.httpMethod = "DELETE"
        _ = try await send(request)
    }

    private func get(_ path: String) async throws -> Data {
        try await send(URLRequest(url: baseURL.appending(path: path)))
    }

    private func send(_ request: URLRequest) async throws -> Data {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw EngineError.unreachable
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            struct Failure: Decodable { let error: String }
            let message = (try? JSONDecoder().decode(Failure.self, from: data))?.error ?? "HTTP \(status)"
            throw EngineError.rejected(status: status, message: message)
        }
        return data
    }

    private func decode<T: Decodable>(_ data: Data) throws -> T {
        try JSONDecoder().decode(T.self, from: data)
    }
}
