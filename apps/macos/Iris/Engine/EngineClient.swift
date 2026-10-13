import Foundation

enum GenerationModel: String, Codable, CaseIterable, Identifiable, Sendable {
    case original, turbo

    var id: Self { self }
    var title: String { self == .turbo ? "Turbo" : "Original" }
    var fullTitle: String { "Iris-3B \(title)" }
}

/// Parameters of one text-to-image request, as `iris-metal serve` accepts them.
struct GenerationRequest: Codable, Hashable, Sendable {
    var prompt: String
    var width: Int
    var height: Int
    var steps: Int
    var cfgScale: Double
    var seed: Int
    var negativePrompt: String
    var model: GenerationModel = .original

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, model
        case cfgScale = "cfg_scale"
        case negativePrompt = "negative_prompt"
    }

    init(prompt: String, width: Int, height: Int, steps: Int, cfgScale: Double, seed: Int,
         negativePrompt: String, model: GenerationModel = .original) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.cfgScale = cfgScale
        self.seed = seed
        self.negativePrompt = negativePrompt
        self.model = model
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        prompt = try values.decode(String.self, forKey: .prompt)
        width = try values.decode(Int.self, forKey: .width)
        height = try values.decode(Int.self, forKey: .height)
        steps = try values.decode(Int.self, forKey: .steps)
        cfgScale = try values.decode(Double.self, forKey: .cfgScale)
        seed = try values.decode(Int.self, forKey: .seed)
        negativePrompt = try values.decode(String.self, forKey: .negativePrompt)
        model = try values.decodeIfPresent(GenerationModel.self, forKey: .model) ?? .original
    }

    /// Saved history keeps canonical values; Turbo's API does not accept adjustable tuning.
    func submissionData() throws -> Data {
        let encoded = try JSONEncoder().encode(self)
        guard model == .turbo else { return encoded }
        var fields = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
        for key in ["steps", "cfg_scale", "negative_prompt"] { fields.removeValue(forKey: key) }
        return try JSONSerialization.data(withJSONObject: fields)
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
    var metadata: GenerationMetadata? = nil
}

/// Engine provenance supplements the user settings in history and the PNG metadata.
struct GenerationMetadata: Codable, Hashable, Sendable {
    let sampler: String?
    let precision: String?
    let noise: String?
    let hfRevision: String?
    let upstreamRevision: String?
    let configSHA256: String?
    let distillSHA256: String?
    let checkpointSHA256: String?
    let attentionDtype: String?
    let sigmas: [Double]?

    enum CodingKeys: String, CodingKey {
        case sampler, precision, noise, sigmas
        case hfRevision = "hf_revision"
        case upstreamRevision = "upstream_revision"
        case configSHA256 = "config_sha256"
        case distillSHA256 = "distill_sha256"
        case checkpointSHA256 = "checkpoint_sha256"
        case attentionDtype = "attention_dtype"
    }

    var precisionLabel: String? {
        switch precision {
        case "bf16-linear-fp32-parameter-bf16-autocast": "BF16 / FP32"
        case "mixed-fp16-bf16-fp32": "FP16 / BF16 / FP32"
        default: precision
        }
    }
}

struct EngineHealth: Decodable, Sendable {
    /// "text-to-image", "depth", "restoration" or nil
    let resident: String?
    let queued: Int
    let engineVersion: String?
    let apiVersion: Int?
    var supportedModels: [String]? = nil
    var loadedModel: String? = nil

    static let instructionsURL = URL(string: "https://github.com/TKY-27/iris-3b-metal#quick-start")!

    enum CodingKeys: String, CodingKey {
        case resident, queued
        case engineVersion = "engine_version"
        case apiVersion = "api_version"
        case supportedModels = "supported_models"
        case loadedModel = "loaded_model"
    }

    func supports(_ model: GenerationModel) -> Bool {
        model == .original || supportedModels?.contains(model.rawValue) == true
    }

    var compatibilityIssue: String? {
        if let apiVersion, apiVersion != 1 {
            return "This engine uses API \(apiVersion); Iris requires API 1. See Engine update instructions in Settings."
        }
        if let engineVersion {
            guard let version = ReleaseVersion(engineVersion), version.major == 0, version.minor == 3 else {
                return "Engine \(engineVersion) is not supported by this app. Use iris-metal 0.3.x; see Settings."
            }
        }
        return nil
    }

    var compatibilityNotice: String? {
        compatibilityIssue ?? ((engineVersion == nil || apiVersion == nil)
            ? "This legacy engine does not report its version. Compatibility cannot be fully verified." : nil)
    }
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
        urlRequest.httpBody = try request.submissionData()
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
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as URLError where error.code == .cancelled {
            throw CancellationError()
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
