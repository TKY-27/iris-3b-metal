"""A small local HTTP API over one resident pipeline (``iris-metal serve``).

Endpoints (JSON unless noted):

    POST   /v1/images              {"prompt", "width", "height", "steps", "cfg_scale",
                                    "seed", "negative_prompt"} -> 202 {"id"}; 429 when full
    GET    /v1/jobs/<id>           status, progress, timing, error
    GET    /v1/jobs/<id>/image     the PNG, once the job is done
    DELETE /v1/jobs/<id>           cancel a queued or running job
    GET    /v1/health              loaded models and queue depth

One worker thread runs the jobs in order, so a single copy of the model stays
resident; cancellation takes effect at the next sampling step. Only the
standard library is used. The server binds to localhost by default and has no
authentication: do not expose it to a network.
"""

import io
import itertools
import json
import queue
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from iris_metal.pipeline import IrisPipeline

MAX_BODY_BYTES = 64 * 1024
MAX_PIXELS = 2048 * 2048
MAX_STEPS = 500
KEEP_FINISHED = 32


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    params: dict
    status: str = "queued"  # queued | running | done | failed | cancelled
    step: int = 0
    steps: int = 0
    error: str | None = None
    png: bytes | None = None
    created: float = field(default_factory=time.time)
    seconds: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "step": self.step,
            "steps": self.steps,
            "seconds": self.seconds,
            "error": self.error,
            "params": self.params,
        }


def validate(body: dict, defaults) -> dict:
    """Request JSON -> generation parameters; raises ValueError with a user-facing message."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    params = {
        "prompt": prompt,
        "width": int(body.get("width", 1024)),
        "height": int(body.get("height", 1024)),
        "steps": int(body.get("steps", defaults.steps)),
        "cfg_scale": float(body.get("cfg_scale", defaults.cfg_scale)),
        "seed": int(body.get("seed", 0)),
        "negative_prompt": str(body.get("negative_prompt", defaults.negative_prompt)),
    }
    width, height = params["width"], params["height"]
    if width < 16 or height < 16 or width % 16 or height % 16 or width * height > MAX_PIXELS:
        raise ValueError(f"width and height must be multiples of 16 with at most {MAX_PIXELS} pixels")
    if not 1 <= params["steps"] <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    return params


class JobRunner:
    def __init__(self, pipe: IrisPipeline, max_queued: int):
        self.pipe = pipe
        self.jobs: dict[str, Job] = {}
        self.pending: queue.Queue[Job] = queue.Queue(maxsize=max_queued)
        self.lock = threading.Lock()
        self._ids = itertools.count(1)
        threading.Thread(target=self._work, name="iris-worker", daemon=True).start()

    def submit(self, params: dict) -> Job:
        job = Job(id=str(next(self._ids)), params=params, steps=params["steps"])
        self.pending.put_nowait(job)  # raises queue.Full
        with self.lock:
            self.jobs[job.id] = job
            finished = [j for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")]
            for old in sorted(finished, key=lambda j: j.created)[:-KEEP_FINISHED]:
                del self.jobs[old.id]
        return job

    def _work(self) -> None:
        while True:
            job = self.pending.get()
            if job.cancel.is_set():
                job.status = "cancelled"
                continue
            job.status = "running"
            start = time.perf_counter()

            def on_step(step: int, total: int, job: Job = job) -> None:
                job.step = step
                if job.cancel.is_set():
                    raise Cancelled

            params = dict(job.params)
            try:
                image = self.pipe([params.pop("prompt")], on_step=on_step, **params)[0]
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                job.png, job.status = buffer.getvalue(), "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as exc:  # reported to the client; the worker keeps serving
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.seconds = time.perf_counter() - start


def make_handler(runner: JobRunner):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict | bytes, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _job(self) -> Job | None:
            parts = self.path.strip("/").split("/")
            job = runner.jobs.get(parts[2]) if len(parts) >= 3 and parts[:2] == ["v1", "jobs"] else None
            if job is None:
                self._send(404, {"error": "no such job"})
            return job

        def do_GET(self) -> None:
            if self.path == "/v1/health":
                self._send(200, {"model_loaded": runner.pipe.loaded, "queued": runner.pending.qsize()})
                return
            job = self._job()
            if job is None:
                return
            if self.path.endswith("/image"):
                if job.png is None:
                    self._send(409, {"error": f"job is {job.status}"})
                else:
                    self._send(200, job.png, "image/png")
                return
            self._send(200, job.summary())

        def do_POST(self) -> None:
            if self.path != "/v1/images":
                self._send(404, {"error": "unknown endpoint"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "request too large"})
                return
            try:
                params = validate(json.loads(self.rfile.read(length) or b"{}"), runner.pipe.cfg.sample)
                job = runner.submit(params)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            except queue.Full:
                self._send(429, {"error": "queue is full"})
                return
            self._send(202, {"id": job.id})

        def do_DELETE(self) -> None:
            job = self._job()
            if job is not None:
                job.cancel.set()
                self._send(200, job.summary())

        def log_message(self, format: str, *args) -> None:  # quiet by default
            pass

    return Handler


class LocalServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind resolves the host's FQDN, which can stall for many seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def serve(pipe: IrisPipeline, host: str = "127.0.0.1", port: int = 8765, max_queued: int = 8) -> None:
    server = LocalServer((host, port), make_handler(JobRunner(pipe, max_queued)))
    print(f"iris-metal serving on http://{host}:{port}", flush=True)
    server.serve_forever()
