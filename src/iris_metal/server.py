"""A small local HTTP API over the engine (``iris-metal serve``).

Endpoints (JSON unless noted):

    POST   /v1/images              {"prompt", "width", "height", "steps", "cfg_scale",
                                    "seed", "negative_prompt"} -> 202 {"id"}
    POST   /v1/depth?max_side=1024 image file as the body -> 202 {"id"}
    POST   /v1/upscale?budget=1&color_fix=1
                                   image file as the body -> 202 {"id"}
    GET    /v1/jobs/<id>           status, progress, timing, error
    GET    /v1/jobs/<id>/image     the result PNG (depth: 16-bit, near = bright)
    DELETE /v1/jobs/<id>           cancel a queued job, a generation at its next step or
                                   an upscale at its next tile (depth runs in one pass)
    GET    /v1/health              the resident model and queue depth

A full queue answers 429. One worker thread runs the jobs in order, and at most
one network is resident: a request for a different task releases the previous
model first, which keeps a 24 GB Mac out of swap. Only the standard library is
used. The server binds to localhost by default and has no authentication: do
not expose it to a network. Any web page the user opens can also reach
localhost, so requests that come from a browser (they carry ``Origin``) or
name a foreign host (DNS rebinding) are refused with 403. Bound to a wildcard
address, the server cannot tell its own names apart and skips the host check.
"""

import io
import itertools
import json
import math
import queue
import socketserver
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from iris_metal.pipeline import IrisPipeline, release_memory
from iris_metal.tasks import check_upscale_input

MAX_JSON_BYTES = 64 * 1024
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_PIXELS = 50_000_000
MAX_PIXELS = 2048 * 2048
MAX_STEPS = 500
MAX_DEPTH_SIDE = 2048
# without the budget, the output area the budget allows (2048 x 4096 at 4x)
MAX_UNBUDGETED_UPSCALE_PIXELS = 512 * 1024
KEEP_FINISHED = 32
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    work: Callable[["Job"], bytes]  # runs on the worker thread, returns the PNG
    status: str = "queued"  # queued | running | done | failed | cancelled
    step: int = 0
    steps: int = 0
    error: str | None = None
    png: bytes | None = None
    created: float = field(default_factory=time.time)
    seconds: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def on_step(self, step: int, total: int) -> None:
        self.step, self.steps = step, total
        self.raise_if_cancelled()

    def raise_if_cancelled(self) -> None:
        if self.cancel.is_set():
            raise Cancelled

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "step": self.step,
            "steps": self.steps,
            "seconds": self.seconds,
            "error": self.error,
            "params": self.params,
        }


class Models:
    """At most one resident network: asking for another task releases the current one."""

    def __init__(self, pipe: IrisPipeline, load_task: Callable[[str], object]):
        self.pipe = pipe
        self._load_task = load_task
        self._task: tuple[str, object] | None = None

    @property
    def resident(self) -> str | None:
        if self._task is not None:
            return self._task[0]
        return "text-to-image" if self.pipe.loaded else None

    def text_to_image(self) -> IrisPipeline:
        if self._task is not None:
            self._task = None
            release_memory()
        return self.pipe

    def task(self, name: str):
        if self._task is None or self._task[0] != name:
            self._task = None
            self.pipe.unload()
            self._task = (name, self._load_task(name))
        return self._task[1]


def _field(body: dict, key: str, default, kinds: tuple[type, ...]):
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise ValueError(f"{key} must be {'an integer' if kinds == (int,) else 'a number'}")
    return value


def validate(body, defaults) -> dict:
    """Request JSON -> generation parameters; raises ValueError with a user-facing message."""
    if not isinstance(body, dict):
        raise ValueError("the request body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    negative_prompt = body.get("negative_prompt")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError("negative_prompt must be a string")
    params = {
        "prompt": prompt,
        "width": _field(body, "width", 1024, (int,)),
        "height": _field(body, "height", 1024, (int,)),
        "steps": _field(body, "steps", defaults.steps, (int,)),
        "cfg_scale": float(_field(body, "cfg_scale", defaults.cfg_scale, (int, float))),
        "seed": _field(body, "seed", 0, (int,)),
        "negative_prompt": defaults.negative_prompt if negative_prompt is None else negative_prompt,
    }
    width, height = params["width"], params["height"]
    if width < 16 or height < 16 or width % 16 or height % 16 or width * height > MAX_PIXELS:
        raise ValueError(f"width and height must be multiples of 16 with at most {MAX_PIXELS} pixels")
    if not 1 <= params["steps"] <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not math.isfinite(params["cfg_scale"]):
        raise ValueError("cfg_scale must be a finite number")
    if not 0 <= params["seed"] < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    return params


def depth_params(query: dict) -> dict:
    max_side = int(query.get("max_side", 1024))
    if not 16 <= max_side <= MAX_DEPTH_SIDE:
        raise ValueError(f"max_side must be in [16, {MAX_DEPTH_SIDE}]")
    return {"max_side": max_side}


def upscale_params(query: dict, image: Image.Image) -> dict:
    check_upscale_input(image)
    params = {"budget": query.get("budget", "1") != "0", "color_fix": query.get("color_fix", "1") != "0"}
    if not params["budget"] and image.width * image.height > MAX_UNBUDGETED_UPSCALE_PIXELS:
        raise ValueError(
            f"without the budget, inputs may have at most {MAX_UNBUDGETED_UPSCALE_PIXELS} pixels"
        )
    return params


def host_name(header: str) -> str:
    """``Host`` header -> the name without port or IPv6 brackets."""
    if header.startswith("["):
        return header[1 : header.find("]")]
    return header.rsplit(":", 1)[0]


def decode_upload(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_UPLOAD_PIXELS:
            raise ValueError(f"images may have at most {MAX_UPLOAD_PIXELS} pixels")
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"not a readable image: {exc}") from exc
    return image


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class JobRunner:
    def __init__(self, models: Models, max_queued: int):
        self.models = models
        self.jobs: dict[str, Job] = {}
        self.pending: queue.Queue[Job] = queue.Queue(maxsize=max_queued)
        self.lock = threading.Lock()
        self._ids = itertools.count(1)
        threading.Thread(target=self._work, name="iris-worker", daemon=True).start()

    def submit(self, kind: str, params: dict, work: Callable[[Job], bytes]) -> Job:
        job = Job(id=str(next(self._ids)), kind=kind, params=params, work=work)
        self.pending.put_nowait(job)  # raises queue.Full
        with self.lock:
            self.jobs[job.id] = job
            finished = [j for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")]
            for old in sorted(finished, key=lambda j: j.created)[:-KEEP_FINISHED]:
                del self.jobs[old.id]
        return job

    def cancel(self, job: Job) -> None:
        with self.lock:  # a queued job is cancelled at once, not when the worker reaches it
            job.cancel.set()
            if job.status == "queued":
                job.status = "cancelled"

    def _work(self) -> None:
        while True:
            job = self.pending.get()
            with self.lock:
                if job.cancel.is_set():
                    job.status = "cancelled"
                    continue
                job.status = "running"
            start = time.perf_counter()
            try:
                job.png, status = job.work(job), "done"
            except Cancelled:
                status = "cancelled"
            except Exception as exc:  # reported to the client; the worker keeps serving
                status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.seconds = time.perf_counter() - start
            job.status = status  # last, so a finished job always has its time and result


def text_to_image_job(models: Models, params: dict) -> Callable[[Job], bytes]:
    def work(job: Job) -> bytes:
        options = dict(params)
        prompt = options.pop("prompt")
        return png(models.text_to_image()([prompt], on_step=job.on_step, **options)[0])

    return work


def depth_job(models: Models, image: Image.Image, max_side: int) -> Callable[[Job], bytes]:
    def work(job: Job) -> bytes:
        from iris_metal.tasks import depth_image

        return png(depth_image(models.task("depth")(image, max_side=max_side)))

    return work


def upscale_job(models: Models, image: Image.Image, budget: bool, color_fix: bool) -> Callable[[Job], bytes]:
    def work(job: Job) -> bytes:
        from iris3b.downstream.restoration import fit_budget

        restorer = models.task("restoration")
        restorer.on_tile = job.raise_if_cancelled  # a cancelled upscale stops between tiles
        try:
            return png(restorer(fit_budget(image) if budget else image, color_fix=color_fix))
        finally:
            restorer.on_tile = None

    return work


def make_handler(runner: JobRunner):
    models = runner.models

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict | bytes, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _refused(self) -> bool:
            """Answer 403 to browser requests and foreign host names; see the module docstring."""
            bound = self.server.server_name
            allowed = LOOPBACK_HOSTS | {bound}
            host_ok = bound in WILDCARD_HOSTS or host_name(self.headers.get("Host", "")) in allowed
            if "Origin" in self.headers or not host_ok:
                self._send(403, {"error": "requests from web pages or for other host names are refused"})
                return True
            return False

        def _job(self) -> Job | None:
            parts = urlsplit(self.path).path.strip("/").split("/")
            job = runner.jobs.get(parts[2]) if len(parts) >= 3 and parts[:2] == ["v1", "jobs"] else None
            if job is None:
                self._send(404, {"error": "no such job"})
            return job

        def do_GET(self) -> None:
            if self._refused():
                return
            path = urlsplit(self.path).path
            if path == "/v1/health":
                self._send(200, {"resident": models.resident, "queued": runner.pending.qsize()})
                return
            job = self._job()
            if job is None:
                return
            if path.endswith("/image"):
                if job.png is None:
                    self._send(409, {"error": f"job is {job.status}"})
                else:
                    self._send(200, job.png, "image/png")
                return
            self._send(200, job.summary())

        def do_POST(self) -> None:
            if self._refused():
                return
            url = urlsplit(self.path)
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = -1
            if length < 0:
                self._send(400, {"error": "invalid Content-Length"})
                return
            limit = MAX_JSON_BYTES if url.path == "/v1/images" else MAX_UPLOAD_BYTES
            if length > limit:
                self._send(413, {"error": "request too large"})
                return
            try:
                body = self.rfile.read(length)
                if url.path == "/v1/images":
                    params = validate(json.loads(body or b"{}"), models.pipe.cfg.sample)
                    job = runner.submit("image", params, text_to_image_job(models, params))
                elif url.path == "/v1/depth":
                    params = depth_params(query)
                    job = runner.submit("depth", params, depth_job(models, decode_upload(body), **params))
                elif url.path == "/v1/upscale":
                    image = decode_upload(body)
                    params = upscale_params(query, image)
                    job = runner.submit("upscale", params, upscale_job(models, image, **params))
                else:
                    self._send(404, {"error": "unknown endpoint"})
                    return
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            except queue.Full:
                self._send(429, {"error": "queue is full"})
                return
            self._send(202, {"id": job.id})

        def do_DELETE(self) -> None:
            if self._refused():
                return
            job = self._job()
            if job is not None:
                runner.cancel(job)
                self._send(200, job.summary())

        def log_message(self, format: str, *args) -> None:  # quiet by default
            pass

    return Handler


class LocalServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind resolves the host's FQDN, which can stall for many seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def serve(models: Models, host: str = "127.0.0.1", port: int = 8765, max_queued: int = 8) -> None:
    server = LocalServer((host, port), make_handler(JobRunner(models, max_queued)))
    print(f"iris-metal serving on http://{host}:{port}", flush=True)
    server.serve_forever()
