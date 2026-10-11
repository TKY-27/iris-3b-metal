"""A small local HTTP API over the engine (``iris-metal serve``).

Endpoints (JSON unless noted):

    POST   /v1/images              {"prompt", "width", "height", "steps", "cfg_scale",
                                    "seed", "negative_prompt"} -> 202 {"id"}
    POST   /v1/depth?max_side=1024 image file as the body -> 202 {"id"}
    POST   /v1/upscale?budget=1&color_fix=1
                                   image file as the body -> 202 {"id"}
    GET    /v1/jobs/<id>           status, progress (steps; tiles of an upscale), timing, error
    GET    /v1/jobs/<id>/image     the result PNG (depth: 16-bit, near = bright)
    DELETE /v1/jobs/<id>           cancel a queued job, a generation at its next step or
                                   an upscale at its next tile (depth runs in one pass)
    GET    /v1/health              the resident model and queue depth

A full queue answers 429; cancelling a queued job frees its place at once.
One worker thread runs the jobs in order, and at most one network is resident:
a request for a different task releases the previous model first, which keeps
a 24 GB Mac out of swap. The HTTP layer uses only the standard library. It binds to
localhost by default and has no authentication: do not expose it to a network.
Any web page the user opens can also reach localhost, so requests that come
from a browser (they carry ``Origin`` or a cross-site ``Sec-Fetch-Site``) or
name a foreign host (DNS rebinding) are refused with 403. Bound to a wildcard
address, the server cannot tell its own names apart and skips the host check.
"""

import io
import itertools
import json
import math
import queue
import re
import socket
import socketserver
import threading
import time
from collections import deque
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
# without the budget, inputs may cover what the budget would leave (512 x 1024; 2048 x 4096 at 4x)
MAX_UNBUDGETED_UPSCALE_PIXELS = 512 * 1024
KEEP_FINISHED = 32
REQUEST_TIMEOUT_S = 120  # a stalled client cannot hold a handler thread forever
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    # runs on the worker thread and returns the PNG; dropped afterwards with the input it holds
    work: Callable[["Job"], bytes] | None
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
            release_memory()  # also after a different task model, which unload() does not cover
            self._task = (name, self._load_task(name))
        return self._task[1]


def _field(body: dict, key: str, default, kinds: tuple[type, ...]):
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise ValueError(f"{key} must be {'an integer' if kinds == (int,) else 'a number'}")
    return value


def _number(body: dict, key: str, default: float) -> float:
    try:
        return float(_field(body, key, default, (int, float)))
    except OverflowError:  # an integer too large for a float
        raise ValueError(f"{key} must be a finite number") from None


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
        "cfg_scale": _number(body, "cfg_scale", defaults.cfg_scale),
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


def _flag(query: dict, key: str) -> bool:
    value = query.get(key, "1")
    if value not in ("0", "1", "true", "false"):
        raise ValueError(f"{key} must be 0, 1, true or false")
    return value in ("1", "true")


def depth_params(query: dict) -> dict:
    max_side = int(query.get("max_side", 1024))
    if not 16 <= max_side <= MAX_DEPTH_SIDE:
        raise ValueError(f"max_side must be in [16, {MAX_DEPTH_SIDE}]")
    return {"max_side": max_side}


def upscale_params(query: dict, image: Image.Image) -> dict:
    check_upscale_input(image)
    params = {"budget": _flag(query, "budget"), "color_fix": _flag(query, "color_fix")}
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
        self.max_queued = max_queued
        self.jobs: dict[str, Job] = {}
        self._pending: deque[Job] = deque()
        self.lock = threading.Condition()
        self.intake = threading.BoundedSemaphore(2)
        self._ids = itertools.count(1)
        threading.Thread(target=self._work, name="iris-worker", daemon=True).start()

    @property
    def queued(self) -> int:
        with self.lock:
            return len(self._pending)

    @property
    def full(self) -> bool:
        return self.queued >= self.max_queued

    def submit(self, kind: str, params: dict, work: Callable[[Job], bytes]) -> Job:
        """Queues a job; raises queue.Full when ``max_queued`` jobs are already waiting."""
        job = Job(id=str(next(self._ids)), kind=kind, params=params, work=work)
        with self.lock:
            if self.queued >= self.max_queued:
                raise queue.Full
            self.jobs[job.id] = job
            finished = [j for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")]
            for old in sorted(finished, key=lambda j: j.created)[:-KEEP_FINISHED]:
                del self.jobs[old.id]
            self._pending.append(job)
            self.lock.notify()
        return job

    def cancel(self, job: Job) -> None:
        with self.lock:  # a queued job is cancelled at once, not when the worker reaches it
            job.cancel.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.work = None
                self._pending.remove(job)

    def _work(self) -> None:
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self._pending)
                job = self._pending.popleft()
                job.status = "running"
            start = time.perf_counter()
            result = None
            try:
                job.raise_if_cancelled()
                result, status = job.work(job), "done"
            except Cancelled:
                status = "cancelled"
            except Exception as exc:  # reported to the client; the worker keeps serving
                status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            finally:
                job.work = None
            with self.lock:
                job.seconds = time.perf_counter() - start
                job.png = None if job.cancel.is_set() else result
                job.status = "cancelled" if job.cancel.is_set() else status


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
        source = fit_budget(image) if budget else image
        tiles, total = itertools.count(), restorer.tile_count(source.size)
        # progress in finished tiles, reported before each; a cancelled upscale stops there
        restorer.on_tile = lambda: job.on_step(next(tiles), total)
        try:
            result = restorer(source, color_fix=color_fix)
        finally:
            restorer.on_tile = None
        job.step = total
        return png(result)

    return work


JOB_PATH = re.compile(r"/v1/jobs/([0-9]+)(/image)?")
POST_LIMITS = {"/v1/images": MAX_JSON_BYTES, "/v1/depth": MAX_UPLOAD_BYTES, "/v1/upscale": MAX_UPLOAD_BYTES}


def make_handler(runner: JobRunner):
    models = runner.models

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S

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
            # a page's fetch carries Origin; its <img> or form carries a cross-site Sec-Fetch-Site
            from_page = "Origin" in self.headers or self.headers.get("Sec-Fetch-Site", "none") != "none"
            if from_page or not host_ok:
                self._send(403, {"error": "requests from web pages or for other host names are refused"})
                return True
            return False

        def _route(self, method) -> None:
            if self._refused():
                return
            try:
                method(urlsplit(self.path))
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client left; nobody to answer
            except Exception as exc:  # a response rather than a dropped connection
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _job(self, path: str, image: bool) -> Job | None:
            match = JOB_PATH.fullmatch(path)
            job = runner.jobs.get(match[1]) if match and bool(match[2]) == image else None
            if job is None:
                self._send(404, {"error": "no such job" if match else "unknown endpoint"})
            return job

        def do_GET(self) -> None:
            self._route(self._get)

        def do_POST(self) -> None:
            self._route(self._post)

        def do_DELETE(self) -> None:
            self._route(self._delete)

        def _get(self, url) -> None:
            if url.path == "/v1/health":
                self._send(200, {"resident": models.resident, "queued": runner.queued})
                return
            image = url.path.endswith("/image")
            job = self._job(url.path, image)
            if job is None:
                return
            if not image:
                self._send(200, job.summary())
            elif job.png is None:
                self._send(409, {"error": f"job is {job.status}"})
            else:
                self._send(200, job.png, "image/png")

        def _post(self, url) -> None:
            limit = POST_LIMITS.get(url.path)
            if limit is None:
                self._send(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = -1
            if length < 0:
                self._send(400, {"error": "invalid Content-Length"})
                return
            if length > limit:
                self._send(413, {"error": "request too large"})
                return
            if runner.full:  # before reading and decoding a large upload
                self._send(429, {"error": "queue is full"})
                return
            if not runner.intake.acquire(blocking=False):
                self._send(429, {"error": "request intake is busy"})
                return
            try:
                self._submit(url, length)
            finally:
                runner.intake.release()

        def _submit(self, url, length: int) -> None:
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            try:
                body = self.rfile.read(length)
                if url.path == "/v1/images":
                    params = validate(json.loads(body or b"{}"), models.pipe.cfg.sample)
                    job = runner.submit("image", params, text_to_image_job(models, params))
                elif url.path == "/v1/depth":
                    params = depth_params(query)
                    job = runner.submit("depth", params, depth_job(models, decode_upload(body), **params))
                else:
                    image = decode_upload(body)
                    params = upscale_params(query, image)
                    job = runner.submit("upscale", params, upscale_job(models, image, **params))
            except (ValueError, TypeError) as exc:  # json.JSONDecodeError is a ValueError
                self._send(400, {"error": str(exc)})
                return
            except queue.Full:
                self._send(429, {"error": "queue is full"})
                return
            self._send(202, {"id": job.id})

        def _delete(self, url) -> None:
            job = self._job(url.path, image=False)
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


class LocalServer6(LocalServer):
    address_family = socket.AF_INET6


def serve(models: Models, host: str = "127.0.0.1", port: int = 8765, max_queued: int = 8) -> None:
    server_type = LocalServer6 if ":" in host else LocalServer
    server = server_type((host, port), make_handler(JobRunner(models, max_queued)))
    print(f"iris-metal serving on http://{f'[{host}]' if ':' in host else host}:{port}", flush=True)
    server.serve_forever()
