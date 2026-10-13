import http.client
import io
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from iris_metal import __version__
from iris_metal.server import JobRunner, LocalServer, Models, host_name, make_handler, validate

DEFAULTS = SimpleNamespace(steps=100, cfg_scale=3.0, negative_prompt="")


class FakePipeline:
    """Calls on_step like the real pipeline; ``gate`` pauses a job mid-way."""

    def __init__(self):
        self.cfg = SimpleNamespace(sample=DEFAULTS)
        self.loaded = True
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, prompts, height, width, steps, on_step, **kwargs):
        for step in range(1, steps + 1):
            self.gate.wait(5)
            on_step(step, steps)
        if prompts[0] == "explode":
            raise RuntimeError("boom")
        return [Image.new("RGB", (width, height))]

    def unload(self):
        self.loaded = False


class FakeDepth:
    def __call__(self, image, max_side):
        return np.linspace(-1, 1, image.width * image.height).reshape(image.height, image.width)


class FakeRestorer:
    """Three tiles, calling on_tile before each like the real restorer."""

    on_tile = None
    tiles_run = 0

    def tile_count(self, size):
        return 3

    def __call__(self, image, color_fix=True):
        for _ in range(3):
            self.on_tile()
            self.tiles_run += 1
        return image.resize((image.width * 4, image.height * 4))


@pytest.fixture
def server():
    pipe = FakePipeline()
    pipe.restorer = FakeRestorer()
    models = Models(pipe, load_task=lambda task: FakeDepth() if task == "depth" else pipe.restorer)
    pipe.runner = JobRunner(models, max_queued=2)
    httpd = LocalServer(("127.0.0.1", 0), make_handler(pipe.runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield pipe, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def request(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_for(base, job_id, status):
    for _ in range(200):
        _, body = request(f"{base}/v1/jobs/{job_id}")
        if json.loads(body)["status"] == status:
            return json.loads(body)
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}")


def test_validate_rejects_bad_sizes_and_prompts():
    with pytest.raises(ValueError, match="multiples of 16"):
        validate({"prompt": "x", "width": 1000}, DEFAULTS)
    with pytest.raises(ValueError, match="prompt"):
        validate({"prompt": "  "}, DEFAULTS)
    assert validate({"prompt": "x"}, DEFAULTS)["steps"] == 100
    with pytest.raises(ValueError, match="cfg_scale"):
        validate(json.loads('{"prompt": "x", "cfg_scale": NaN}'), DEFAULTS)
    for seed in (-1, 2**63):
        with pytest.raises(ValueError, match="seed"):
            validate({"prompt": "x", "seed": seed}, DEFAULTS)


def test_health_reports_engine_and_api_versions(server):
    _, base = server
    status, body = request(f"{base}/v1/health")
    assert status == 200
    assert json.loads(body) == {
        "resident": "text-to-image",
        "queued": 0,
        "engine_version": __version__,
        "api_version": 1,
        "supported_models": ["original"],
        "loaded_model": "original",
    }


@pytest.mark.parametrize(
    "body",
    [["fox"], "fox", {"prompt": "x", "width": 1e999}, {"prompt": "x", "width": 1024.5},
     {"prompt": "x", "steps": True}, {"prompt": "x", "seed": "7"}, {"prompt": "x", "negative_prompt": 3}],
)  # fmt: skip
def test_validate_rejects_malformed_bodies(body):
    with pytest.raises(ValueError):
        validate(body, DEFAULTS)


def test_validate_treats_null_as_default():
    params = validate({"prompt": "x", "negative_prompt": None, "seed": None}, DEFAULTS)
    assert params["negative_prompt"] == DEFAULTS.negative_prompt and params["seed"] == 0


def raw(base, method, path, headers, body=b""):
    """A request with exactly these headers, which urllib would normalize."""
    connection = http.client.HTTPConnection(base.removeprefix("http://"))
    connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for name, value in headers.items():
        connection.putheader(name, value)
    connection.endheaders(body)
    response = connection.getresponse()
    return response.status, response.read()


def test_requests_from_web_pages_and_foreign_hosts_are_refused(server):
    _, base = server
    host = base.removeprefix("http://")
    assert raw(base, "GET", "/v1/health", {"Host": host})[0] == 200
    assert raw(base, "GET", "/v1/health", {"Host": host.replace("127.0.0.1", "localhost")})[0] == 200
    # DNS rebinding: a page on evil.example whose name now resolves to 127.0.0.1
    assert raw(base, "GET", "/v1/health", {"Host": "evil.example:80"})[0] == 403
    body = json.dumps({"prompt": "fox", "width": 16, "height": 16, "steps": 1}).encode()
    page = {"Host": host, "Origin": "https://evil.example", "Content-Length": str(len(body))}
    assert raw(base, "POST", "/v1/images", page, body)[0] == 403
    assert raw(base, "DELETE", "/v1/jobs/1", {"Host": host, "Origin": "null"})[0] == 403
    assert [host_name(h) for h in ("[::1]:8765", "localhost", "127.0.0.1:1")] == [
        "::1",
        "localhost",
        "127.0.0.1",
    ]


def test_invalid_content_length_is_rejected(server):
    _, base = server
    host = base.removeprefix("http://")
    for length in ("-1", "many"):
        assert raw(base, "POST", "/v1/images", {"Host": host, "Content-Length": length})[0] == 400


def test_generate_poll_and_fetch_png(server):
    _, base = server
    status, body = request(
        f"{base}/v1/images", "POST", {"prompt": "fox", "width": 32, "height": 48, "steps": 3}
    )
    assert status == 202
    job = wait_for(base, json.loads(body)["id"], "done")
    assert job["step"] == 3
    status, png = request(f"{base}/v1/jobs/{job['id']}/image")
    assert status == 200 and png.startswith(b"\x89PNG")


def test_cancel_running_job_and_report_failures(server):
    pipe, base = server
    pipe.gate.clear()
    _, body = request(f"{base}/v1/images", "POST", {"prompt": "fox", "width": 16, "height": 16, "steps": 50})
    job_id = json.loads(body)["id"]
    wait_for(base, job_id, "running")
    request(f"{base}/v1/jobs/{job_id}", "DELETE")
    pipe.gate.set()
    assert wait_for(base, job_id, "cancelled")["step"] < 50

    _, body = request(
        f"{base}/v1/images", "POST", {"prompt": "explode", "width": 16, "height": 16, "steps": 1}
    )
    assert "boom" in wait_for(base, json.loads(body)["id"], "failed")["error"]


def test_full_queue_is_rejected(server):
    pipe, base = server
    pipe.gate.clear()
    codes = [
        request(f"{base}/v1/images", "POST", {"prompt": "fox", "width": 16, "height": 16, "steps": 1})[0]
        for _ in range(4)
    ]
    pipe.gate.set()
    assert codes[-1] == 429  # one running, two queued, the fourth refused


def test_depth_upload_switches_the_resident_model(server):
    pipe, base = server
    request(f"{base}/v1/images", "POST", {"prompt": "fox", "width": 16, "height": 16, "steps": 1})
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buffer, format="JPEG")
    req = urllib.request.Request(f"{base}/v1/depth?max_side=512", data=buffer.getvalue(), method="POST")
    with urllib.request.urlopen(req) as response:
        job_id = json.loads(response.read())["id"]
    job = wait_for(base, job_id, "done")
    assert job["params"] == {"max_side": 512}
    assert not pipe.loaded  # the text-to-image model was released for the depth model
    _, health = request(f"{base}/v1/health")
    assert json.loads(health)["resident"] == "depth"
    _, depth_png = request(f"{base}/v1/jobs/{job_id}/image")
    assert Image.open(io.BytesIO(depth_png)).size == (20, 10)


def upload(base, path, size):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    req = urllib.request.Request(f"{base}{path}", data=buffer.getvalue(), method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_task_inputs_are_bounded(server):
    _, base = server
    assert upload(base, "/v1/depth?max_side=0", (16, 16)) == 400  # 0 would mean native resolution
    assert upload(base, "/v1/depth?max_side=4096", (16, 16)) == 400
    assert upload(base, "/v1/upscale?budget=0", (1024, 1024)) == 400  # 16 MP out without the budget
    assert upload(base, "/v1/upscale?budget=0", (16, 16)) == 202
    assert upload(base, "/v1/upscale", (1, 1024)) == 400  # would become a 1024 x 1048576 work area
    assert upload(base, "/v1/upscale", (64, 256)) == 202  # 4:1, the widest native ratio


def test_queued_jobs_cancel_at_once_and_queries_are_ignored(server):
    pipe, base = server
    pipe.gate.clear()
    ids = [
        json.loads(
            request(f"{base}/v1/images", "POST", {"prompt": "fox", "width": 16, "height": 16, "steps": 5})[1]
        )["id"]
        for _ in range(2)
    ]
    wait_for(base, ids[0], "running")
    request(f"{base}/v1/jobs/{ids[1]}", "DELETE")
    assert json.loads(request(f"{base}/v1/jobs/{ids[1]}")[1])["status"] == "cancelled"
    pipe.gate.set()
    wait_for(base, ids[0], "done")
    assert request(f"{base}/v1/health?verbose=1")[0] == 200
    status, png = request(f"{base}/v1/jobs/{ids[0]}/image?download=1")
    assert status == 200 and png.startswith(b"\x89PNG")


def test_unreadable_upload_is_rejected(server):
    _, base = server
    req = urllib.request.Request(f"{base}/v1/upscale", data=b"not an image", method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req)
    assert error.value.code == 400


def test_cancelling_a_queued_job_frees_its_place(server):
    pipe, base = server
    pipe.gate.clear()
    body = {"prompt": "fox", "width": 16, "height": 16, "steps": 1}
    ids = [json.loads(request(f"{base}/v1/images", "POST", body)[1])["id"] for _ in range(3)]
    wait_for(base, ids[0], "running")  # one running, two queued: full
    assert request(f"{base}/v1/images", "POST", body)[0] == 429
    request(f"{base}/v1/jobs/{ids[2]}", "DELETE")
    assert json.loads(request(f"{base}/v1/health")[1])["queued"] == 1
    assert request(f"{base}/v1/images", "POST", body)[0] == 202
    pipe.gate.set()


def test_finished_jobs_release_their_inputs(server):
    pipe, base = server
    assert upload(base, "/v1/depth", (20, 10)) == 202
    wait_for(base, "1", "done")
    job = pipe.runner.jobs["1"]
    assert job.work is None and job.png is not None  # the decoded upload went with work


def test_malformed_requests_get_answers_not_dropped_connections(server):
    _, base = server
    host = base.removeprefix("http://")
    huge = b'{"prompt": "x", "cfg_scale": 1' + b"0" * 400 + b"}"
    assert raw(base, "POST", "/v1/images", {"Host": host, "Content-Length": str(len(huge))}, huge)[0] == 400
    assert upload(base, "/v1/upscale?budget=false", (16, 16)) == 202
    assert upload(base, "/v1/upscale?budget=no", (16, 16)) == 400
    assert raw(base, "POST", "/v1/elsewhere", {"Host": host, "Content-Length": "0"})[0] == 404
    assert request(f"{base}/v1/jobs/1/image", "DELETE")[0] == 404
    assert request(f"{base}/v1/jobs/1/other")[0] == 404
    assert request(f"{base}/v1/nothing")[0] == 404


def test_cross_site_browser_requests_are_refused(server):
    _, base = server
    host = base.removeprefix("http://")
    assert raw(base, "GET", "/v1/jobs/1/image", {"Host": host, "Sec-Fetch-Site": "cross-site"})[0] == 403
    assert raw(base, "GET", "/v1/health", {"Host": host, "Sec-Fetch-Site": "none"})[0] == 200


def test_upscale_reports_its_progress_in_tiles(server):
    pipe, base = server
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="PNG")
    req = urllib.request.Request(f"{base}/v1/upscale", data=buffer.getvalue(), method="POST")
    with urllib.request.urlopen(req) as response:
        job = json.loads(response.read())["id"]
    done = wait_for(base, job, "done")
    assert pipe.restorer.tiles_run == 3
    assert (done["step"], done["steps"]) == (3, 3)
    assert pipe.restorer.on_tile is None


def test_cancel_during_single_pass_discards_the_result(server):
    pipe, base = server
    started, finish = threading.Event(), threading.Event()

    def single_pass(job):
        started.set()
        assert finish.wait(5)
        return b"result"

    job = pipe.runner.submit("depth", {}, single_pass)
    assert started.wait(5)
    try:
        assert request(f"{base}/v1/jobs/{job.id}", "DELETE")[0] == 200
    finally:
        finish.set()
    wait_for(base, job.id, "cancelled")
    assert job.png is None and job.work is None
    assert request(f"{base}/v1/jobs/{job.id}/image")[0] == 409


def test_cancelled_queue_entries_do_not_accumulate(server):
    pipe, base = server
    pipe.gate.clear()
    body = {"prompt": "fox", "width": 16, "height": 16, "steps": 1}
    first = json.loads(request(f"{base}/v1/images", "POST", body)[1])["id"]
    wait_for(base, first, "running")
    try:
        for _ in range(50):
            job = pipe.runner.submit("image", {}, lambda job: b"unused")
            pipe.runner.cancel(job)
        assert not pipe.runner._pending
        assert pipe.runner.queued == 0
        assert len(pipe.runner.jobs) <= 34
    finally:
        pipe.gate.set()


def test_upload_intake_is_bounded_without_blocking_polling(server, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import iris_metal.server as api

    _, base = server
    entered, release = threading.Barrier(3), threading.Event()
    decode = api.decode_upload

    def blocked_decode(data):
        entered.wait(timeout=5)
        assert release.wait(5)
        return decode(data)

    monkeypatch.setattr(api, "decode_upload", blocked_decode)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(upload, base, "/v1/depth", (16, 16)) for _ in range(2)]
        try:
            entered.wait(timeout=5)
            assert upload(base, "/v1/depth", (16, 16)) == 429
            assert request(f"{base}/v1/health")[0] == 200
        finally:
            release.set()
        assert [future.result() for future in futures] == [202, 202]
    monkeypatch.setattr(api, "decode_upload", decode)
    assert upload(base, "/v1/depth", (16, 16)) == 202


def test_turbo_http_selects_model_and_reports_fixed_progress_and_metadata(server):
    pipe, base = server

    class Turbo(FakePipeline):
        model = "turbo"
        last_metadata = {"model": "turbo", "sampler": "deterministic-flow-euler"}

        def __call__(self, prompts, height, width, on_step, **kwargs):
            assert "steps" not in kwargs and "cfg_scale" not in kwargs and "negative_prompt" not in kwargs
            self.loaded = True
            return super().__call__(prompts, height, width, 4, on_step, **kwargs)

    turbo = Turbo()
    pipe.runner.models._load_generation = lambda model: turbo
    status, body = request(f"{base}/v1/images", "POST", {"prompt": "cat", "model": "turbo", "steps": 4})
    assert status == 400
    status, body = request(
        f"{base}/v1/images", "POST", {"prompt": "cat", "model": "turbo", "width": 16, "height": 16}
    )
    assert status == 202
    done = wait_for(base, json.loads(body)["id"], "done")
    assert (done["step"], done["steps"]) == (4, 4)
    assert done["metadata"] == turbo.last_metadata
    _, body = request(f"{base}/v1/health")
    assert json.loads(body)["loaded_model"] == "turbo"
    assert not pipe.loaded
    request(f"{base}/v1/images", "POST", {"prompt": "cat", "steps": 1, "width": 16, "height": 16})
    wait_for(base, "2", "done")
    assert pipe.runner.models.pipe is pipe
    assert not turbo.loaded


def test_turbo_load_error_does_not_poison_original_queue(server):
    pipe, base = server

    def fail(model):
        raise FileNotFoundError("Turbo checkpoint missing")

    pipe.runner.models._load_generation = fail
    status, body = request(
        f"{base}/v1/images", "POST", {"prompt": "cat", "model": "turbo", "width": 16, "height": 16}
    )
    assert status == 202
    failed = wait_for(base, json.loads(body)["id"], "failed")
    assert "Turbo checkpoint missing" in failed["error"]
    status, body = request(
        f"{base}/v1/images", "POST", {"prompt": "cat", "steps": 1, "width": 16, "height": 16}
    )
    assert status == 202
    assert wait_for(base, json.loads(body)["id"], "done")["error"] is None


def test_remote_original_omitted_controls_use_checkpoint_defaults_on_cold_and_warm(server):
    pipe, base = server
    custom = SimpleNamespace(steps=2, cfg_scale=5.0, negative_prompt="custom-negative")
    pipe.cfg.sample = custom
    seen = []

    class Custom(FakePipeline):
        def __init__(self):
            super().__init__()
            self.cfg.sample = custom

        def __call__(self, prompts, height, width, steps, on_step, **kwargs):
            seen.append((steps, kwargs["cfg_scale"], kwargs["negative_prompt"]))
            return super().__call__(prompts, height, width, steps, on_step, **kwargs)

    models = pipe.runner.models
    models.pipe = models.original = None
    models._load_generation = lambda model: Custom()
    jobs = []
    for _ in range(2):
        status, body = request(f"{base}/v1/images", "POST", {"prompt": "cat", "width": 16, "height": 16})
        assert status == 202
        jobs.append(wait_for(base, json.loads(body)["id"], "done"))
    assert seen == [(2, 5.0, "custom-negative"), (2, 5.0, "custom-negative")]
    assert all(j["params"]["steps"] == 2 and j["params"]["cfg_scale"] == 5.0 for j in jobs)
