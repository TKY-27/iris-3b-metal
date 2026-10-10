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


@pytest.fixture
def server():
    pipe = FakePipeline()
    models = Models(pipe, load_task=lambda task: FakeDepth())
    httpd = LocalServer(("127.0.0.1", 0), make_handler(JobRunner(models, max_queued=2)))
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
