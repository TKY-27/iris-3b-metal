"""Before/after slider and point-cloud viewer for the demo, rendered as plain HTML.

Gradio does not run scripts inside ``gr.HTML``, so the elements carry their data
inline and ``HEAD_JS`` (passed to ``gr.Blocks(head=...)``) wires up every
``.iris-ba`` / ``.iris-cloud`` element as it appears.
"""

import base64
import io

import numpy as np
from PIL import Image

HFOV_DEG = 60  # assumed horizontal field of view for back-projection
MAX_POINTS = 80_000


def _data_uri(image: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "WEBP", quality=quality)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def slider_html(before: Image.Image, after: Image.Image, before_label: str, after_label: str) -> str:
    before = before.convert("RGB").resize(after.size, Image.Resampling.BICUBIC)
    return (
        f'<div class="iris-ba" style="aspect-ratio:{after.width}/{after.height}">'
        f'<img class="iris-ba-after" src="{_data_uri(after)}" alt="{after_label}" draggable="false">'
        f'<img class="iris-ba-before" src="{_data_uri(before)}" alt="{before_label}" draggable="false">'
        '<div class="iris-ba-line"><span>↔</span></div>'
        f'<span class="iris-ba-label" style="left:16px">{before_label}</span>'
        f'<span class="iris-ba-label" style="right:16px">{after_label}</span></div>'
    )


def cloud_html(depth: np.ndarray, image: Image.Image) -> str:
    """Back-project relative log depth to a coloured point cloud.

    The prediction is log depth up to scale and shift, so ``exp`` of it gives
    depth up to scale (shift in log space is a global scale); it is rescaled
    to a median of 3 for viewing.
    """
    h, w = depth.shape
    rgb = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR))
    stride = max(1, int(np.ceil(np.sqrt(h * w / MAX_POINTS))))
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = np.exp(depth[v, u].astype(np.float64))
    z = 3 * z / np.median(z)
    f = (w / 2) / np.tan(np.radians(HFOV_DEG) / 2)
    xyz = np.stack([(u + 0.5 - w / 2) * z / f, (v + 0.5 - h / 2) * z / f, z], -1).reshape(-1, 3)
    payload = xyz.astype("<f4").tobytes() + rgb[v, u].reshape(-1, 3).astype(np.uint8).tobytes()
    data = base64.b64encode(payload).decode()
    return (
        f'<div class="iris-cloud" data-points="{len(xyz)}" data-cloud="{data}" '
        'aria-label="Point cloud from predicted depth. Drag to orbit, scroll to zoom."></div>'
    )


CSS = """
.iris-ba { position: relative; width: 100%; max-height: 70vh; margin: 0 auto; cursor: ew-resize;
  touch-action: none; user-select: none; overflow: hidden; background: #0c0c0c; }
.iris-ba img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; display: block; }
.iris-ba-line { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: #fff; z-index: 3; }
.iris-ba-line span { position: absolute; top: 50%; left: 50%; width: 40px; height: 40px; transform: translate(-50%, -50%);
  display: flex; align-items: center; justify-content: center; border-radius: 999px; background: #fff; color: #000;
  font: 12px 'IBM Plex Mono', monospace; box-shadow: 0 4px 14px rgba(0,0,0,.35); }
.iris-ba-label { position: absolute; bottom: 16px; z-index: 2; pointer-events: none; border-radius: 4px;
  background: rgba(0,0,0,.6); padding: 6px 10px; color: #fff; backdrop-filter: blur(4px);
  font: 10px/1 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .12em; }
.iris-cloud { position: relative; width: 100%; aspect-ratio: 4 / 3; cursor: grab; touch-action: none; overflow: hidden;
  border: 1px solid var(--sl-rule); margin-top: 12px; }
.iris-cloud:active { cursor: grabbing; }
.iris-cloud canvas { display: block; width: 100%; height: 100%; }
"""

HEAD_JS = """
<script type="module">
import { Mesh, Vec3, Orbit, Camera, Program, Geometry, Renderer, Transform } from "https://esm.sh/ogl@1.0.11";

const VERTEX = `attribute vec3 position; attribute vec3 color;
uniform mat4 modelViewMatrix; uniform mat4 projectionMatrix; uniform float uSize; varying vec3 vColor;
void main() { vColor = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv; gl_PointSize = uSize / -mv.z; }`;
const FRAGMENT = `precision highp float; varying vec3 vColor; void main() { gl_FragColor = vec4(vColor, 1.0); }`;

function slider(el) {
  const before = el.querySelector(".iris-ba-before"), line = el.querySelector(".iris-ba-line");
  const set = (x) => { const r = el.getBoundingClientRect();
    const s = Math.max(0, Math.min(100, ((x - r.left) / r.width) * 100));
    before.style.clipPath = `inset(0 ${100 - s}% 0 0)`; line.style.left = `${s}%`; };
  before.style.clipPath = "inset(0 50% 0 0)";
  el.addEventListener("pointerdown", (e) => { el.setPointerCapture(e.pointerId); set(e.clientX); });
  el.addEventListener("pointermove", (e) => { if (e.buttons) set(e.clientX); });
}

// camera-space cloud (x right, y down, z forward), flipped to GL axes, wiggling until touched
function cloud(wrap) {
  const points = +wrap.dataset.points;
  const buf = Uint8Array.from(atob(wrap.dataset.cloud), (c) => c.charCodeAt(0)).buffer;
  const xyz = new Float32Array(buf, 0, points * 3), rgb = new Uint8Array(buf, points * 12, points * 3);
  const position = new Float32Array(points * 3), color = new Float32Array(points * 3);
  let cz = 0;
  for (let i = 0; i < points * 3; i += 3) {
    position[i] = xyz[i]; position[i + 1] = -xyz[i + 1]; position[i + 2] = -xyz[i + 2]; cz += position[i + 2];
    color[i] = rgb[i] / 255; color[i + 1] = rgb[i + 1] / 255; color[i + 2] = rgb[i + 2] / 255;
  }
  cz /= points;
  let touched = false;
  wrap.addEventListener("pointerdown", () => (touched = true));
  wrap.addEventListener("wheel", () => (touched = true), { passive: true });
  const renderer = new Renderer({ dpr: Math.min(window.devicePixelRatio, 2), alpha: true });
  const gl = renderer.gl;
  wrap.appendChild(gl.canvas);
  const camera = new Camera(gl, { fov: 50, near: 0.05, far: 200 });
  const scene = new Transform(), pivot = new Transform();
  pivot.setParent(scene);
  const resize = () => { renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    camera.perspective({ aspect: wrap.clientWidth / wrap.clientHeight }); };
  new ResizeObserver(resize).observe(wrap);
  resize();
  const geometry = new Geometry(gl, { position: { size: 3, data: position }, color: { size: 3, data: color } });
  const program = new Program(gl, { vertex: VERTEX, fragment: FRAGMENT, uniforms: { uSize: { value: 8 * renderer.dpr } } });
  const mesh = new Mesh(gl, { mode: gl.POINTS, geometry, program });
  pivot.position.z = cz; mesh.position.z = -cz; mesh.setParent(pivot);
  camera.position.set(0, 0, 0.01);
  const orbit = new Orbit(camera, { element: wrap, target: new Vec3(0, 0, cz), ease: 0.12, inertia: 0.85 });
  const t0 = performance.now();
  const tick = (now) => {
    if (!wrap.isConnected) { gl.getExtension("WEBGL_lose_context")?.loseContext(); return; }
    if (!touched) { const w = 2 * Math.PI * 0.1 * ((now - t0) / 1000);
      pivot.rotation.y = 0.18 * Math.sin(w); pivot.rotation.x = 0.05 * Math.sin(2 * w); }
    orbit.update(); renderer.render({ scene, camera }); requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const init = () => {
  document.querySelectorAll(".iris-ba:not([data-ready])").forEach((el) => { el.dataset.ready = 1; slider(el); });
  document.querySelectorAll(".iris-cloud:not([data-ready])").forEach((el) => { el.dataset.ready = 1; cloud(el); });
};
new MutationObserver(init).observe(document.documentElement, { childList: true, subtree: true });
init();
</script>
"""
