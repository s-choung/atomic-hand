"""Headless smoke test: verifies the rendering core works on this platform
(no GUI / camera needed). Run with QT_QPA_PLATFORM=offscreen.

Checks: deps import, MoleculeRenderer render, transparent HQ render, and the
separate-process hq_render.py path. Used by CI to validate Windows/Linux/macOS.
"""
import os
import sys
import glob
import json
import tempfile
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import ovito  # noqa: E402
import mediapipe  # noqa: E402
import PySide6  # noqa: E402
import cv2  # noqa: E402
import numpy  # noqa: E402
print("imports OK | ovito", getattr(ovito, "version", "?"), "| mediapipe", mediapipe.__version__)

# app module imports cleanly (catches syntax/import errors on this platform)
import app_window  # noqa: E402,F401
print("app_window import OK")

from molecule_renderer import MoleculeRenderer  # noqa: E402

files = sorted(glob.glob(os.path.join(ROOT, "structures", "*.xyz")))
assert files, "no .xyz structures found"

r = MoleculeRenderer(files[0])
img = r.render(320, 240)
assert img is not None and img.shape == (240, 320, 3), img.shape
rgba = r.render_hq_rgba(400, 300, aa=4, ao=False, shadows=False)
assert rgba.shape == (300, 400, 4), rgba.shape
assert bool((rgba[:, :, 3] == 0).any()), "expected transparent background pixels"
print("render OK", img.shape, "| hq", rgba.shape)

# separate-process render path (what the app uses for background save)
out = os.path.join(tempfile.gettempdir(), "smoke_hq.png")
params = {
    "filepath": files[0], "output": out,
    "width": 300, "height": 220, "aa": 4, "ao": False, "shadows": False,
    "azimuth": -55.0, "elevation": 20.0, "distance": float(r.distance),
    "center": [float(c) for c in r.center], "fov": float(r.fov),
    "style_idx": 0, "hidden": [], "cell_visible": False,
}
jp = os.path.join(tempfile.gettempdir(), "smoke_params.json")
with open(jp, "w") as f:
    json.dump(params, f)
rc = subprocess.call([sys.executable, os.path.join(ROOT, "src", "hq_render.py"), jp])
assert rc == 0 and os.path.exists(out), f"hq_render subprocess failed rc={rc}"
print("hq_render subprocess OK ->", out)

print("SMOKE OK")
