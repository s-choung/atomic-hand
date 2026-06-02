"""Standalone high-quality renderer, meant to be run as a SEPARATE PROCESS so the
live app never blocks. Reads a JSON parameter file (argv[1]) describing the
structure + current camera, renders a transparent-background PNG, and exits.

Usage:  python hq_render.py params.json
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecule_renderer import MoleculeRenderer


def main():
    if len(sys.argv) < 2:
        print("usage: hq_render.py params.json", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        p = json.load(f)

    r = MoleculeRenderer(p["filepath"])

    # match the live view's appearance
    r.style_idx = int(p.get("style_idx", 0))
    r._apply_style()
    hidden = p.get("hidden") or []
    if hidden:
        r.set_hidden_elements(set(hidden))
    if "cell_visible" in p:
        try:
            r.set_cell_visible(bool(p["cell_visible"]))
        except Exception:
            pass

    # match the live view's exact camera
    r.azimuth = float(p["azimuth"])
    r.elevation = float(p["elevation"])
    r.distance = float(p["distance"])
    r.center = np.array(p["center"], dtype=float)
    r.fov = float(p["fov"])

    img = r.render_hq_rgba(
        int(p.get("width", 2400)), int(p.get("height", 1800)),
        int(p.get("aa", 20)), bool(p.get("ao", True)), bool(p.get("shadows", True)),
    )
    ok = cv2.imwrite(p["output"], img)
    if not ok:
        print("imwrite failed", file=sys.stderr)
        return 1
    print("OK", p["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
