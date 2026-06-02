# Atomic Hand

> Rotate, zoom, and switch 3D molecules with your bare hands. No mouse, no keyboard. Just a webcam.

**Atomic Hand** is a hand-gesture molecular viewer. It pairs [MediaPipe](https://developers.google.com/mediapipe) hand tracking with [OVITO](https://www.ovito.org/)'s Tachyon ray tracer, so you can manipulate atomic structures in real time using only gestures in front of your laptop camera.

Built and used on **macOS**. The dependency + rendering core is also **CI-tested on Windows** (the GUI/camera path on Windows is still experimental, see [Platform support](#platform-support)).

![Atomic Hand demo](docs/media/atomichand_poster.jpg)

**Landing page & demo videos:** [atomic-hand.schoung.com](https://atomic-hand.schoung.com)

## Demos

| Clip | File |
|---|---|
| Inspiration to reality: Minority Report (left) + Atomic Hand (right), side by side | [`docs/media/combined_demo.mp4`](docs/media/combined_demo.mp4) |
| Atomic Hand in action | [`docs/media/AtomicHand_demo_compressed.mp4`](docs/media/AtomicHand_demo_compressed.mp4) |

## Install

The fastest path is one line in your terminal. It installs Miniconda if you don't have it, creates an isolated environment, installs everything, and drops a launcher on your Desktop.

**macOS**
```bash
curl -fsSL https://raw.githubusercontent.com/s-choung/atomic-hand/main/install.command | bash
```

**Windows** (PowerShell)
```powershell
irm https://raw.githubusercontent.com/s-choung/atomic-hand/main/install.ps1 | iex
```

**Using Claude Code or Codex?** Just ask it: `"clone https://github.com/s-choung/atomic-hand and install it"`, and let it run the steps.

After installing, double-click the generated **Atomic Hand** launcher (on your Desktop) and grant camera permission when asked. Snapshots are saved to `~/Desktop/atomic-hand/`.

<details>
<summary>Manual install</summary>

Prereqs: `git` and `conda` (Miniconda is fine). On macOS without git: `xcode-select --install`.

```bash
git clone https://github.com/s-choung/atomic-hand.git
cd atomic-hand
conda create -n atomic-hand python=3.11 -y
conda activate atomic-hand
pip install ovito                        # install OVITO first (it brings its own Qt)
pip install mediapipe==0.10.14 opencv-python PySide6 numpy
python src/app_window.py structures      # the app globs the folder itself
```
</details>

## Gestures

| Gesture | Action |
|---|---|
| 🖐️ Open hand + move | Rotate |
| 🤚 Wrist twist | Spin (azimuth) |
| 👐 Two hands spread / pinch | Zoom in / out |
| 👉 Swipe left / right | Previous / next structure |
| ✊ Fist | Pause tracking |
| 👍 Thumbs up | Save a transparent HQ snapshot (toggle in panel) |
| 🖱️ Left-drag / right-drag | Rotate / pan |
| Scroll, arrow keys, click a gallery thumbnail | Zoom / switch structure |
| `S` | Toggle the settings panel |

## Features

- **Real-time hand tracking** (MediaPipe) for rotate / zoom / swipe / pause.
- **Background HQ render** that never freezes the live view: a thumbs-up (or the HQ button) renders a **transparent-background PNG** in a separate process. Quality presets (Fast to Ultra) and a chosen save folder.
- **OVITO Tachyon** rendering with ball-and-stick / space-filling / wireframe styles, automatic van-der-Waals bonds, optional unit-cell box.
- Mouse + keyboard fallback, drag-and-drop or Open File to load structures, clickable thumbnail gallery, camera on/off toggle.
- Clean diagonal default view, framed to the molecule rather than the simulation cell.

## Run

```bash
conda activate atomic-hand
python src/app_window.py structures            # all bundled molecules
python src/app_window.py structures/C60.xyz    # a specific file
```

Hold an open hand in front of the camera and move it to rotate. Thumbs-up to save.

## Platform support

| Platform | Status |
|---|---|
| macOS (Apple Silicon / Intel) | Primary platform, built and used here. |
| Windows | Dependencies + rendering + background save are **CI-tested** on `windows-latest` ([workflow](.github/workflows/windows-test.yml), [smoke test](tests/smoke_test.py)). The GUI window, camera, and gestures are not yet confirmed by a human on Windows, and `install.ps1` is a draft. Try it and please report back. |
| Linux | Same stack should work; untested. |

## Architecture

```
CameraReaderThread  -> grab frames, drain buffer, own the camera device
MediaPipeThread     -> latest frame -> MediaPipe -> gesture engine
MainThread (QTimer) -> consume gesture, Tachyon render (low-res while moving), overlays
hq_render.py        -> separate process: full-quality transparent PNG, off the UI thread
```

- **`src/app_window.py`** main app (PySide6 + threaded camera)
- **`src/gesture_engine_v2.py`** gesture classification (rotate, zoom, swipe, wrist twist, thumbs-up)
- **`src/molecule_renderer.py`** OVITO Tachyon wrapper (camera, styles, bonds)
- **`src/hq_render.py`** standalone high-quality renderer
- **`structures/`** demo molecules (`.xyz`)
- **`install.command` / `install.ps1`** one-click installers
- **`tests/smoke_test.py`** headless import + render check (run in CI)

## Demo molecules

`C60`, `C60_buckyball`, `ethanol`, `ethane`, `acetaldehyde`, `glyoxal`, `formic_acid`, `pyrrole`, `water`, all in `structures/` as `.xyz`. Drop your own `.xyz` onto the window, or use **Open File** in the settings panel (`S`).

## Performance notes

- Tachyon is **CPU-only**. While interacting, the molecule renders at reduced resolution (then upscaled) and snaps to full quality when you stop moving.
- The background HQ render runs at low priority with capped threads so the live view stays responsive.
- On an 8 GB machine, close heavy apps; swap pressure hurts more than CPU. Turn the camera off (panel button) when you only want to view.

## Status & limitations

Atomic Hand is an **early prototype (v1.3)**, vibe-coded for fun. Viewing gestures are solid; hands-on atom editing is experimental. The native stack pins some older versions (e.g. `mediapipe==0.10.14`), so stick to the install steps.

It is **100% open source** (MIT). No telemetry, no network calls: the app opens your webcam locally and renders on your machine. Read the [source](src/) before you run it.

## Support

It's MIT licensed, so use it however you like. If it's useful:

- ⭐ **[Star it on GitHub](https://github.com/s-choung/atomic-hand)** — genuinely makes my day.
- ☕ **[Buy me a coffee](https://www.buymeacoffee.com/schoung)** — optional, always appreciated.

## License

[MIT](LICENSE) © 2026 Seokhyun Choung
