# Atomic Hand v1.2 Handover

## Status: WORKING

Hand gesture molecular viewer. Tachyon ray-tracing + MediaPipe hand tracking.

## Run

```bash
# From CLI
conda run -n base python src/app_window.py [structure_files...]

# Or launch from ~/Applications/Atomic Hand.app
```

## Gestures

| Gesture | Action |
|---|---|
| ✋ Open hand + move | Rotate |
| ✋ Wrist twist | Rotate (azimuth) |
| 👐 Two hands spread/pinch | Zoom in/out |
| ✊ Fist | Pause |
| ✊ Fist + swipe L/R | Switch structure |
| 🖱 Mouse drag | Rotate |
| ⚙ Mouse scroll | Zoom |
| ⌨ ← → | Switch structure |
| S key | Toggle settings panel |

## Architecture

```
CameraReaderThread  -> grab() max speed, drain buffer
MediaPipeThread     -> latest frame -> MediaPipe -> gesture_engine
MainThread          -> QTimer 16ms: consume gesture, Tachyon render, overlay
```

## Rendering

- Interaction: Tachyon AA=2, no AO/shadows (~26ms/frame)
- Idle (0.3s): Tachyon AA=4, AO on (~154ms, one frame)
- HQ Export: Tachyon AA=8, AO+shadows, 1920x1440 -> Desktop PNG

## Settings Panel (right side, S to toggle)

- Structure info (filename, atoms, fps)
- Open File button
- Controls guide with gesture icons
- Sensitivity sliders (rotation, zoom)
- Display toggles (axes, PBC cell)
- Element checkboxes (hide/show per element)
- HQ Render -> PNG + Open button

## Key Design Decisions

- Tachyon only (CPU ray-tracer). No OpenGLRenderer (segfaults on macOS Metal + PySide6)
- No cleanup() in closeEvent (OVITO pipeline.remove_from_scene crashes during render_image). os._exit(0) instead
- Pipeline modifications deferred to _tick via flags (prevents segfault from concurrent pipeline access)
- H-H bond cutoff = 0.0 (pairwise), others = 1.85 A
- Hand size filter > 0.08 (rejects glasses/face false positives)
- Non-periodic structures: cell hidden if volume < 1.0

## File Structure

```
117_gesture_mol_v2/
  src/
    app_window.py           <- Main (PySide6 + threaded camera)
    gesture_engine_v2.py    <- Gestures (rotation, zoom, swipe, wrist)
    molecule_renderer.py    <- OVITO Tachyon wrapper
  structures/               <- Demo molecules (C60, ethanol, pyrrole, etc.)
  handover.md               <- THIS FILE
~/Applications/Atomic Hand.app  <- macOS app bundle
```

## Dependencies

- Python 3.11 (conda base)
- mediapipe==0.10.14
- OVITO 3.13+ (PySide6 bundled)
- OpenCV

## Performance Notes

- M2 MacBook Air 8GB: close PowerPoint/Chrome tabs for smooth operation
- Tachyon is CPU-only. GPU (Metal) not used for rendering
- AA=2 is sweet spot (26ms). AA=3 jumps to 114ms
- 8GB RAM is tight. Swap kills performance more than CPU
