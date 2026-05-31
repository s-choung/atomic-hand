#!/usr/bin/env python3
"""
bench_camera.py — Camera pipeline latency benchmark for GestureMol.

Measures each stage of the camera pipeline independently:
  1. Raw camera read (grab + retrieve)
  2. Resize + colorspace conversion
  3. MediaPipe hand inference
  4. Float32 fade-mask overlay composite
  5. CAP_PROP_BUFFERSIZE effectiveness test

Run:
    conda activate gesture-mol  # or whatever env
    python src/bench_camera.py
"""

import sys
import os
import time
import subprocess
import numpy as np
import cv2
import mediapipe as mp

N_FRAMES = 30
CAM_WIDTH = 320
CAM_HEIGHT = 240
SMALL_W, SMALL_H = 160, 120


# ── helpers ──────────────────────────────────────────────────────────

def find_macbook_camera():
    has_iphone = False
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "iPhone" in line or "iphone" in line:
                has_iphone = True
                break
    except Exception:
        pass
    return [1, 0, 2] if has_iphone else [0, 1, 2]


def open_camera():
    indices = find_macbook_camera()
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Opened camera index {idx}")
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_buf = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            print(f"  Requested: {CAM_WIDTH}x{CAM_HEIGHT}, buffer=1")
            print(f"  Actual:    {actual_w:.0f}x{actual_h:.0f}, buffer={actual_buf:.0f}")
            return cap
        cap.release()
    return None


def make_fade_mask(dw, dh):
    """Same logic as app_window._get_fade_mask"""
    fade = min(20, dw // 4, dh // 4)
    mask = np.ones((dh, dw), dtype=np.float32) * 0.55
    if fade > 0:
        mask[:fade, :] *= np.linspace(0, 1, fade)[:, None]
        mask[-fade:, :] *= np.linspace(1, 0, fade)[:, None]
        mask[:, :fade] *= np.linspace(0, 1, fade)[None, :]
        mask[:, -fade:] *= np.linspace(1, 0, fade)[None, :]
    return mask[:, :, None]


# ── benchmarks ───────────────────────────────────────────────────────

def bench_raw_camera_read(cam):
    """Measure raw grab+retrieve latency."""
    print(f"\n{'='*60}")
    print("1. RAW CAMERA READ  (grab + retrieve)")
    print(f"{'='*60}")
    times = []
    for i in range(N_FRAMES):
        t0 = time.perf_counter()
        cam.grab()
        ret, frame = cam.retrieve()
        dt = time.perf_counter() - t0
        times.append(dt)
        if not ret:
            print(f"  Frame {i}: FAILED")
    times = np.array(times)
    print(f"  Frames:  {len(times)}")
    print(f"  Mean:    {times.mean()*1000:.1f} ms")
    print(f"  Median:  {np.median(times)*1000:.1f} ms")
    print(f"  Min:     {times.min()*1000:.1f} ms")
    print(f"  Max:     {times.max()*1000:.1f} ms")
    print(f"  Std:     {times.std()*1000:.1f} ms")
    print(f"  Effective FPS: {1.0/times.mean():.1f}")
    return times


def bench_resize_and_convert(cam):
    """Measure resize + flip + cvtColor."""
    print(f"\n{'='*60}")
    print("2. RESIZE + FLIP + BGR→RGB")
    print(f"{'='*60}")
    # grab a frame first
    cam.grab()
    ret, frame = cam.retrieve()
    if not ret:
        print("  Cannot read frame!")
        return
    times = []
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        flipped = cv2.flip(frame, 1)
        small = cv2.resize(flipped, (SMALL_W, SMALL_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        dt = time.perf_counter() - t0
        times.append(dt)
    times = np.array(times)
    print(f"  Mean:  {times.mean()*1000:.2f} ms")
    print(f"  Max:   {times.max()*1000:.2f} ms")


def bench_mediapipe(cam):
    """Measure MediaPipe hand inference at 160x120."""
    print(f"\n{'='*60}")
    print("3. MEDIAPIPE HAND INFERENCE  (160x120, complexity=0)")
    print(f"{'='*60}")
    mp_hands = mp.solutions.hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        max_num_hands=2,
    )
    # warm up
    cam.grab()
    ret, frame = cam.retrieve()
    small = cv2.resize(cv2.flip(frame, 1), (SMALL_W, SMALL_H))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    mp_hands.process(rgb)

    times = []
    detected = 0
    for i in range(N_FRAMES):
        cam.grab()
        ret, frame = cam.retrieve()
        if not ret:
            continue
        small = cv2.resize(cv2.flip(frame, 1), (SMALL_W, SMALL_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0 = time.perf_counter()
        results = mp_hands.process(rgb)
        dt = time.perf_counter() - t0
        times.append(dt)
        if results.multi_hand_landmarks:
            detected += 1

    times = np.array(times)
    print(f"  Frames:      {len(times)}")
    print(f"  Detected:    {detected}/{len(times)}")
    print(f"  Mean:        {times.mean()*1000:.1f} ms")
    print(f"  Median:      {np.median(times)*1000:.1f} ms")
    print(f"  Min:         {times.min()*1000:.1f} ms")
    print(f"  Max:         {times.max()*1000:.1f} ms")
    print(f"  Std:         {times.std()*1000:.1f} ms")
    print(f"  Max FPS:     {1.0/times.mean():.1f}  (inference only)")
    mp_hands.close()
    return times


def bench_full_camera_thread_cycle(cam):
    """Simulate one full CameraThread iteration: grab+retrieve+flip+resize+mediapipe."""
    print(f"\n{'='*60}")
    print("4. FULL CameraThread CYCLE  (grab → mediapipe → store)")
    print(f"{'='*60}")
    mp_hands = mp.solutions.hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        max_num_hands=2,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_connections = mp.solutions.hands.HAND_CONNECTIONS

    # warm up
    cam.grab()
    ret, frame = cam.retrieve()
    small = cv2.resize(cv2.flip(frame, 1), (SMALL_W, SMALL_H))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_hands.process(rgb)

    times = []
    t_grab_list = []
    t_preprocess_list = []
    t_infer_list = []
    t_draw_list = []

    for i in range(N_FRAMES):
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        cam.grab()
        ret, cam_frame = cam.retrieve()
        t_grab = time.perf_counter() - t0
        if not ret:
            continue

        t0 = time.perf_counter()
        cam_frame = cv2.flip(cam_frame, 1)
        small = cv2.resize(cam_frame, (SMALL_W, SMALL_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t_preprocess = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = mp_hands.process(rgb)
        t_infer = time.perf_counter() - t0

        t0 = time.perf_counter()
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(cam_frame, hlm, mp_connections)
        t_draw = time.perf_counter() - t0

        total = time.perf_counter() - t_total_start
        times.append(total)
        t_grab_list.append(t_grab)
        t_preprocess_list.append(t_preprocess)
        t_infer_list.append(t_infer)
        t_draw_list.append(t_draw)

    times = np.array(times)
    t_grab_list = np.array(t_grab_list)
    t_preprocess_list = np.array(t_preprocess_list)
    t_infer_list = np.array(t_infer_list)
    t_draw_list = np.array(t_draw_list)

    print(f"  {'Stage':<20} {'Mean ms':>10} {'Max ms':>10}")
    print(f"  {'-'*40}")
    print(f"  {'grab+retrieve':<20} {t_grab_list.mean()*1000:>10.1f} {t_grab_list.max()*1000:>10.1f}")
    print(f"  {'flip+resize+cvt':<20} {t_preprocess_list.mean()*1000:>10.1f} {t_preprocess_list.max()*1000:>10.1f}")
    print(f"  {'mediapipe':<20} {t_infer_list.mean()*1000:>10.1f} {t_infer_list.max()*1000:>10.1f}")
    print(f"  {'draw_landmarks':<20} {t_draw_list.mean()*1000:>10.1f} {t_draw_list.max()*1000:>10.1f}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<20} {times.mean()*1000:>10.1f} {times.max()*1000:>10.1f}")
    print(f"  Effective CameraThread FPS: {1.0/times.mean():.1f}")
    mp_hands.close()


def bench_overlay_composite():
    """Measure the float32 fade-mask blending in _draw_camera_overlay."""
    print(f"\n{'='*60}")
    print("5. OVERLAY COMPOSITE  (float32 fade-mask blend)")
    print(f"{'='*60}")

    # Simulate typical sizes
    mol_w, mol_h = 800, 600
    display_h = int(mol_h * 0.20)   # 120
    display_w = int(320 * display_h / 240)  # ~160
    print(f"  Overlay size: {display_w}x{display_h}")
    print(f"  Mol frame:    {mol_w}x{mol_h}")

    cam_small = np.random.randint(0, 255, (display_h, display_w, 3), dtype=np.uint8)
    roi = np.random.randint(0, 255, (display_h, display_w, 3), dtype=np.uint8)
    mask3 = make_fade_mask(display_w, display_h)

    # warm up
    _ = (cam_small.astype(np.float32) * mask3 + roi.astype(np.float32) * (1 - mask3)).astype(np.uint8)

    times = []
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        blended = (cam_small.astype(np.float32) * mask3 + roi.astype(np.float32) * (1 - mask3)).astype(np.uint8)
        dt = time.perf_counter() - t0
        times.append(dt)

    times = np.array(times)
    print(f"  Mean:  {times.mean()*1000:.2f} ms")
    print(f"  Max:   {times.max()*1000:.2f} ms")

    # Compare with uint8 cv2.addWeighted approach
    print("\n  [Alternative: cv2.addWeighted with fixed alpha]")
    times2 = []
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        blended2 = cv2.addWeighted(cam_small, 0.55, roi, 0.45, 0)
        dt = time.perf_counter() - t0
        times2.append(dt)
    times2 = np.array(times2)
    print(f"  Mean:  {times2.mean()*1000:.2f} ms")
    print(f"  Max:   {times2.max()*1000:.2f} ms")
    print(f"  Speedup: {times.mean()/times2.mean():.1f}x")


def bench_status_overlay_gradient():
    """Measure the per-row alpha gradient in _draw_status_overlay."""
    print(f"\n{'='*60}")
    print("6. STATUS OVERLAY  (per-row alpha gradient, 48 rows)")
    print(f"{'='*60}")

    frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    times = []
    for _ in range(N_FRAMES):
        f = frame.copy()
        t0 = time.perf_counter()
        for y in range(48):
            alpha = 0.6 * (1.0 - y / 48.0)
            f[y, :] = (f[y, :].astype(float) * (1 - alpha)).astype(np.uint8)
        dt = time.perf_counter() - t0
        times.append(dt)

    times = np.array(times)
    print(f"  Per-row loop:  Mean {times.mean()*1000:.2f} ms, Max {times.max()*1000:.2f} ms")

    # Vectorized alternative
    times2 = []
    for _ in range(N_FRAMES):
        f = frame.copy()
        t0 = time.perf_counter()
        alphas = 0.6 * (1.0 - np.arange(48) / 48.0)
        f[:48] = (f[:48] * (1 - alphas[:, None, None])).astype(np.uint8)
        dt = time.perf_counter() - t0
        times2.append(dt)
    times2 = np.array(times2)
    print(f"  Vectorized:    Mean {times2.mean()*1000:.2f} ms, Max {times2.max()*1000:.2f} ms")


def bench_buffer_effectiveness(cam):
    """Test if CAP_PROP_BUFFERSIZE=1 actually discards stale frames on macOS."""
    print(f"\n{'='*60}")
    print("7. BUFFER EFFECTIVENESS TEST")
    print(f"{'='*60}")
    print("  Strategy: read 5 frames, sleep 2s, read 1 frame.")
    print("  If buffer=1 works, the post-sleep frame should be fresh.")

    # drain
    for _ in range(5):
        cam.grab()
        cam.retrieve()

    # read a burst to establish timing baseline
    burst_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        cam.grab()
        ret, _ = cam.retrieve()
        dt = time.perf_counter() - t0
        burst_times.append(dt)
    burst_mean = np.mean(burst_times)

    print(f"  Burst read mean: {burst_mean*1000:.1f} ms")
    print(f"  Sleeping 2 seconds...")
    time.sleep(2.0)

    # Post-sleep read with multiple grab attempts
    t0 = time.perf_counter()
    cam.grab()
    ret, _ = cam.retrieve()
    dt_single = time.perf_counter() - t0

    print(f"  Post-sleep single read: {dt_single*1000:.1f} ms")

    # Now test: grab multiple times to drain any buffer, measure last
    time.sleep(2.0)
    t0 = time.perf_counter()
    # Grab 3 times to drain buffer, only retrieve last
    cam.grab()
    cam.grab()
    cam.grab()
    ret, _ = cam.retrieve()
    dt_multi_grab = time.perf_counter() - t0

    print(f"  Post-sleep 3x grab + retrieve: {dt_multi_grab*1000:.1f} ms")

    if dt_single > burst_mean * 3:
        print("  ⚠ LIKELY BUFFERED: post-sleep read was much slower (camera had to")
        print("    deliver a fresh frame or the driver flushed a stale queue).")
        print("    CAP_PROP_BUFFERSIZE=1 may NOT be effective on this macOS backend.")
    else:
        print("  Post-sleep latency looks normal. Buffer setting may be working.")

    print("\n  [Recommendation]")
    print("  On macOS AVFoundation, CAP_PROP_BUFFERSIZE is often ignored.")
    print("  Workaround: use multiple grab() calls before retrieve() to drain stale frames,")
    print("  or set CAP_PROP_BUFFERSIZE to 0 and call grab() in a tight loop.")


def bench_draw_landmarks_on_full_frame(cam):
    """
    CRITICAL BOTTLENECK TEST:
    The app draws MediaPipe landmarks on cam_frame (320x240, the FULL camera frame)
    but only uses a 160x120 crop for detection. The landmark coordinates are in
    normalized [0,1] space, so draw_landmarks scales to the target image size.
    This means landmarks are drawn on the 320x240 frame — then later resized down
    for the overlay. This is wasteful but probably not the main bottleneck.
    """
    print(f"\n{'='*60}")
    print("8. DRAW LANDMARKS ON FULL-RES FRAME (320x240)")
    print(f"{'='*60}")
    mp_hands = mp.solutions.hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        max_num_hands=2,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_connections = mp.solutions.hands.HAND_CONNECTIONS

    # Find a frame with hands detected
    found = False
    for _ in range(30):
        cam.grab()
        ret, frame = cam.retrieve()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (SMALL_W, SMALL_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = mp_hands.process(rgb)
        if results.multi_hand_landmarks:
            found = True
            break

    if not found:
        print("  No hands detected in 30 frames — skipping this test.")
        print("  (Show your hand to the camera and re-run.)")
        mp_hands.close()
        return

    print(f"  Hands detected: {len(results.multi_hand_landmarks)}")
    times = []
    for _ in range(N_FRAMES):
        f = frame.copy()
        t0 = time.perf_counter()
        for hlm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(f, hlm, mp_connections)
        dt = time.perf_counter() - t0
        times.append(dt)

    times = np.array(times)
    print(f"  Mean:  {times.mean()*1000:.2f} ms")
    print(f"  Max:   {times.max()*1000:.2f} ms")
    mp_hands.close()


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("GestureMol Camera Pipeline Benchmark")
    print("=" * 60)

    cam = open_camera()
    if cam is None:
        print("ERROR: No camera found!")
        sys.exit(1)

    # Warm up camera (first few frames are often slow)
    print("Warming up camera (10 frames)...")
    for _ in range(10):
        cam.grab()
        cam.retrieve()

    bench_raw_camera_read(cam)
    bench_resize_and_convert(cam)
    bench_mediapipe(cam)
    bench_full_camera_thread_cycle(cam)
    bench_overlay_composite()
    bench_status_overlay_gradient()
    bench_buffer_effectiveness(cam)
    bench_draw_landmarks_on_full_frame(cam)

    cam.release()

    print(f"\n{'='*60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    print("""
Known bottlenecks in the current architecture:

1. CAMERA BUFFER (HIGH IMPACT):
   macOS AVFoundation often ignores CAP_PROP_BUFFERSIZE=1.
   The camera driver buffers 3-5 frames internally. When the
   CameraThread reads slower than the camera produces (e.g.,
   because MediaPipe takes 30-50ms), frames pile up in the
   buffer. Each read returns a STALE frame from the queue.
   → Fix: call grab() 2-3x before retrieve(), or drain with
     a tight grab() loop discarding frames until latency < threshold.

2. MEDIAPIPE BLOCKING THE CAMERA THREAD (HIGH IMPACT):
   MediaPipe inference runs synchronously in CameraThread.run().
   If inference takes 30-50ms, the thread can only read ~20-33 fps.
   Combined with buffering, this creates compound lag: the thread
   reads stale buffered frames AND processes them slowly.
   → Fix: decouple camera reading from inference. Read camera in
     one thread (always draining buffer), run MediaPipe in another.

3. OVITO TachyonRenderer ON EVERY DIRTY FRAME (MEDIUM IMPACT):
   When mol_dirty=True, _tick() calls renderer.render() which
   invokes TachyonRenderer (a ray tracer with antialiasing=4,
   ambient_occlusion=True, shadows=True) at 75% window resolution.
   This easily takes 50-200ms, BLOCKING the main thread _tick().
   While this doesn't directly cause camera lag (camera runs in
   a background thread), it means the main thread can't consume
   new camera frames during render, adding another frame of delay.
   → Fix: render OVITO in a background thread, or reduce quality
     during gesture interaction, or use OpenGL renderer.

4. FULL-RES LANDMARK DRAWING (LOW-MEDIUM IMPACT):
   draw_landmarks() is called on cam_frame (320x240) even though
   the overlay display is only ~160x120. Landmarks are drawn at
   full resolution, then the frame is resized down for display.
   Wasted work.
   → Fix: draw landmarks on the small (160x120) frame instead.

5. FLOAT32 OVERLAY BLENDING (LOW IMPACT):
   _draw_camera_overlay casts to float32 for fade-mask blending.
   At ~160x120, this is small (~0.1ms). Not a real bottleneck.

6. STATUS OVERLAY PER-ROW LOOP (LOW IMPACT):
   _draw_status_overlay has a Python for-loop over 48 rows with
   per-row float cast. Slow pattern but only 48 iterations.
   → Fix: vectorize with numpy broadcasting.

7. frame.copy() IN EVERY _tick (LOW IMPACT):
   mol_frame = self._mol_cache.copy() allocates a new 800x600x3
   array every tick. At ~1.4MB, this is ~0.5ms. Marginal.

8. GALLERY OVERLAY (LOW IMPACT):
   _draw_gallery does frame.copy() for overlay + addWeighted on
   the full frame. Could use ROI instead.

PRIMARY CAUSE OF "SEVERAL SECONDS LAG":
   The combination of #1 (buffer not drained) + #2 (slow reads)
   creates a pipeline delay. If the camera produces 30fps but
   CameraThread only consumes 20fps (due to MediaPipe), 10 frames/sec
   pile up in the OS buffer. After 3 seconds, you are viewing frames
   that are 3 seconds old. The buffer grows without bound on some
   macOS backends.
""")


if __name__ == "__main__":
    main()
