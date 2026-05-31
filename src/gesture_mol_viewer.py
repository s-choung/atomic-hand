"""
GestureMol v4 (Simplified B)
손 펴고 이동 = 회전, 주먹 = 멈춤.
줌/스타일은 마우스+키보드로만.

사용법:
  conda run -n base python src/gesture_mol_viewer.py [structure_files...]
"""

import sys
import os
import subprocess
import cv2
import mediapipe as mp
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecule_renderer import MoleculeRenderer
from gesture_engine import GestureEngine

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

DEFAULT_FILES = [
    "/Users/sean/Library/CloudStorage/GoogleDrive-wjdtjrgus9967@gmail.com/My Drive/Research_2026/playground/97_understand_ark/ARKdistillationV2_repo/OMat24_and_HTS/src/fairchem/data/odac/force_field/FF_calcs/CO2_example/frame.cif",
    "/Users/sean/Library/CloudStorage/GoogleDrive-wjdtjrgus9967@gmail.com/My Drive/Research_2026/playground/96_matlantis/1_Analysis_of_from20250522/ceo2_slabs/CeO2_111_12x12_t10.xyz",
]

GESTURE_INFO = {
    "OPEN": ("ROTATING", (50, 200, 255)),
    "FIST": ("PAUSED", (80, 80, 255)),
    "IDLE": ("", (100, 100, 100)),
    "OTHER": ("", (100, 100, 100)),
}


class MouseState:
    def __init__(self):
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.scroll_delta = 0
        self.renderer = None

mouse = MouseState()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.dragging = True
        mouse.last_x = x
        mouse.last_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse.dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse.dragging:
        dx = x - mouse.last_x
        dy = y - mouse.last_y
        mouse.last_x = x
        mouse.last_y = y
        if mouse.renderer:
            mouse.renderer.rotate(dx * 0.5, -dy * 0.5)
    elif event == cv2.EVENT_MOUSEWHEEL:
        mouse.scroll_delta = -1 if flags > 0 else 1


def find_macbook_camera():
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=5,
        )
        names = []
        for line in result.stdout.split("\n"):
            s = line.strip()
            if s.endswith(":") and "Camera" in s:
                names.append(s.rstrip(":"))
        for idx, name in enumerate(names):
            if "FaceTime" in name:
                return idx
    except Exception:
        pass
    return 0


def draw_gesture_guide(frame, gesture_name):
    h, w = frame.shape[:2]
    guide_x = 10
    guide_y = h - 145

    overlay = frame.copy()
    cv2.rectangle(overlay, (guide_x, guide_y), (guide_x + 270, h - 10), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "CONTROLS", (guide_x + 8, guide_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.line(frame, (guide_x + 5, guide_y + 24), (guide_x + 260, guide_y + 24), (60, 60, 60), 1)

    items = [
        ("OPEN",  "Open hand + move",   "Rotate",       (50, 200, 255)),
        ("FIST",  "Fist",               "Pause",        (80, 80, 255)),
        ("MOUSE", "Mouse drag",         "Rotate",       (180, 180, 180)),
        ("MOUSE", "Mouse scroll",       "Zoom in/out",  (180, 180, 180)),
        ("KEY",   "T / R / N / P / Q",  "Style/Reset/File/Quit", (140, 140, 140)),
    ]

    for i, (gest, how, what, color) in enumerate(items):
        y = guide_y + 44 + i * 20
        is_active = (gesture_name == gest)
        c = color if is_active else tuple(min(255, int(cc * 0.5)) for cc in color)
        text_c = (255, 255, 255) if is_active else (170, 170, 170)
        cv2.circle(frame, (guide_x + 10, y - 3), 4, c, -1 if is_active else 1)
        cv2.putText(frame, f"{how}  ->  {what}", (guide_x + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_c, 1)


def draw_camera_overlay(mol_frame, cam_frame, gesture_name):
    mol_h, mol_w = mol_frame.shape[:2]
    cam_display_h = int(mol_h * 0.20)
    cam_h, cam_w = cam_frame.shape[:2]
    cam_display_w = int(cam_w * cam_display_h / cam_h)
    cam_small = cv2.resize(cam_frame, (cam_display_w, cam_display_h))

    x1 = mol_w - cam_display_w - 10
    y1 = mol_h - cam_display_h - 150
    x2 = x1 + cam_display_w
    y2 = y1 + cam_display_h

    if y1 >= 0 and x1 >= 0:
        roi = mol_frame[y1:y2, x1:x2]
        if roi.shape == cam_small.shape:
            blended = cv2.addWeighted(cam_small, 0.35, roi, 0.65, 0)
            mol_frame[y1:y2, x1:x2] = blended

        info_text, info_color = GESTURE_INFO.get(gesture_name, ("", (100, 100, 100)))
        if info_text:
            cv2.putText(mol_frame, info_text, (x1 + 5, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, info_color, 1)
        cv2.rectangle(mol_frame, (x1, y1), (x2, y2), info_color, 1)


def draw_status_bar(frame, renderer, fps, file_idx, total_files, is_playing):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    filename = os.path.basename(renderer.filepath)
    fi = f"[{file_idx+1}/{total_files}] " if total_files > 1 else ""
    state = "PLAY" if is_playing else "PAUSE"
    info = f"{fi}{filename}  |  {renderer.num_atoms} atoms  |  {renderer.current_style()}  |  {state}  |  {fps:.0f} fps"
    cv2.putText(frame, info, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)


def draw_action_popup(frame, text, color):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
    cx, cy = w // 2, h // 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx-tw//2-12, cy-th-12), (cx+tw//2+12, cy+8), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (cx-tw//2, cy), font, 0.9, color, 2)


def main():
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
    else:
        file_list = [f for f in DEFAULT_FILES if os.path.exists(f)]
        if not file_list:
            print("No files. Pass file paths as arguments.")
            return

    file_idx = 0
    renderer = MoleculeRenderer(file_list[file_idx])
    print(f"Loaded: {os.path.basename(file_list[file_idx])} ({renderer.num_atoms} atoms)")

    cam_idx = find_macbook_camera()
    cam = cv2.VideoCapture(cam_idx)
    if not cam.isOpened():
        for i in range(3):
            if i == cam_idx:
                continue
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                break
    if not cam.isOpened():
        print("ERROR: No camera. Check camera permissions.")
        renderer.cleanup()
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    engine = GestureEngine()
    mouse.renderer = renderer

    action_label = None
    action_color = (255, 255, 255)
    action_until = 0
    fps = 0
    mol_w, mol_h = 1000, 750

    win = "GestureMol"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win, mol_w, mol_h)
    cv2.setMouseCallback(win, mouse_callback)

    print("=== GestureMol v4 ===")
    print("  Open hand + move -> Rotate")
    print("  Fist             -> Pause")
    print("  Mouse drag/scroll -> Rotate/Zoom")
    print("  T=Style R=Reset N/P=File Q=Quit")

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        max_num_hands=1,
    ) as hands:
        while True:
            t0 = time.time()

            ret, cam_frame = cam.read()
            if not ret:
                break
            cam_frame = cv2.flip(cam_frame, 1)

            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            landmarks = None
            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        cam_frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    landmarks = hlm.landmark

            result = engine.process(landmarks)
            now = time.time()

            d_az, d_el = result.rotation_delta
            if d_az != 0 or d_el != 0:
                renderer.rotate(d_az, d_el)

            if mouse.scroll_delta != 0:
                renderer.zoom(1.08 if mouse.scroll_delta > 0 else 0.92)
                mouse.scroll_delta = 0

            mol_frame = renderer.render(mol_w, mol_h)
            render_time = time.time() - t0
            fps = fps * 0.85 + (1.0 / max(render_time, 0.001)) * 0.15

            draw_status_bar(mol_frame, renderer, fps, file_idx, len(file_list), result.is_playing)
            draw_camera_overlay(mol_frame, cam_frame, result.gesture)
            draw_gesture_guide(mol_frame, result.gesture)

            if now < action_until:
                draw_action_popup(mol_frame, action_label, action_color)

            cv2.imshow(win, mol_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                renderer.reset_camera()
            elif key == ord("t"):
                renderer.next_style()
                action_label = renderer.current_style()
                action_color = (200, 100, 255)
                action_until = time.time() + 1.2
            elif key == ord("n") and len(file_list) > 1:
                file_idx = (file_idx + 1) % len(file_list)
                renderer.cleanup()
                renderer = MoleculeRenderer(file_list[file_idx])
                mouse.renderer = renderer
                action_label = os.path.basename(file_list[file_idx])
                action_color = (255, 150, 0)
                action_until = time.time() + 1.5
            elif key == ord("p") and len(file_list) > 1:
                file_idx = (file_idx - 1) % len(file_list)
                renderer.cleanup()
                renderer = MoleculeRenderer(file_list[file_idx])
                mouse.renderer = renderer
                action_label = os.path.basename(file_list[file_idx])
                action_color = (255, 150, 0)
                action_until = time.time() + 1.5

    cam.release()
    renderer.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
