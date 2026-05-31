"""
키보드 컨트롤 분자 뷰어 (디버그/테스트용).
카메라 없이 키보드로 회전/줌 조작.

사용법:
  conda run -n base python src/viewer.py <CIF/XYZ 파일 경로>

키:
  A/D     — 좌/우 회전 (azimuth)
  W/S     — 상/하 회전 (elevation)
  +/=/-   — 줌 인/아웃
  T       — 렌더 스타일 전환
  R       — 카메라 리셋
  P       — 스크린샷 저장
  Q/ESC   — 종료
"""

import sys
import os
import cv2
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecule_renderer import MoleculeRenderer

ROTATE_STEP = 5.0
ZOOM_IN = 0.9
ZOOM_OUT = 1.1


def draw_overlay(frame, renderer, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    filename = os.path.basename(renderer.filepath)
    info = f"{filename}  |  {renderer.num_atoms} atoms  |  {renderer.current_style()}"
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cam = f"Az:{renderer.azimuth:.0f}  El:{renderer.elevation:.0f}  Dist:{renderer.distance:.0f}  FPS:{fps:.0f}"
    cv2.putText(frame, cam, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    keys = "A/D:Rotate  W/S:Tilt  +/-:Zoom  T:Style  R:Reset  P:Screenshot  Q:Quit"
    cv2.putText(frame, keys, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <structure_file>")
        print("  Supports: .cif, .xyz, .pdb, .poscar, .vasp, .lammps")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Loading: {filepath}")

    renderer = MoleculeRenderer(filepath)
    print(f"Loaded: {renderer.num_atoms} atoms, {renderer.num_frames} frames")

    width, height = 1000, 750
    fps = 0
    screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")

    while True:
        t0 = time.time()
        frame = renderer.render(width, height)
        render_time = time.time() - t0
        fps = 1.0 / max(render_time, 0.001)

        draw_overlay(frame, renderer, fps)
        cv2.imshow("Molecule Viewer", frame)

        key = cv2.waitKey(16) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("a"):
            renderer.rotate(-ROTATE_STEP, 0)
        elif key == ord("d"):
            renderer.rotate(ROTATE_STEP, 0)
        elif key == ord("w"):
            renderer.rotate(0, ROTATE_STEP)
        elif key == ord("s"):
            renderer.rotate(0, -ROTATE_STEP)
        elif key == ord("=") or key == ord("+"):
            renderer.zoom(ZOOM_IN)
        elif key == ord("-"):
            renderer.zoom(ZOOM_OUT)
        elif key == ord("t"):
            style = renderer.next_style()
            print(f"Style: {style}")
        elif key == ord("r"):
            renderer.reset_camera()
            print("Camera reset")
        elif key == ord("p"):
            os.makedirs(screenshot_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = os.path.splitext(os.path.basename(filepath))[0]
            path = os.path.join(screenshot_dir, f"{name}_{ts}.png")
            hires = renderer.render(1920, 1440)
            cv2.imwrite(path, hires)
            print(f"Screenshot: {path}")

    renderer.cleanup()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
