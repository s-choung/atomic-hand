"""
Gesture-controlled Video Player v2
왼쪽: 웹캠 + 제스처 가이드, 오른쪽: 영상 재생
MacBook 내장 FaceTime 카메라 자동 선택.

사용법:
  conda run -n base python gesture_player.py [video_path]
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

DEFAULT_VIDEO = "/Users/sean/Downloads/rotate_montage_3s.mp4"

GESTURE_GUIDE = [
    ("FIST",  "Grip fist",     "Play / Pause"),
    ("OPEN",  "Swipe Right",   "Forward +1s"),
    ("OPEN",  "Swipe Left",    "Rewind -1s"),
    ("POINT", "Point Up",      "Speed Up"),
    ("POINT", "Point Down",    "Speed Down"),
    ("PEACE", "Peace sign",    "Restart"),
]


def find_macbook_camera():
    """FaceTime HD Camera (MacBook 내장) 인덱스를 찾는다."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.split("\n")
        camera_names = []
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(":") and "Camera" in stripped:
                camera_names.append(stripped.rstrip(":"))

        for idx, name in enumerate(camera_names):
            if "FaceTime" in name:
                print(f"MacBook 카메라 발견: [{idx}] {name}")
                return idx

    except Exception:
        pass

    print("FaceTime 카메라 자동 탐지 실패, index 0 사용")
    return 0


class GestureDetector:
    def __init__(self):
        self.palm_x_history = deque(maxlen=15)
        self.cooldown = 0.45
        self.swipe_cooldown = 0.5
        self.last_action_time = 0
        self.last_swipe_time = 0
        self.prev_fist_state = None
        self.prev_gesture = None
        self.swipe_threshold = 0.08

    def _finger_extended(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def classify(self, lm):
        fingers = [
            self._finger_extended(lm, 8, 6),
            self._finger_extended(lm, 12, 10),
            self._finger_extended(lm, 16, 14),
            self._finger_extended(lm, 20, 18),
        ]
        extended = sum(fingers)

        if extended == 0:
            return "FIST"
        if extended >= 3:
            return "OPEN"
        if fingers[0] and not any(fingers[1:]):
            return "POINT"
        if fingers[0] and fingers[1] and not any(fingers[2:]):
            return "PEACE"
        return "OTHER"

    def detect_swipe(self, palm_x):
        now = time.time()
        self.palm_x_history.append((palm_x, now))

        if len(self.palm_x_history) < 4:
            return None
        if now - self.last_swipe_time < self.swipe_cooldown:
            return None

        recent = [(x, t) for x, t in self.palm_x_history if now - t < 0.35]
        if len(recent) < 3:
            return None

        dx = recent[-1][0] - recent[0][0]
        if abs(dx) > self.swipe_threshold:
            self.last_swipe_time = now
            self.palm_x_history.clear()
            return "LEFT" if dx > 0 else "RIGHT"
        return None

    def process(self, lm):
        gesture = self.classify(lm)
        now = time.time()
        action = None

        palm_x = (lm[0].x + lm[5].x + lm[17].x) / 3

        is_fist = (gesture == "FIST")
        if is_fist and self.prev_fist_state is False:
            if now - self.last_action_time > self.cooldown:
                action = "TOGGLE_PLAY"
                self.last_action_time = now
        self.prev_fist_state = is_fist

        if gesture == "OPEN":
            swipe = self.detect_swipe(palm_x)
            if swipe == "RIGHT":
                action = "SEEK_FORWARD"
            elif swipe == "LEFT":
                action = "SEEK_BACKWARD"

        if gesture == "POINT" and now - self.last_action_time > self.cooldown:
            tip_y = lm[8].y
            if tip_y < 0.35:
                action = "SPEED_UP"
                self.last_action_time = now
            elif tip_y > 0.65:
                action = "SPEED_DOWN"
                self.last_action_time = now

        if gesture == "PEACE":
            if self.prev_gesture != "PEACE" and now - self.last_action_time > self.cooldown:
                action = "RESTART"
                self.last_action_time = now

        self.prev_gesture = gesture
        return gesture, action


class VideoPlayer:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없음: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.playing = True
        self.speed = 1.0
        self.current_frame = None

    def read_frame(self):
        if not self.playing and self.current_frame is not None:
            return self.current_frame

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
        return self.current_frame

    def toggle_play(self):
        self.playing = not self.playing

    def seek(self, delta_sec):
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + delta_sec * self.fps
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, min(pos, self.total_frames - 1))))
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.playing = True

    def change_speed(self, delta):
        self.speed = round(max(0.25, min(4.0, self.speed + delta)), 2)

    def position_sec(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps

    def release(self):
        self.cap.release()


ACTION_LABELS = {
    "TOGGLE_PLAY": ("PLAY / PAUSE", (0, 200, 255)),
    "SEEK_FORWARD": (">>> FORWARD", (255, 150, 0)),
    "SEEK_BACKWARD": ("<<< REWIND", (255, 150, 0)),
    "SPEED_UP": ("SPEED UP", (0, 255, 150)),
    "SPEED_DOWN": ("SPEED DOWN", (0, 255, 150)),
    "RESTART": ("RESTART", (200, 0, 255)),
}

GESTURE_COLORS = {
    "FIST": (0, 100, 255),
    "OPEN": (0, 255, 100),
    "POINT": (255, 200, 0),
    "PEACE": (200, 100, 255),
    "OTHER": (150, 150, 150),
}


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_gesture_guide(frame, current_gesture, current_action):
    """화면 하단에 제스처 가이드 패널을 그린다."""
    h, w = frame.shape[:2]
    panel_h = 160
    panel_y = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "GESTURE GUIDE", (10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.line(frame, (10, panel_y + 28), (w - 10, panel_y + 28), (60, 60, 60), 1)

    col_w = w // 3
    items = [
        ("FIST",  "Grip Fist",    "Play/Pause"),
        ("OPEN",  "Swipe L / R",  "Rewind / Forward"),
        ("POINT", "Point Up/Down", "Speed +/-"),
        ("PEACE", "Peace Sign",   "Restart"),
    ]

    for i, (gest, how, what) in enumerate(items):
        row = i // 3
        col = i % 3
        x = 15 + col * col_w
        y = panel_y + 48 + row * 55

        is_active = (current_gesture == gest)
        color = GESTURE_COLORS.get(gest, (150, 150, 150))
        alpha_color = color if is_active else tuple(c // 2 for c in color)

        cv2.circle(frame, (x + 8, y + 5), 8, alpha_color, -1 if is_active else 1)
        cv2.putText(frame, how, (x + 22, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255) if is_active else (160, 160, 160), 1)
        cv2.putText(frame, what, (x + 22, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, alpha_color, 1)


def draw_action_popup(frame, action_text, action_color):
    """액션 발생 시 화면 중앙에 크게 표시."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(action_text, font, scale, thickness)

    cx, cy = w // 2, h // 3
    pad = 15
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (cx - tw // 2 - pad, cy - th - pad),
                  (cx + tw // 2 + pad, cy + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, action_text, (cx - tw // 2, cy),
                font, scale, action_color, thickness)


def draw_video_overlay(frame, player):
    """영상 위에 상태 정보 오버레이."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    progress = player.position_sec() / max(player.duration, 0.01)
    bar_x1, bar_x2 = 10, w - 10
    bar_y = h - 30
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 6), (80, 80, 80), -1)
    fill_x = bar_x1 + int((bar_x2 - bar_x1) * min(progress, 1.0))
    cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + 6), (0, 200, 255), -1)
    cv2.circle(frame, (fill_x, bar_y + 3), 6, (0, 200, 255), -1)

    state_icon = ">" if player.playing else "||"
    pos = player.position_sec()
    info = f"{state_icon}  {pos:.1f}s / {player.duration:.1f}s    x{player.speed:.2f}"
    cv2.putText(frame, info, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO

    print("=" * 50)
    print("Gesture Video Player v2")
    print("=" * 50)
    print(f"Video: {video_path}")
    print()
    print("  Grip fist       -> Play/Pause")
    print("  Open hand swipe -> Rewind / Forward")
    print("  Point up/down   -> Speed +/- 0.25x")
    print("  Peace sign      -> Restart")
    print("  q / ESC         -> Quit")
    print("=" * 50)

    cam_idx = find_macbook_camera()
    player = VideoPlayer(video_path)

    cam = cv2.VideoCapture(cam_idx)
    if not cam.isOpened():
        print(f"ERROR: 카메라 [{cam_idx}]를 열 수 없음.")
        print("다른 인덱스 시도 중...")
        for try_idx in range(3):
            if try_idx == cam_idx:
                continue
            cam = cv2.VideoCapture(try_idx)
            if cam.isOpened():
                print(f"카메라 [{try_idx}] 열림!")
                break
        if not cam.isOpened():
            print("ERROR: 사용 가능한 카메라 없음. 카메라 권한을 확인하세요.")
            player.release()
            return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = GestureDetector()
    action_label = None
    action_color = (255, 255, 255)
    action_until = 0

    display_h = 700

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        max_num_hands=1,
    ) as hands:
        while True:
            ret_cam, cam_frame = cam.read()
            if not ret_cam:
                break
            cam_frame = cv2.flip(cam_frame, 1)

            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            gesture = None
            action = None
            now = time.time()

            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        cam_frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    gesture, action = detector.process(hlm.landmark)

            if action:
                label_info = ACTION_LABELS.get(action, (action, (255, 255, 255)))
                action_label = label_info[0]
                action_color = label_info[1]
                action_until = now + 1.2

                if action == "TOGGLE_PLAY":
                    player.toggle_play()
                elif action == "SEEK_FORWARD":
                    player.seek(1.0)
                elif action == "SEEK_BACKWARD":
                    player.seek(-1.0)
                elif action == "SPEED_UP":
                    player.change_speed(0.25)
                elif action == "SPEED_DOWN":
                    player.change_speed(-0.25)
                elif action == "RESTART":
                    player.restart()

            show_action = now < action_until

            vid_frame = player.read_frame()
            if vid_frame is None:
                break

            # 카메라 리사이즈
            cam_h, cam_w = cam_frame.shape[:2]
            cam_new_w = int(cam_w * display_h / cam_h)
            cam_display = cv2.resize(cam_frame, (cam_new_w, display_h), interpolation=cv2.INTER_LINEAR)

            # 영상 리사이즈 (비율 유지, 높은 품질)
            vid_h, vid_w = vid_frame.shape[:2]
            vid_new_w = int(vid_w * display_h / vid_h)
            vid_display = cv2.resize(vid_frame, (vid_new_w, display_h), interpolation=cv2.INTER_LANCZOS4)

            # 카메라: 제스처 가이드
            draw_gesture_guide(cam_display, gesture, action)

            # 영상: 상태 오버레이
            draw_video_overlay(vid_display, player)

            # 액션 팝업
            if show_action:
                draw_action_popup(vid_display, action_label, action_color)

            combined = np.hstack([cam_display, vid_display])

            max_win_w = 1400
            if combined.shape[1] > max_win_w:
                scale = max_win_w / combined.shape[1]
                combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Gesture Video Player", combined)

            delay = max(1, int((1000 / player.fps) / player.speed))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                player.toggle_play()

    cam.release()
    player.release()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == "__main__":
    main()
