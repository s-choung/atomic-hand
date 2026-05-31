"""
Hand Gesture Video Controller using MediaPipe
MacBook Air 카메라로 손 제스처를 인식해서 영상 플레이어를 컨트롤한다.

제스처 매핑:
  ✊ 주먹 쥐기        → Space (재생/일시정지)
  👋 손바닥 펴고 왼쪽  → ← 뒤로 감기
  👋 손바닥 펴고 오른쪽 → → 앞으로 감기
  ☝️ 검지 위로        → ↑ 볼륨 업
  👇 검지 아래로       → ↓ 볼륨 다운
  ✌️ 피스 사인        → F 전체화면 토글

사용법:
  conda run -n base python hand_gesture_controller.py
  (종료: q 키 또는 ESC)
"""

import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class GestureState:
    def __init__(self):
        self.prev_gesture = None
        self.gesture_start_time = 0
        self.cooldown = 0.6  # 같은 제스처 재인식 쿨다운 (초)
        self.last_action_time = 0
        self.palm_x_history = deque(maxlen=10)
        self.swipe_threshold = 0.12  # 손바닥 x좌표 이동 임계값
        self.swipe_cooldown = 0.8
        self.last_swipe_time = 0
        self.prev_fist_state = None  # True=fist, False=open


def is_finger_extended(landmarks, finger_tip, finger_pip):
    return landmarks[finger_tip].y < landmarks[finger_pip].y


def is_thumb_extended(landmarks):
    return abs(landmarks[4].x - landmarks[2].x) > 0.05


def classify_gesture(landmarks):
    thumb = is_thumb_extended(landmarks)
    index = is_finger_extended(landmarks, 8, 6)
    middle = is_finger_extended(landmarks, 12, 10)
    ring = is_finger_extended(landmarks, 16, 14)
    pinky = is_finger_extended(landmarks, 20, 18)

    extended = sum([index, middle, ring, pinky])

    if extended == 0:
        return "FIST"
    if extended == 4:
        return "OPEN"
    if index and not middle and not ring and not pinky:
        return "POINT"
    if index and middle and not ring and not pinky:
        return "PEACE"

    return "OTHER"


def get_palm_center(landmarks):
    x = (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3
    y = (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
    return x, y


def detect_swipe(state, palm_x):
    state.palm_x_history.append((palm_x, time.time()))

    if len(state.palm_x_history) < 5:
        return None

    now = time.time()
    if now - state.last_swipe_time < state.swipe_cooldown:
        return None

    recent = [(x, t) for x, t in state.palm_x_history if now - t < 0.4]
    if len(recent) < 3:
        return None

    dx = recent[-1][0] - recent[0][0]

    if abs(dx) > state.swipe_threshold:
        state.last_swipe_time = now
        state.palm_x_history.clear()
        # 카메라 미러링이므로 방향 반전
        return "SWIPE_LEFT" if dx > 0 else "SWIPE_RIGHT"

    return None


def process_gesture(state, gesture, landmarks):
    now = time.time()
    palm_x, palm_y = get_palm_center(landmarks)

    action = None
    action_label = None

    # 주먹 쥐기 → 재생/일시정지 (open→fist 전환 시에만)
    current_fist = (gesture == "FIST")
    if current_fist and state.prev_fist_state is False:
        if now - state.last_action_time > state.cooldown:
            pyautogui.press("space")
            action = "space"
            action_label = "PLAY/PAUSE"
            state.last_action_time = now

    state.prev_fist_state = current_fist

    # 손바닥 펴고 스와이프 → 앞/뒤 감기
    if gesture == "OPEN":
        swipe = detect_swipe(state, palm_x)
        if swipe == "SWIPE_RIGHT":
            pyautogui.press("right")
            action = "right"
            action_label = "FORWARD >>>"
        elif swipe == "SWIPE_LEFT":
            pyautogui.press("left")
            action = "left"
            action_label = "<<< REWIND"

    # 검지 포인팅 → 위치에 따라 볼륨
    if gesture == "POINT" and now - state.last_action_time > state.cooldown:
        tip_y = landmarks[8].y
        if tip_y < 0.3:
            pyautogui.press("up")
            action = "up"
            action_label = "VOLUME UP"
            state.last_action_time = now
        elif tip_y > 0.7:
            pyautogui.press("down")
            action = "down"
            action_label = "VOLUME DOWN"
            state.last_action_time = now

    # 피스 → 전체화면
    if gesture == "PEACE":
        if state.prev_gesture != "PEACE" and now - state.last_action_time > state.cooldown:
            pyautogui.press("f")
            action = "f"
            action_label = "FULLSCREEN"
            state.last_action_time = now

    state.prev_gesture = gesture
    return action_label


def draw_hud(frame, gesture, action_label, landmarks, state):
    h, w, _ = frame.shape

    # 제스처 이름
    gesture_icons = {
        "FIST": "FIST",
        "OPEN": "OPEN PALM",
        "POINT": "POINTING",
        "PEACE": "PEACE",
        "OTHER": "---",
    }
    label = gesture_icons.get(gesture, gesture)
    cv2.putText(frame, f"Gesture: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 액션 피드백
    if action_label:
        cv2.putText(frame, f"ACTION: {action_label}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # 도움말
    help_lines = [
        "Fist: Play/Pause",
        "Swipe L/R: Rewind/Forward",
        "Point Up/Down: Vol +/-",
        "Peace: Fullscreen",
        "Q/ESC: Quit",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(frame, line, (w - 280, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    print("=" * 50)
    print("Hand Gesture Video Controller")
    print("=" * 50)
    print("MacBook 카메라로 손 제스처를 인식합니다.")
    print()
    print("  주먹 쥐기      → 재생/일시정지 (Space)")
    print("  손 펴고 좌/우   → 뒤로/앞으로 감기")
    print("  검지 위/아래    → 볼륨 업/다운")
    print("  피스 사인       → 전체화면 (F)")
    print()
    print("종료: q 또는 ESC")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: 카메라를 열 수 없습니다.")
        print("시스템 설정 > 개인정보 보호 > 카메라 에서 터미널 접근을 허용하세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = GestureState()
    action_label = None
    action_display_until = 0

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            gesture = None
            now = time.time()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    lm = hand_landmarks.landmark
                    gesture = classify_gesture(lm)
                    result = process_gesture(state, gesture, lm)
                    if result:
                        action_label = result
                        action_display_until = now + 1.5

            current_action = action_label if now < action_display_until else None
            draw_hud(frame, gesture or "---", current_action, None, state)

            cv2.imshow("Hand Gesture Controller", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == "__main__":
    main()
