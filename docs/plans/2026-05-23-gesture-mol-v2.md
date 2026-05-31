# GestureMol v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** OpenCV imshow 기반 GestureMol v1을 PySide6 기반 v2로 전면 개편. 두 손 제스처, 깔끔한 UI, 갤러리, 드래그&드롭 지원.

**Architecture:** PySide6 QMainWindow으로 전환하여 macOS 네이티브 윈도우 컨트롤 확보. 메인 뷰 영역은 QLabel 기반 이미지 디스플레이 + QPainter 오버레이. 하단 갤러리는 QScrollArea + 썸네일 위젯. GestureEngine을 v2로 확장하여 2-hand zoom + swipe 감지. OVITO OpenGLRenderer 유지하되 설정 대폭 개선.

**Tech Stack:** Python 3.11, PySide6 (OVITO 번들), OVITO 3.13+, MediaPipe 0.10.14, OpenCV (카메라 캡처만), numpy

**Base Directory:** `/Users/sean/Library/CloudStorage/GoogleDrive-wjdtjrgus9967@gmail.com/My Drive/Research_2026/playground/101_gesture_mol_v2/`

---

## Dependency Graph

```
Task 1 (Qt Window Shell)
  ├── Task 2 (Mouse Controls) ──┐
  ├── Task 3 (Camera + MediaPipe) ── Task 4 (Gesture Engine v2)
  ├── Task 5 (Status Bar UI)    ──┤
  ├── Task 6 (Gallery)          ──┤── Task 9 (Integration + Polish)
  ├── Task 7 (Drag & Drop)     ──┤
  └── Task 8 (Rendering Quality) ─┘
```

**병렬 가능:** Tasks 2, 3, 5, 6, 7, 8은 Task 1 완료 후 독립 병렬 실행 가능.
**순차 필수:** Task 3 → Task 4 (MediaPipe 통합 후 제스처 엔진 확장)

---

## Task 1: PySide6 Main Window Shell

**Files:**
- Create: `src/app_window.py` (메인 윈도우 — gesture_mol_viewer.py 대체)
- Modify: `src/molecule_renderer.py` (변경 없음, 호환 확인만)

**Goal:** OpenCV imshow를 PySide6 QMainWindow으로 교체. macOS traffic light 버튼 동작 확인.

**Step 1: 기본 QMainWindow 생성**

```python
"""
GestureMol v2 — PySide6 기반 메인 윈도우
"""
import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecule_renderer import MoleculeRenderer

DEFAULT_FILES = [
    "/Users/sean/Library/CloudStorage/GoogleDrive-wjdtjrgus9967@gmail.com/My Drive/Research_2026/playground/97_understand_ark/ARKdistillationV2_repo/OMat24_and_HTS/src/fairchem/data/odac/force_field/FF_calcs/CO2_example/frame.cif",
]


class MolView(QLabel):
    """OVITO 렌더 결과를 표시하는 메인 뷰 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #1a1a2e;")

    def update_frame(self, bgr_array):
        """numpy BGR array → QPixmap 변환 후 표시."""
        h, w, ch = bgr_array.shape
        rgb = bgr_array[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))


class GestureMolWindow(QMainWindow):
    def __init__(self, file_list):
        super().__init__()
        self.setWindowTitle("GestureMol")
        self.resize(1100, 850)

        self.file_list = file_list
        self.file_idx = 0
        self.renderer = MoleculeRenderer(file_list[0])

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.mol_view = MolView()
        layout.addWidget(self.mol_view)

        # Render timer (60fps target)
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)  # ~60fps

    def _tick(self):
        w = self.mol_view.width()
        h = self.mol_view.height()
        if w < 10 or h < 10:
            return
        frame = self.renderer.render(w, h)
        self.mol_view.update_frame(frame)

    def closeEvent(self, event):
        self.timer.stop()
        self.renderer.cleanup()
        event.accept()


def main():
    file_list = sys.argv[1:] if len(sys.argv) > 1 else [
        f for f in DEFAULT_FILES if os.path.exists(f)
    ]
    if not file_list:
        print("No structure files found. Pass file paths as arguments.")
        return

    app = QApplication(sys.argv)
    window = GestureMolWindow(file_list)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

**Step 2: 실행 및 확인**

```bash
conda run -n base python src/app_window.py
```

Expected:
- macOS 윈도우 열림 (traffic light 버튼 ✅)
- 빨간 X 버튼으로 창 닫기 ✅
- OVITO 렌더링된 분자 구조 표시 ✅

---

## Task 2: Mouse Controls (줌인 버그 수정 포함)

**Files:**
- Modify: `src/app_window.py` (MolView 클래스에 마우스 이벤트 추가)

**Goal:** Qt 네이티브 마우스 이벤트로 드래그=회전, 스크롤=줌. macOS 트랙패드 양방향 줌 동작 보장.

**Step 1: MolView에 마우스 이벤트 핸들러 추가**

`MolView` 클래스에 추가:

```python
def __init__(self, parent=None):
    super().__init__(parent)
    # ... existing code ...
    self.renderer = None  # set by parent
    self._dragging = False
    self._last_pos = None

def mousePressEvent(self, event):
    if event.button() == Qt.LeftButton:
        self._dragging = True
        self._last_pos = event.position()

def mouseReleaseEvent(self, event):
    if event.button() == Qt.LeftButton:
        self._dragging = False
        self._last_pos = None

def mouseMoveEvent(self, event):
    if self._dragging and self._last_pos and self.renderer:
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self.renderer.rotate(dx * 0.5, -dy * 0.5)
        self._last_pos = pos

def wheelEvent(self, event):
    if self.renderer:
        delta = event.angleDelta().y()
        # angleDelta > 0 = scroll up = zoom in (closer)
        # angleDelta < 0 = scroll down = zoom out (farther)
        if delta > 0:
            self.renderer.zoom(0.92)  # smaller distance = zoom in
        elif delta < 0:
            self.renderer.zoom(1.08)  # larger distance = zoom out
```

**v1 줌인 버그 원인:** OpenCV `EVENT_MOUSEWHEEL`의 `flags` 값이 macOS 트랙패드에서 제대로 전달 안 됨. Qt `wheelEvent.angleDelta().y()`는 macOS에서 정상 동작.

**Step 2: GestureMolWindow에서 renderer 연결**

```python
# __init__에서:
self.mol_view.renderer = self.renderer
```

**Step 3: 실행 확인**

- 마우스 드래그 → 회전 ✅
- 트랙패드/마우스 스크롤 위 → 줌인 ✅
- 트랙패드/마우스 스크롤 아래 → 줌아웃 ✅

---

## Task 3: Camera Feed + MediaPipe 통합

**Files:**
- Modify: `src/app_window.py` (카메라 캡처 + MediaPipe + 오버레이)

**Goal:** 카메라 피드를 OVITO 렌더 위에 오버레이. 얼굴 크기를 v1(20%)에서 30%로 확대.

**Step 1: 카메라 + MediaPipe 초기화**

`GestureMolWindow.__init__`에 추가:

```python
import cv2
import mediapipe as mp

# Camera
self.cam = cv2.VideoCapture(0)
self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe
self.mp_hands = mp.solutions.hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
    max_num_hands=2,  # v2: 두 손 지원
)
self.mp_drawing = mp.solutions.drawing_utils
self.mp_hand_connections = mp.solutions.hands.HAND_CONNECTIONS

# Gesture engine
from gesture_engine import GestureEngine
self.gesture_engine = GestureEngine()

self.cam_frame = None
self.gesture_name = "IDLE"
```

**Step 2: _tick()에서 카메라 프레임 처리**

```python
def _tick(self):
    # Camera capture
    ret, cam_frame = self.cam.read()
    if ret:
        cam_frame = cv2.flip(cam_frame, 1)
        rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.mp_hands.process(rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(cam_frame, hlm, self.mp_hand_connections)

        # Process gesture (Task 4에서 확장)
        landmarks = None
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
        result = self.gesture_engine.process(landmarks)

        d_az, d_el = result.rotation_delta
        if d_az != 0 or d_el != 0:
            self.renderer.rotate(d_az, d_el)

        self.cam_frame = cam_frame
        self.gesture_name = result.gesture

    # Render molecule
    w = self.mol_view.width()
    h = self.mol_view.height()
    if w < 10 or h < 10:
        return
    mol_frame = self.renderer.render(w, h)

    # Camera overlay (30% height — v1은 20%)
    if self.cam_frame is not None:
        self._draw_camera_overlay(mol_frame)

    self.mol_view.update_frame(mol_frame)
```

**Step 3: 카메라 오버레이 (얼굴 30% 크기)**

```python
def _draw_camera_overlay(self, mol_frame):
    mol_h, mol_w = mol_frame.shape[:2]
    cam_h, cam_w = self.cam_frame.shape[:2]

    # v2: 30% height (v1은 20%)
    display_h = int(mol_h * 0.30)
    display_w = int(cam_w * display_h / cam_h)
    cam_small = cv2.resize(self.cam_frame, (display_w, display_h))

    # 우측 하단 (갤러리 위)
    margin = 12
    x1 = mol_w - display_w - margin
    y1 = mol_h - display_h - margin
    x2, y2 = x1 + display_w, y1 + display_h

    if y1 >= 0 and x1 >= 0:
        roi = mol_frame[y1:y2, x1:x2]
        if roi.shape == cam_small.shape:
            blended = cv2.addWeighted(cam_small, 0.5, roi, 0.5, 0)
            mol_frame[y1:y2, x1:x2] = blended

        # 테두리
        border_color = {"OPEN": (50, 200, 255), "FIST": (80, 80, 255)}.get(
            self.gesture_name, (80, 80, 80)
        )
        cv2.rectangle(mol_frame, (x1-1, y1-1), (x2+1, y2+1), border_color, 2)
```

**Step 4: cleanup 추가**

```python
def closeEvent(self, event):
    self.timer.stop()
    self.renderer.cleanup()
    if self.cam:
        self.cam.release()
    self.mp_hands.close()
    event.accept()
```

---

## Task 4: Gesture Engine v2 — Two-Hand Zoom + Swipe

**Files:**
- Create: `src/gesture_engine_v2.py` (기존 gesture_engine.py 대체)

**Goal:** 양손 벌리기=줌인, 오므리기=줌아웃. 한 손 빠른 스와이프=다음/이전 구조.

**Step 1: GestureResult 확장**

```python
@dataclass
class GestureResult:
    gesture: str = "IDLE"           # IDLE, ROTATE, ZOOM, SWIPE_LEFT, SWIPE_RIGHT, PAUSED
    is_playing: bool = True
    rotation_delta: tuple = (0.0, 0.0)
    zoom_delta: float = 0.0         # < 0 = zoom in, > 0 = zoom out
    swipe: str = ""                 # "LEFT", "RIGHT", ""
```

**Step 2: Two-hand zoom detection**

```python
def _palm_center(self, lm):
    x = (lm[0].x + lm[5].x + lm[17].x) / 3
    y = (lm[0].y + lm[5].y + lm[17].y) / 3
    return x, y

def _two_hand_distance(self, lm1, lm2):
    """두 손 중심 간 거리."""
    c1 = self._palm_center(lm1)
    c2 = self._palm_center(lm2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
```

**Step 3: Swipe detection**

```python
class SwipeDetector:
    def __init__(self, threshold=0.15, time_window=0.3):
        self.threshold = threshold      # 화면 비율 기준 이동 거리
        self.time_window = time_window  # 초
        self.history = deque(maxlen=10)
        self.last_swipe_time = 0
        self.cooldown = 0.8  # 스와이프 간 쿨다운

    def update(self, palm_x, timestamp):
        self.history.append((palm_x, timestamp))

    def detect(self, timestamp):
        if timestamp - self.last_swipe_time < self.cooldown:
            return ""
        if len(self.history) < 3:
            return ""

        recent = [(x, t) for x, t in self.history if timestamp - t < self.time_window]
        if len(recent) < 2:
            return ""

        dx = recent[-1][0] - recent[0][0]
        dt = recent[-1][1] - recent[0][1]
        if dt < 0.05:
            return ""

        velocity = dx / dt
        if abs(dx) > self.threshold and abs(velocity) > 0.5:
            self.last_swipe_time = timestamp
            self.history.clear()
            return "LEFT" if dx > 0 else "RIGHT"  # 미러: 화면 좌로 = 오른손 좌로
        return ""

    def reset(self):
        self.history.clear()
```

**Step 4: 전체 GestureEngineV2**

```python
class GestureEngineV2:
    def __init__(self):
        self.state = "IDLE"
        self.is_playing = True
        self.prev_palm_x = None
        self.prev_palm_y = None
        self.prev_two_hand_dist = None
        self.smooth_dx = SmoothValue(window=4)
        self.smooth_dy = SmoothValue(window=4)
        self.smooth_zoom = SmoothValue(window=3)
        self.swipe_detector = SwipeDetector()
        self.rotation_sensitivity = 80.0
        self.zoom_sensitivity = 3.0

    def process(self, hand_landmarks_list):
        """
        hand_landmarks_list: list of hand landmark sets (0, 1, or 2 hands)
        Returns: GestureResult
        """
        if not hand_landmarks_list:
            self._reset_state()
            return GestureResult(gesture="IDLE", is_playing=self.is_playing)

        now = time.time()

        # === 두 손: 줌 ===
        if len(hand_landmarks_list) >= 2:
            lm1 = hand_landmarks_list[0].landmark
            lm2 = hand_landmarks_list[1].landmark

            g1 = self._classify_hand(lm1)
            g2 = self._classify_hand(lm2)

            if g1 == "OPEN" and g2 == "OPEN":
                dist = self._two_hand_distance(lm1, lm2)

                zoom_delta = 0.0
                if self.prev_two_hand_dist is not None:
                    raw_delta = dist - self.prev_two_hand_dist
                    zoom_delta = self.smooth_zoom.update(raw_delta) * self.zoom_sensitivity

                self.prev_two_hand_dist = dist
                self.state = "ZOOM"
                self._reset_rotation_state()
                return GestureResult(
                    gesture="ZOOM", is_playing=True, zoom_delta=zoom_delta
                )

            self.prev_two_hand_dist = None

        # === 한 손 ===
        lm = hand_landmarks_list[0].landmark
        gesture = self._classify_hand(lm)
        result = GestureResult(gesture=gesture, is_playing=self.is_playing)

        if gesture == "OPEN":
            palm_x, palm_y = self._palm_center(lm)
            self.swipe_detector.update(palm_x, now)

            # 스와이프 체크 (회전보다 우선)
            swipe = self.swipe_detector.detect(now)
            if swipe:
                self._reset_rotation_state()
                result.gesture = f"SWIPE_{swipe}"
                result.swipe = swipe
                return result

            # 회전
            if self.state == "ROTATING" and self.prev_palm_x is not None:
                raw_dx = palm_x - self.prev_palm_x
                raw_dy = palm_y - self.prev_palm_y
                dx = self.smooth_dx.update(raw_dx)
                dy = self.smooth_dy.update(raw_dy)
                result.rotation_delta = (
                    dx * self.rotation_sensitivity,
                    -dy * self.rotation_sensitivity,
                )

            self.state = "ROTATING"
            self.prev_palm_x = palm_x
            self.prev_palm_y = palm_y
            self.is_playing = True
            result.is_playing = True

        elif gesture == "FIST":
            self.state = "PAUSED"
            self._reset_rotation_state()
            self.is_playing = False
            result.is_playing = False

        else:
            self._reset_rotation_state()

        return result

    def _reset_state(self):
        self._reset_rotation_state()
        self.prev_two_hand_dist = None
        self.smooth_zoom.reset()
        self.swipe_detector.reset()

    def _reset_rotation_state(self):
        self.prev_palm_x = None
        self.prev_palm_y = None
        self.smooth_dx.reset()
        self.smooth_dy.reset()
```

**Step 5: app_window.py에서 GestureEngineV2 통합**

`_tick()`에서 변경:

```python
# 기존: landmarks = results.multi_hand_landmarks[0].landmark
# 변경: 전체 hand list 전달
hand_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
result = self.gesture_engine.process(hand_list)

# Zoom 처리
if result.zoom_delta != 0:
    factor = 1.0 - result.zoom_delta * 0.1  # delta > 0 = 벌림 = zoom in
    self.renderer.zoom(max(0.5, min(2.0, factor)))

# Swipe 처리
if result.swipe == "LEFT":
    self._next_structure()
elif result.swipe == "RIGHT":
    self._prev_structure()
```

---

## Task 5: Status Bar UI 리디자인

**Files:**
- Modify: `src/app_window.py` (상태바 QPainter 오버레이)

**Goal:** OpenAI classic 스타일 — Inter 폰트, 반투명 배경, 깔끔한 pill badge. 현재 OpenCV putText의 투박한 텍스트 대체.

**디자인 스펙:**
- 상단 좌측: 파일명 (Inter/SF Pro, 12px, white, 60% opacity)
- 상단 우측: atom count pill (배경 #ffffff15, 둥근 모서리)
- 전체 상단: 반투명 그라데이션 (위쪽 검정 40% → 투명)
- 하단 gesture 상태: 우측 카메라 옆에 작은 dot indicator

**Step 1: QPainter 기반 상태바**

`MolView.paintEvent` 오버라이드 또는 numpy 이미지에 OpenCV로 그리되, 더 세련되게:

```python
def _draw_status_overlay(self, frame):
    """OpenAI-classic 스타일 상태바 오버레이."""
    h, w = frame.shape[:2]

    # 상단 그라데이션 배경 (48px)
    for y in range(48):
        alpha = 0.6 * (1.0 - y / 48.0)
        frame[y, :] = (frame[y, :].astype(float) * (1 - alpha)).astype(np.uint8)

    # 파일명 (좌측)
    filename = os.path.basename(self.renderer.filepath)
    cv2.putText(frame, filename, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Atom count pill (우측)
    atom_text = f"{self.renderer.num_atoms} atoms"
    (tw, th), _ = cv2.getTextSize(atom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    pill_x = w - tw - 32
    pill_y = 14

    # Pill 배경 (둥근 사각형)
    overlay = frame.copy()
    cv2.rectangle(overlay, (pill_x - 8, pill_y - 6), (pill_x + tw + 8, pill_y + th + 6),
                  (60, 60, 70), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, atom_text, (pill_x, pill_y + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 210), 1, cv2.LINE_AA)

    # FPS (우측, pill 옆)
    fps_text = f"{self.fps:.0f} fps"
    (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    cv2.putText(frame, fps_text, (pill_x - fw - 20, pill_y + fh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 130), 1, cv2.LINE_AA)

    # 구조 인덱스 (여러 파일일 때)
    if len(self.file_list) > 1:
        idx_text = f"{self.file_idx + 1}/{len(self.file_list)}"
        cv2.putText(frame, idx_text, (16, 28 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 170), 1, cv2.LINE_AA)
```

**핵심 차이 vs v1:**
- `cv2.LINE_AA` 안티앨리어싱 적용
- 그라데이션 배경 (단색 사각형 → 자연스러운 페이드)
- Pill badge로 atom count 표시
- 불필요한 "PLAY/PAUSE", 스타일 이름 등 제거
- 폰트 크기 축소, 여백 확보

---

## Task 6: Bottom Gallery

**Files:**
- Modify: `src/app_window.py` (갤러리 위젯 추가)

**Goal:** 하단에 구조 파일 썸네일 horizontal 스크롤. 방향키로 선택, 현재 파일 하이라이트.

**Step 1: 갤러리를 OpenCV 오버레이로 구현**

PySide6 위젯 분리보다 OVITO 렌더 이미지 하단에 직접 그리는 게 간단하고 일관적:

```python
def _draw_gallery(self, frame):
    """하단 갤러리 — 구조 썸네일 horizontal strip."""
    if len(self.file_list) <= 1:
        return

    h, w = frame.shape[:2]
    gallery_h = 64
    gallery_y = h - gallery_h

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, gallery_y), (w, h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 썸네일 배치
    thumb_w = 80
    thumb_h = 50
    padding = 8
    total_width = len(self.file_list) * (thumb_w + padding) - padding
    start_x = max(0, (w - total_width) // 2)

    for i, filepath in enumerate(self.file_list):
        x = start_x + i * (thumb_w + padding)
        y = gallery_y + (gallery_h - thumb_h) // 2

        if x + thumb_w > w:
            break

        # 썸네일: 캐시된 미리 렌더링 이미지 사용
        if i < len(self.thumbnails):
            thumb = self.thumbnails[i]
            if thumb is not None:
                resized = cv2.resize(thumb, (thumb_w, thumb_h))
                frame[y:y+thumb_h, x:x+thumb_w] = resized

        # 현재 선택 하이라이트
        if i == self.file_idx:
            cv2.rectangle(frame, (x-2, y-2), (x+thumb_w+2, y+thumb_h+2),
                          (50, 200, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+thumb_w, y+thumb_h),
                          (60, 60, 70), 1)

        # 파일명 (짧게)
        name = os.path.splitext(os.path.basename(filepath))[0][:10]
        cv2.putText(frame, name, (x + 2, y + thumb_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 140, 150), 1, cv2.LINE_AA)
```

**Step 2: 썸네일 캐시 생성**

`__init__`에서:

```python
self.thumbnails = []
for fp in self.file_list:
    try:
        r = MoleculeRenderer(fp)
        thumb = r.render(160, 100)
        r.cleanup()
        self.thumbnails.append(thumb)
    except Exception:
        self.thumbnails.append(None)
```

**Step 3: 방향키 핸들러**

```python
def keyPressEvent(self, event):
    if event.key() == Qt.Key_Right:
        self._next_structure()
    elif event.key() == Qt.Key_Left:
        self._prev_structure()
    elif event.key() == Qt.Key_Escape:
        self.close()

def _next_structure(self):
    if len(self.file_list) > 1:
        self.file_idx = (self.file_idx + 1) % len(self.file_list)
        self._load_current_structure()

def _prev_structure(self):
    if len(self.file_list) > 1:
        self.file_idx = (self.file_idx - 1) % len(self.file_list)
        self._load_current_structure()

def _load_current_structure(self):
    self.renderer.cleanup()
    self.renderer = MoleculeRenderer(self.file_list[self.file_idx])
    self.mol_view.renderer = self.renderer
```

---

## Task 7: Drag & Drop

**Files:**
- Modify: `src/app_window.py` (드래그&드롭 이벤트)

**Goal:** XYZ, CIF, PDB 등 구조 파일을 윈도우에 드래그&드롭하면 목록에 추가하고 바로 표시.

**Step 1: QMainWindow에 드롭 활성화**

```python
class GestureMolWindow(QMainWindow):
    def __init__(self, file_list):
        super().__init__()
        self.setAcceptDrops(True)
        # ... rest of init ...

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.xyz', '.cif', '.pdb', '.poscar', '.vasp', '.lammps'):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        supported = ('.xyz', '.cif', '.pdb', '.poscar', '.vasp', '.lammps')
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in supported and path not in self.file_list:
                self.file_list.append(path)

                # 썸네일 생성
                try:
                    r = MoleculeRenderer(path)
                    thumb = r.render(160, 100)
                    r.cleanup()
                    self.thumbnails.append(thumb)
                except Exception:
                    self.thumbnails.append(None)

                # 드롭된 파일로 바로 전환
                self.file_idx = len(self.file_list) - 1
                self._load_current_structure()
```

---

## Task 8: Rendering Quality 개선

**Files:**
- Modify: `src/molecule_renderer.py`

**Goal:** structure-visualizer 스킬 수준의 렌더링 품질. OpenGLRenderer 유지 (실시간 필수).

**Step 1: 렌더러 설정 개선**

```python
def __init__(self, filepath):
    # ... existing pipeline setup ...

    self.renderer = OpenGLRenderer()
    self.renderer.antialiasing_level = 8  # 6 → 8

    # 배경색: 짙은 네이비 (OpenAI 스타일)
    self.vp.background_color = (0.06, 0.06, 0.12)  # #0F0F1F

    # ... rest of init ...
```

**Step 2: 스타일별 설정 개선**

```python
def _apply_style(self):
    data = self.pipeline.compute()
    pvis = data.particles.vis

    # 공통: Sphere 형태 강제
    pvis.shape = pvis.Shape.Sphere

    style = self.STYLES[self.style_idx]

    if style == "ball_stick":
        pvis.radius = 0.5  # 0.4 → 0.5 (좀 더 풍성)
        self.bonds_mod.enabled = True
        if data.particles.bonds is not None:
            bvis = data.particles.bonds.vis
            bvis.width = 0.25  # 0.2 → 0.25
            bvis.enabled = True

    elif style == "spacefill":
        pvis.radius = 1.2  # 1.0 → 1.2
        self.bonds_mod.enabled = False

    elif style == "wireframe":
        pvis.radius = 0.2  # 0.15 → 0.2
        self.bonds_mod.enabled = True
        if data.particles.bonds is not None:
            bvis = data.particles.bonds.vis
            bvis.width = 0.1
            bvis.enabled = True
```

**Step 3: v2에서는 ball_stick만 유지 (단순화)**

키보드 T 키 스타일 전환 제거되므로, 기본 ball_stick 고정. spacefill/wireframe 코드는 남겨두되 UI에서 노출 안 함.

---

## Task 9: Integration + Polish + Keyboard 단순화

**Files:**
- Modify: `src/app_window.py` (최종 통합)
- Remove: v1의 `draw_gesture_guide()` (CONTROLS 패널 제거)

**Goal:** 모든 기능 통합, 불필요한 키보드 제거, 최종 동작 확인.

**Step 1: 키보드 단순화**

v2에서 남기는 키:
- `←` / `→`: 이전/다음 구조
- `ESC`: 종료 (또는 빨간 X 버튼)

제거:
- `T` (스타일 전환) — ball_stick 고정
- `R` (카메라 리셋) — 불필요
- `N` / `P` (다음/이전) — 방향키로 대체
- `Q` (종료) — ESC + 창 닫기로 대체
- `M` — 이미 없음

**Step 2: v1 UI 요소 제거**

- `draw_gesture_guide()` 전체 제거 (좌측 하단 CONTROLS 패널)
- `draw_action_popup()` 제거 (중앙 팝업)
- `GESTURE_INFO` dict 제거

**Step 3: _tick() 최종 순서**

```python
def _tick(self):
    t0 = time.time()

    # 1. Camera + gesture
    self._process_camera()

    # 2. OVITO render
    w, h = self.mol_view.width(), self.mol_view.height()
    if w < 10 or h < 10:
        return
    mol_frame = self.renderer.render(w, h)

    # 3. Overlays (순서 중요)
    self._draw_status_overlay(mol_frame)
    self._draw_gallery(mol_frame)
    self._draw_camera_overlay(mol_frame)  # 카메라는 갤러리 위에

    # 4. Display
    self.mol_view.update_frame(mol_frame)

    # 5. FPS
    dt = time.time() - t0
    self.fps = self.fps * 0.85 + (1.0 / max(dt, 0.001)) * 0.15
```

**Step 4: 최종 실행 확인**

```bash
conda run -n base python src/app_window.py
```

체크리스트:
- [ ] 창 열림 + traffic light 버튼 동작
- [ ] 빨간 X로 앱 종료
- [ ] 한 손 회전
- [ ] 두 손 줌인/줌아웃
- [ ] 빠른 스와이프 → 다음 구조
- [ ] 마우스 드래그 회전
- [ ] 트랙패드 스크롤 줌인/줌아웃 양방향
- [ ] 상태바 깔끔한 UI
- [ ] 하단 갤러리 표시
- [ ] 방향키로 구조 선택
- [ ] XYZ/CIF 드래그&드롭
- [ ] 카메라 피드 30% 크기
- [ ] ESC로 종료

---

## 실행 환경 참고

```bash
# 필수 패키지 (이미 설치돼 있을 것)
conda run -n base pip install mediapipe==0.10.14
# OVITO에 PySide6 번들 포함
# OpenCV는 카메라 캡처 + 이미지 처리용으로 유지
```

## 파일 구조 (v2 최종)

```
101_gesture_mol_v2/
  src/
    app_window.py           ← NEW (메인 앱, gesture_mol_viewer.py 대체)
    gesture_engine_v2.py    ← NEW (2-hand zoom + swipe)
    molecule_renderer.py    ← MODIFIED (렌더링 품질 개선)
    gesture_engine.py       ← KEPT (v1 reference, 사용 안 함)
    gesture_mol_viewer.py   ← KEPT (v1 reference, 사용 안 함)
    viewer.py               ← KEPT (디버그용)
  docs/plans/
    2026-05-23-gesture-mol-v2.md  ← THIS PLAN
  handover.md               ← UPDATE after completion
```
