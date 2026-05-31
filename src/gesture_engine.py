"""
GestureEngine v4 (Simplified B)
단 2가지 손 제스처만 사용. 줌/스타일은 마우스+키보드.

  OPEN (손 펴기) + 이동 → 모델 회전
  FIST (주먹)           → 일시정지
  그 외                 → 무시 (겹침 방지)
"""

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class GestureResult:
    gesture: str = "IDLE"
    is_playing: bool = True
    rotation_delta: tuple = (0.0, 0.0)


class SmoothValue:
    def __init__(self, window=5):
        self.buffer = deque(maxlen=window)

    def update(self, val):
        self.buffer.append(val)
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class GestureEngine:
    def __init__(self):
        self.state = "IDLE"
        self.is_playing = True

        self.prev_palm_x = None
        self.prev_palm_y = None

        self.smooth_dx = SmoothValue(window=4)
        self.smooth_dy = SmoothValue(window=4)

        self.last_fist_time = 0
        self.fist_cooldown = 0.5

        self.rotation_sensitivity = 80.0

    def _finger_extended(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def _classify_hand(self, lm):
        fingers = [
            self._finger_extended(lm, 8, 6),
            self._finger_extended(lm, 12, 10),
            self._finger_extended(lm, 16, 14),
            self._finger_extended(lm, 20, 18),
        ]
        extended = sum(fingers)

        if extended <= 1:
            return "FIST"
        if extended >= 3:
            return "OPEN"
        return "OTHER"

    def _palm_center(self, lm):
        x = (lm[0].x + lm[5].x + lm[17].x) / 3
        y = (lm[0].y + lm[5].y + lm[17].y) / 3
        return x, y

    def process(self, landmarks):
        if landmarks is None:
            self.state = "IDLE"
            self.prev_palm_x = None
            self.prev_palm_y = None
            self.smooth_dx.reset()
            self.smooth_dy.reset()
            return GestureResult(is_playing=self.is_playing)

        lm = landmarks
        gesture = self._classify_hand(lm)
        now = time.time()
        result = GestureResult(gesture=gesture, is_playing=self.is_playing)

        if gesture == "OPEN":
            palm_x, palm_y = self._palm_center(lm)

            if self.state == "ROTATING" and self.prev_palm_x is not None:
                raw_dx = palm_x - self.prev_palm_x
                raw_dy = palm_y - self.prev_palm_y

                dx = self.smooth_dx.update(raw_dx)
                dy = self.smooth_dy.update(raw_dy)

                d_az = dx * self.rotation_sensitivity
                d_el = -dy * self.rotation_sensitivity
                result.rotation_delta = (d_az, d_el)

            self.state = "ROTATING"
            self.prev_palm_x = palm_x
            self.prev_palm_y = palm_y
            self.is_playing = True
            result.is_playing = True

        elif gesture == "FIST":
            if self.state != "PAUSED":
                if now - self.last_fist_time > self.fist_cooldown:
                    self.is_playing = False
                    self.last_fist_time = now
            self.state = "PAUSED"
            self.prev_palm_x = None
            self.prev_palm_y = None
            self.smooth_dx.reset()
            self.smooth_dy.reset()
            result.is_playing = False

        else:
            self.prev_palm_x = None
            self.prev_palm_y = None
            self.smooth_dx.reset()
            self.smooth_dy.reset()

        return result
