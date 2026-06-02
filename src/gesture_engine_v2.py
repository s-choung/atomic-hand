import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GestureResult:
    gesture: str = "IDLE"
    is_playing: bool = True
    rotation_delta: tuple = (0.0, 0.0)
    pan_delta: tuple = (0.0, 0.0)
    zoom_delta: float = 0.0
    swipe: str = ""


class SmoothValue:
    def __init__(self, window=5):
        self.buffer = deque(maxlen=window)

    def update(self, val):
        self.buffer.append(val)
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class SwipeDetector:
    def __init__(self, threshold=0.05, time_window=0.35):
        self.threshold = threshold
        self.time_window = time_window
        self.history = deque(maxlen=12)
        self.last_swipe_time = 0
        self.cooldown = 0.45

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
        if abs(dx) > self.threshold and abs(velocity) > 0.13:
            self.last_swipe_time = timestamp
            self.history.clear()
            return "LEFT" if dx > 0 else "RIGHT"
        return ""

    def reset(self):
        self.history.clear()


class GestureEngineV2:
    def __init__(self):
        self.state = "IDLE"
        self.is_playing = True
        self.prev_palm_x = None
        self.prev_palm_y = None
        self.prev_fist_x = None
        self.prev_fist_y = None
        self.prev_hand_angle = None
        self.prev_two_hand_dist = None
        self.smooth_dx = SmoothValue(window=1)
        self.smooth_dy = SmoothValue(window=1)
        self.smooth_angle = SmoothValue(window=1)
        self.smooth_zoom = SmoothValue(window=1)
        self.swipe_detector = SwipeDetector()
        self.rotation_sensitivity = 500.0
        self.zoom_sensitivity = 50.0
        self.wrist_rotation_sensitivity = 120.0

    def _finger_extended(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def _thumb_extended(self, lm):
        return abs(lm[4].x - lm[2].x) > abs(lm[3].x - lm[2].x) * 0.8

    def _classify_hand(self, lm):
        fingers = [
            self._thumb_extended(lm),
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

    def _is_thumbs_up(self, lm):
        """Thumb sticking straight up, the other four fingers folded. Used as a
        deliberate 'save snapshot' trigger that won't collide with fist/open."""
        folded = (not self._finger_extended(lm, 8, 6)
                  and not self._finger_extended(lm, 12, 10)
                  and not self._finger_extended(lm, 16, 14)
                  and not self._finger_extended(lm, 20, 18))
        if not folded:
            return False
        # y grows downward: thumb tip must be clearly above IP/MCP, wrist and index MCP
        return (lm[4].y < lm[3].y < lm[2].y
                and lm[4].y < lm[5].y - 0.04
                and lm[4].y < lm[0].y - 0.06)

    def _palm_center(self, lm):
        x = (lm[0].x + lm[5].x + lm[17].x) / 3
        y = (lm[0].y + lm[5].y + lm[17].y) / 3
        return x, y

    def _hand_size(self, lm):
        xs = [lm[i].x for i in (0, 5, 9, 13, 17)]
        ys = [lm[i].y for i in (0, 5, 9, 13, 17)]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def _hand_angle(self, lm):
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return math.atan2(dx, -dy)

    def _two_hand_distance(self, lm1, lm2):
        c1 = self._palm_center(lm1)
        c2 = self._palm_center(lm2)
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def process(self, hand_landmarks_list):
        if not hand_landmarks_list:
            self._reset_state()
            return GestureResult(gesture="IDLE", is_playing=self.is_playing)

        valid_hands = [h for h in hand_landmarks_list if self._hand_size(h.landmark) > 0.08]
        if not valid_hands:
            self._reset_state()
            return GestureResult(gesture="IDLE", is_playing=self.is_playing)
        hand_landmarks_list = valid_hands

        now = time.time()

        if len(hand_landmarks_list) >= 2:
            lm1 = hand_landmarks_list[0].landmark
            lm2 = hand_landmarks_list[1].landmark
            g1 = self._classify_hand(lm1)
            g2 = self._classify_hand(lm2)

            if g1 == "FIST" or g2 == "FIST":
                self.prev_two_hand_dist = None
                self.state = "PAUSED"
                self._reset_rotation_state()
                self.is_playing = False
                return GestureResult(gesture="FIST", is_playing=False)

            dist = self._two_hand_distance(lm1, lm2)
            if dist < 0.15:
                self.prev_two_hand_dist = None
            else:
                zoom_delta = 0.0
                if self.prev_two_hand_dist is not None:
                    raw_delta = dist - self.prev_two_hand_dist
                    zoom_delta = self.smooth_zoom.update(raw_delta) * self.zoom_sensitivity
                self.prev_two_hand_dist = dist
                self.state = "ZOOM"
                self._reset_rotation_state()
                return GestureResult(gesture="ZOOM", is_playing=True, zoom_delta=zoom_delta)

        lm = hand_landmarks_list[0].landmark

        if self._is_thumbs_up(lm):
            self.state = "PAUSED"
            self._reset_rotation_state()
            return GestureResult(gesture="THUMBS_UP", is_playing=False)

        # Swipe detection runs for any single hand, independent of open/fist,
        # so a quick horizontal flick reliably switches structures even when the
        # finger classification flickers mid-swipe.
        palm_x, palm_y = self._palm_center(lm)
        self.swipe_detector.update(palm_x, now)
        swipe = self.swipe_detector.detect(now)
        if swipe:
            self._reset_rotation_state()
            return GestureResult(gesture=f"SWIPE_{swipe}", swipe=swipe,
                                 is_playing=self.is_playing)

        gesture = self._classify_hand(lm)
        result = GestureResult(gesture=gesture, is_playing=self.is_playing)

        if gesture == "OPEN":
            angle = self._hand_angle(lm)

            if self.state == "ROTATING" and self.prev_palm_x is not None:
                raw_dx = palm_x - self.prev_palm_x
                raw_dy = palm_y - self.prev_palm_y
                dx = self.smooth_dx.update(raw_dx)
                dy = self.smooth_dy.update(raw_dy)

                wrist_delta = 0.0
                if self.prev_hand_angle is not None:
                    raw_angle = angle - self.prev_hand_angle
                    if abs(raw_angle) < 0.5:
                        wrist_delta = self.smooth_angle.update(raw_angle) * self.wrist_rotation_sensitivity

                result.rotation_delta = (
                    dx * self.rotation_sensitivity + wrist_delta,
                    -dy * self.rotation_sensitivity,
                )

            self.state = "ROTATING"
            self.prev_palm_x = palm_x
            self.prev_palm_y = palm_y
            self.prev_hand_angle = angle
            self._reset_pan_state()
            self.is_playing = True
            result.is_playing = True

        elif gesture == "FIST":
            self.state = "PAUSED"
            self._reset_rotation_state()
            result.gesture = "FIST"

        else:
            self._reset_rotation_state()

        return result

    def _reset_state(self):
        self._reset_rotation_state()
        self._reset_pan_state()
        self.prev_two_hand_dist = None
        self.smooth_zoom.reset()
        self.swipe_detector.reset()

    def _reset_rotation_state(self):
        self.prev_palm_x = None
        self.prev_palm_y = None
        self.prev_hand_angle = None
        self.smooth_dx.reset()
        self.smooth_dy.reset()
        self.smooth_angle.reset()

    def _reset_pan_state(self):
        self.prev_fist_x = None
        self.prev_fist_y = None
