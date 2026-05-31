import sys
import os
import time
import subprocess
import threading
import numpy as np
import cv2
import mediapipe as mp

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QDockWidget, QGroupBox, QCheckBox, QHBoxLayout, QScrollArea, QPushButton,
    QSlider, QFileDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecule_renderer import MoleculeRenderer
from gesture_engine_v2 import GestureEngineV2

_STRUCT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "structures")
DEFAULT_FILES = [
    os.path.join(_STRUCT_DIR, "C60_buckyball.xyz"),
    os.path.join(_STRUCT_DIR, "ethanol.xyz"),
    os.path.join(_STRUCT_DIR, "pyrrole.xyz"),
    "/Users/sean/Library/CloudStorage/GoogleDrive-wjdtjrgus9967@gmail.com/My Drive/Research_2026/playground/97_understand_ark/ARKdistillationV2_repo/OMat24_and_HTS/src/fairchem/data/odac/force_field/FF_calcs/CO2_example/frame.cif",
]

SUPPORTED_EXTENSIONS = ('.xyz', '.cif', '.pdb', '.poscar', '.vasp', '.lammps')


class CameraReaderThread(threading.Thread):
    """grab() 전용 — 최대 속도로 버퍼를 drain하고 최신 프레임만 보관."""
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.running = True
        self.lock = threading.Lock()
        self._frame = None
        self._frame_id = 0

    def run(self):
        while self.running:
            if not self.cam or not self.cam.isOpened():
                time.sleep(0.05)
                continue
            self.cam.grab()
            ret, frame = self.cam.retrieve()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            with self.lock:
                self._frame = frame
                self._frame_id += 1

    def get_latest(self):
        with self.lock:
            return self._frame, self._frame_id

    def stop(self):
        self.running = False


class MediaPipeThread(threading.Thread):
    """CameraReaderThread에서 최신 프레임 가져와 MediaPipe 추론 + 제스처 처리."""
    def __init__(self, reader, mp_hands, mp_drawing, mp_connections, gesture_engine):
        super().__init__(daemon=True)
        self.reader = reader
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.mp_connections = mp_connections
        self.gesture_engine = gesture_engine
        self.running = True
        self.lock = threading.Lock()
        self.cam_frame = None
        self.gesture_result = None
        self._result_id = 0
        self._last_reader_id = -1

    def run(self):
        while self.running:
            frame, fid = self.reader.get_latest()
            if frame is None or fid == self._last_reader_id:
                time.sleep(0.002)
                continue
            self._last_reader_id = fid

            cam_frame = frame.copy()
            small = cv2.resize(cam_frame, (160, 120))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.mp_hands.process(rgb)

            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(cam_frame, hlm, self.mp_connections)

            hand_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            result = self.gesture_engine.process(hand_list)

            with self.lock:
                self.cam_frame = cam_frame
                self.gesture_result = result
                self._result_id += 1

    def get_latest(self):
        with self.lock:
            return self.cam_frame, self.gesture_result, self._result_id

    def stop(self):
        self.running = False


DARK_SS = """
QDockWidget { color: #ddd; }
QDockWidget::title { background: #1a1a2e; padding: 8px; font-size: 12px; }
QGroupBox { color: #bbb; border: 1px solid #333; border-radius: 5px;
            margin-top: 14px; padding: 14px 8px 10px 8px; font-size: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QCheckBox { color: #ccc; spacing: 8px; font-size: 12px; padding: 3px 0; }
QCheckBox::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:unchecked { border: 1px solid #555; border-radius: 3px; background: #222; }
QCheckBox::indicator:checked { border: 1px solid #4af; border-radius: 3px; background: #4af; }
QScrollArea { border: none; background: #111122; }
QWidget#settingsContainer { background: #111122; }
"""


class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsContainer")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(10)

        # --- Structure Info ---
        info_group = QGroupBox("Structure")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 16, 8, 8)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #ccc; font-size: 11px;")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addWidget(self.fps_label)

        btn_style = ("QPushButton { background: #2a3a5a; color: #ccc; border: 1px solid #444; "
                     "border-radius: 4px; padding: 5px; font-size: 11px; } "
                     "QPushButton:hover { background: #3a4a7a; color: #fff; }")

        self.open_btn = QPushButton("Open File...")
        self.open_btn.setStyleSheet(btn_style)
        info_layout.addWidget(self.open_btn)

        layout.addWidget(info_group)

        # --- Controls (gesture guide) ---
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(6)
        controls_layout.setContentsMargins(8, 16, 8, 8)

        gestures = [
            ("✋", "Move", "Rotate", "#32C8FF"),
            ("👐", "Two Hands", "Zoom", "#50E080"),
            ("✊", "Fist", "Pause", "#5050FF"),
            ("🖱", "Drag", "Rotate", "#AAA"),
            ("⚙", "Scroll", "Zoom", "#AAA"),
            ("👋", "Swipe", "Switch", "#E0A040"),
            ("⌨", "← →", "Switch", "#AAA"),
        ]
        for icon, action, result, color in gestures:
            row = QHBoxLayout()
            row.setSpacing(4)
            icon_label = QLabel(icon)
            icon_label.setFixedWidth(24)
            icon_label.setStyleSheet(f"font-size: 14px; color: {color};")
            desc_label = QLabel(f"{action} → {result}")
            desc_label.setStyleSheet("color: #ccc; font-size: 11px;")
            row.addWidget(icon_label)
            row.addWidget(desc_label)
            row.addStretch()
            controls_layout.addLayout(row)

        self.gesture_status = QLabel("IDLE")
        self.gesture_status.setStyleSheet("color: #666; font-size: 10px; padding-top: 4px;")
        controls_layout.addWidget(self.gesture_status)

        layout.addWidget(controls_group)

        # --- Sensitivity ---
        sens_group = QGroupBox("Sensitivity")
        sens_layout = QVBoxLayout(sens_group)
        sens_layout.setSpacing(4)
        sens_layout.setContentsMargins(8, 16, 8, 8)

        sens_layout.addWidget(QLabel("Rotation"))
        self.rot_slider = QSlider(Qt.Horizontal)
        self.rot_slider.setRange(100, 1000)
        self.rot_slider.setValue(500)
        self.rot_slider.setStyleSheet("QSlider::handle { background: #4af; }")
        sens_layout.addWidget(self.rot_slider)

        sens_layout.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 150)
        self.zoom_slider.setValue(50)
        self.zoom_slider.setStyleSheet("QSlider::handle { background: #4af; }")
        sens_layout.addWidget(self.zoom_slider)

        layout.addWidget(sens_group)

        # --- Display ---
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(8)
        display_layout.setContentsMargins(8, 16, 8, 8)

        self.cb_axes = QCheckBox("Show Axes")
        self.cb_axes.setChecked(True)
        display_layout.addWidget(self.cb_axes)

        self.cb_cell = QCheckBox("Show Unit Cell (PBC)")
        self.cb_cell.setChecked(True)
        display_layout.addWidget(self.cb_cell)

        layout.addWidget(display_group)

        # --- Elements (populated dynamically) ---
        self.elements_group = QGroupBox("Elements")
        self.elements_layout = QVBoxLayout(self.elements_group)
        self.elements_layout.setSpacing(6)
        self.elements_layout.setContentsMargins(8, 16, 8, 8)
        self.element_checkboxes = {}
        layout.addWidget(self.elements_group)

        # --- Export ---
        render_group = QGroupBox("Export")
        render_layout = QVBoxLayout(render_group)
        render_layout.setSpacing(6)
        render_layout.setContentsMargins(8, 16, 8, 8)

        self.render_btn = QPushButton("HQ Render → PNG")
        self.render_btn.setStyleSheet(
            "QPushButton { background: #2a3a5a; color: #ccc; border: 1px solid #4af; "
            "border-radius: 4px; padding: 6px; font-size: 11px; } "
            "QPushButton:hover { background: #3a4a7a; color: #fff; }"
        )
        render_layout.addWidget(self.render_btn)

        self.render_status = QLabel("")
        self.render_status.setStyleSheet("color: #666; font-size: 10px;")
        render_layout.addWidget(self.render_status)

        self.open_render_btn = QPushButton("Open Image")
        self.open_render_btn.setStyleSheet(
            "QPushButton { background: #2a5a3a; color: #ccc; border: 1px solid #4a4; "
            "border-radius: 4px; padding: 5px; font-size: 11px; } "
            "QPushButton:hover { background: #3a7a4a; color: #fff; }"
        )
        self.open_render_btn.hide()
        render_layout.addWidget(self.open_render_btn)

        layout.addWidget(render_group)

        layout.addStretch()

    def populate_elements(self, element_info):
        for cb in self.element_checkboxes.values():
            self.elements_layout.removeWidget(cb)
            cb.deleteLater()
        self.element_checkboxes.clear()

        for elem in element_info:
            name = elem['name']
            count = elem.get('count', 0)
            r, g, b = [int(c * 255) for c in elem['color']]
            cb = QCheckBox(f"  {name}  ({count})")
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox::indicator:checked {{ background: rgb({r},{g},{b}); border: 1px solid rgb({r},{g},{b}); }}"
                f"QCheckBox::indicator:unchecked {{ background: #222; border: 1px solid rgb({min(r+40,255)},{min(g+40,255)},{min(b+40,255)}); }}"
            )
            self.elements_layout.addWidget(cb)
            self.element_checkboxes[name] = cb

    def get_hidden_elements(self):
        return {name for name, cb in self.element_checkboxes.items() if not cb.isChecked()}


class MolView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #0F0F1F;")
        self.renderer = None
        self._dragging = False
        self._last_pos = None
        self.on_view_changed = None

    def update_frame(self, bgr_array):
        h, w, ch = bgr_array.shape
        rgb = bgr_array[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

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
            if self.on_view_changed:
                self.on_view_changed()

    def wheelEvent(self, event):
        if self.renderer:
            delta = event.angleDelta().y()
            if delta > 0:
                self.renderer.zoom(0.85)
            elif delta < 0:
                self.renderer.zoom(1.15)
            if self.on_view_changed:
                self.on_view_changed()


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
    if has_iphone:
        return [1, 0, 2]
    return [0, 1, 2]


class GestureMolWindow(QMainWindow):
    def __init__(self, file_list):
        super().__init__()
        self.setWindowTitle("Atomic Hand v1.2 by Seokhyun Choung")
        self.resize(1300, 850)
        self.setAcceptDrops(True)

        self.file_list = file_list
        self.file_idx = 0
        self.renderer = MoleculeRenderer(file_list[0])
        self.fps = 0.0
        self.render_scale = 0.75
        self._show_axes = True

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.mol_view = MolView()
        self.mol_view.renderer = self.renderer
        self.mol_view.on_view_changed = self._mark_dirty
        layout.addWidget(self.mol_view)

        self._setup_settings_panel()
        self.thumbnails = self._generate_thumbnails()

        cam_indices = find_macbook_camera()
        self.cam = None
        for idx in cam_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cam = cap
                break
            cap.release()
        if self.cam and self.cam.isOpened():
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.gesture_engine = GestureEngineV2()
        self.cam_frame = None
        self.gesture_name = "IDLE"
        self._last_cam_frame_id = -1

        mp_hands = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        )
        mp_drawing = mp.solutions.drawing_utils
        mp_connections = mp.solutions.hands.HAND_CONNECTIONS

        self.cam_reader = CameraReaderThread(self.cam)
        self.cam_reader.start()

        self.cam_thread = MediaPipeThread(
            self.cam_reader, mp_hands, mp_drawing, mp_connections, self.gesture_engine
        )
        self.cam_thread.start()

        self._mol_cache = None
        self._mol_cache_size = (0, 0)
        self._mol_dirty = True

        self._fade_mask_cache = None
        self._fade_mask_size = (0, 0)

        self._last_render_time = 0.0
        self._last_interaction_time = time.time()
        self._is_high_quality = False
        self._hq_rerender_pending = True

        self._pending_cell_change = None
        self._pending_element_change = None

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    def _setup_settings_panel(self):
        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.settings_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        dock = self.settings_dock

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(340)

        self.settings_panel = SettingsPanel()
        self.settings_panel.setStyleSheet(DARK_SS)
        scroll.setWidget(self.settings_panel)
        scroll.setStyleSheet("background: #111122;")
        dock.setWidget(scroll)
        dock.setStyleSheet(DARK_SS)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.settings_toggle_btn = QPushButton("☰", self)
        self.settings_toggle_btn.setFixedSize(36, 36)
        self.settings_toggle_btn.setStyleSheet(
            "QPushButton { background: rgba(30,30,50,0.8); color: #aaa; border: 1px solid #444; "
            "border-radius: 4px; font-size: 18px; } "
            "QPushButton:hover { background: rgba(60,60,90,0.9); color: #fff; }"
        )
        self.settings_toggle_btn.clicked.connect(lambda: self.settings_dock.setVisible(True))
        self.settings_toggle_btn.hide()
        self.settings_dock.visibilityChanged.connect(
            lambda vis: self.settings_toggle_btn.setVisible(not vis))

        self.settings_panel.populate_elements(self.renderer.get_element_info())

        if not self.renderer.has_cell():
            self.settings_panel.cb_cell.setEnabled(False)
            self.settings_panel.cb_cell.setChecked(False)

        self.settings_panel.cb_axes.stateChanged.connect(self._on_axes_toggled)
        self.settings_panel.cb_cell.stateChanged.connect(self._on_cell_toggled)

        for name, cb in self.settings_panel.element_checkboxes.items():
            cb.stateChanged.connect(self._on_element_toggled)

        self.settings_panel.render_btn.clicked.connect(self._on_hq_render)
        self.settings_panel.open_render_btn.clicked.connect(self._on_open_render)

        self.settings_panel.rot_slider.valueChanged.connect(
            lambda v: setattr(self.gesture_engine, 'rotation_sensitivity', float(v)))
        self.settings_panel.zoom_slider.valueChanged.connect(
            lambda v: setattr(self.gesture_engine, 'zoom_sensitivity', float(v)))

        self.settings_panel.open_btn.clicked.connect(self._on_open_file)
        self._update_structure_info()

    def _update_structure_info(self):
        fname = os.path.basename(self.renderer.filepath)
        self.settings_panel.info_label.setText(
            f"{fname}\n{self.renderer.num_atoms} atoms")

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Structure", "",
            "Structure Files (*.xyz *.cif *.pdb *.poscar *.vasp *.lammps);;All (*)")
        if path and path not in self.file_list:
            self.file_list.append(path)
            try:
                r = MoleculeRenderer(path)
                thumb = r.render(160, 100)
                r.cleanup()
                self.thumbnails.append(thumb)
            except Exception:
                self.thumbnails.append(None)
            self.file_idx = len(self.file_list) - 1
            self._load_current_structure()

    def _on_hq_render(self):
        self.settings_panel.render_status.setText("Rendering...")
        self.settings_panel.open_render_btn.hide()
        self.settings_panel.render_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from ovito.vis import TachyonRenderer as TR
            from molecule_renderer import qimage_to_bgr
            r = TR()
            r.antialiasing_samples = 8
            r.ambient_occlusion = True
            r.ambient_occlusion_brightness = 0.65
            r.shadows = True
            self.renderer._update_camera()
            qimg = self.renderer.vp.render_image(size=(1920, 1440), renderer=r)
            result = qimage_to_bgr(qimg)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.basename(self.renderer.filepath).rsplit('.', 1)[0]
            self._last_render_path = os.path.expanduser(f"~/Desktop/GestureMol_{fname}_{ts}.png")
            cv2.imwrite(self._last_render_path, result)
            self.settings_panel.render_status.setText(f"Saved: {os.path.basename(self._last_render_path)}")
            self.settings_panel.open_render_btn.show()
        except Exception as e:
            self.settings_panel.render_status.setText(f"Error: {e}")
        self.settings_panel.render_btn.setEnabled(True)

    def _on_open_render(self):
        if hasattr(self, '_last_render_path') and os.path.exists(self._last_render_path):
            subprocess.Popen(["open", self._last_render_path])

    def _on_axes_toggled(self, state):
        self._show_axes = state == Qt.Checked.value
        self._mol_dirty = True

    def _on_cell_toggled(self, state):
        self._pending_cell_change = (state == Qt.Checked.value)

    def _on_element_toggled(self, _state):
        self._pending_element_change = self.settings_panel.get_hidden_elements()

    def _draw_axes_overlay(self, frame):
        if not self._show_axes:
            return
        h, w = frame.shape[:2]
        right, up, _fwd = self.renderer.get_camera_vectors()

        cx, cy = 45, h - 140
        length = 32
        axes = [
            (np.array([1, 0, 0]), (80, 80, 255), "X"),
            (np.array([0, 1, 0]), (80, 255, 80), "Y"),
            (np.array([0, 0, 1]), (255, 160, 60), "Z"),
        ]
        for axis, color, label in axes:
            sx = float(np.dot(axis, right))
            sy = float(-np.dot(axis, up))
            ex = int(cx + sx * length)
            ey = int(cy + sy * length)
            cv2.line(frame, (cx, cy), (ex, ey), color, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (ex + 4, ey + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (180, 180, 180), -1, cv2.LINE_AA)

    def _mark_dirty(self):
        self._mol_dirty = True
        self._last_interaction_time = time.time()
        if self._is_high_quality:
            self._is_high_quality = False
            self.renderer.set_quality(False)
        self._hq_rerender_pending = True

    def _generate_thumbnails(self):
        thumbs = []
        for fp in self.file_list:
            try:
                r = MoleculeRenderer(fp)
                thumb = r.render(160, 100)
                r.cleanup()
                thumbs.append(thumb)
            except Exception:
                thumbs.append(None)
        return thumbs

    def _apply_pending_changes(self):
        if self._pending_cell_change is not None:
            self.renderer.set_cell_visible(self._pending_cell_change)
            self._pending_cell_change = None
            self._mol_dirty = True
        if self._pending_element_change is not None:
            self.renderer.set_hidden_elements(self._pending_element_change)
            self._pending_element_change = None
            self._mol_dirty = True

    def _tick(self):
        if getattr(self, '_closing', False):
            return
        t0 = time.time()
        self._apply_pending_changes()

        cam_frame, gesture_result, frame_id = self.cam_thread.get_latest()
        if frame_id != self._last_cam_frame_id:
            self._last_cam_frame_id = frame_id
            if gesture_result:
                d_az, d_el = gesture_result.rotation_delta
                if d_az != 0 or d_el != 0:
                    self.renderer.rotate(d_az, d_el)
                    self._mark_dirty()
                if gesture_result.zoom_delta != 0:
                    factor = 1.0 - gesture_result.zoom_delta * 0.3
                    self.renderer.zoom(max(0.5, min(2.0, factor)))
                    self._mark_dirty()
                if gesture_result.swipe == "LEFT":
                    self._next_structure()
                elif gesture_result.swipe == "RIGHT":
                    self._prev_structure()
                self.gesture_name = gesture_result.gesture
                status_map = {
                    "OPEN": "✋ Rotating", "FIST": "✊ Paused",
                    "ZOOM": "✋✋ Zooming", "IDLE": "No hand",
                }
                status = status_map.get(self.gesture_name, self.gesture_name)
                self.settings_panel.gesture_status.setText(status)
            if cam_frame is not None:
                self.cam_frame = cam_frame

        w = self.mol_view.width()
        h = self.mol_view.height()
        if w < 10 or h < 10:
            return

        now = time.time()
        if self._hq_rerender_pending and not self._is_high_quality and (now - self._last_interaction_time) > 0.3:
            self._is_high_quality = True
            self.renderer.set_quality(True)
            self._mol_dirty = True
            self._hq_rerender_pending = False

        if self._mol_dirty or self._mol_cache is None or self._mol_cache_size != (w, h):
            if not self._is_high_quality and (now - self._last_render_time) < 0.03:
                pass
            else:
                rw = w
                rh = h
                self._mol_cache = self.renderer.render(rw, rh)
                self._mol_cache_size = (w, h)
                self._mol_dirty = False
                self._last_render_time = now

        mol_frame = self._mol_cache.copy()

        if int(t0) != getattr(self, '_last_fps_update', 0):
            self._last_fps_update = int(t0)
            self.settings_panel.fps_label.setText(f"{self.fps:.0f} fps")
        self._draw_axes_overlay(mol_frame)
        self._draw_gallery(mol_frame)
        self._draw_camera_overlay(mol_frame)

        self.mol_view.update_frame(mol_frame)

        dt = time.time() - t0
        self.fps = self.fps * 0.85 + (1.0 / max(dt, 0.001)) * 0.15

    def _draw_status_overlay(self, frame):
        h, w = frame.shape[:2]

        for y in range(48):
            alpha = 0.6 * (1.0 - y / 48.0)
            frame[y, :] = (frame[y, :].astype(float) * (1 - alpha)).astype(np.uint8)

        filename = os.path.basename(self.renderer.filepath)
        cv2.putText(frame, filename, (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        atom_text = f"{self.renderer.num_atoms} atoms"
        (tw, th), _ = cv2.getTextSize(atom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        pill_x = w - tw - 32
        pill_y = 14

        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x - 8, pill_y - 6), (pill_x + tw + 8, pill_y + th + 6),
                      (60, 60, 70), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, atom_text, (pill_x, pill_y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 210), 1, cv2.LINE_AA)

        fps_text = f"{self.fps:.0f} fps"
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.putText(frame, fps_text, (pill_x - fw - 20, pill_y + fh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 130), 1, cv2.LINE_AA)

        if len(self.file_list) > 1:
            idx_text = f"{self.file_idx + 1}/{len(self.file_list)}"
            cv2.putText(frame, idx_text, (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 170), 1, cv2.LINE_AA)

        if not self.settings_dock.isVisible():
            cv2.putText(frame, "[S] Settings", (w - 100, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 120), 1, cv2.LINE_AA)

    def _draw_gallery(self, frame):
        if len(self.file_list) <= 1:
            return

        h, w = frame.shape[:2]
        gallery_h = 64
        gallery_y = h - gallery_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, gallery_y), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

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

            if i < len(self.thumbnails) and self.thumbnails[i] is not None:
                resized = cv2.resize(self.thumbnails[i], (thumb_w, thumb_h))
                frame[y:y+thumb_h, x:x+thumb_w] = resized

            if i == self.file_idx:
                cv2.rectangle(frame, (x-2, y-2), (x+thumb_w+2, y+thumb_h+2),
                              (50, 200, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+thumb_w, y+thumb_h),
                              (60, 60, 70), 1)

            name = os.path.splitext(os.path.basename(filepath))[0][:10]
            cv2.putText(frame, name, (x + 2, y + thumb_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 140, 150), 1, cv2.LINE_AA)

    def _get_fade_mask(self, dw, dh):
        if self._fade_mask_size == (dw, dh) and self._fade_mask_cache is not None:
            return self._fade_mask_cache
        fade = min(20, dw // 4, dh // 4)
        mask = np.ones((dh, dw), dtype=np.float32) * 0.55
        if fade > 0:
            mask[:fade, :] *= np.linspace(0, 1, fade)[:, None]
            mask[-fade:, :] *= np.linspace(1, 0, fade)[:, None]
            mask[:, :fade] *= np.linspace(0, 1, fade)[None, :]
            mask[:, -fade:] *= np.linspace(1, 0, fade)[None, :]
        self._fade_mask_cache = mask[:, :, None]
        self._fade_mask_size = (dw, dh)
        return self._fade_mask_cache

    def _draw_camera_overlay(self, frame):
        if self.cam_frame is None:
            return

        mol_h, mol_w = frame.shape[:2]
        cam_h, cam_w = self.cam_frame.shape[:2]

        display_h = int(mol_h * 0.28)
        display_w = int(cam_w * display_h / cam_h)
        cam_small = cv2.resize(self.cam_frame, (display_w, display_h))

        margin = 12
        gallery_offset = 76 if len(self.file_list) > 1 else 0
        x1 = mol_w - display_w - margin
        y1 = mol_h - display_h - margin - gallery_offset
        x2, y2 = x1 + display_w, y1 + display_h

        if y1 >= 0 and x1 >= 0:
            roi = frame[y1:y2, x1:x2]
            if roi.shape == cam_small.shape:
                mask3 = self._get_fade_mask(display_w, display_h)
                blended = (cam_small.astype(np.float32) * mask3 + roi.astype(np.float32) * (1 - mask3)).astype(np.uint8)
                frame[y1:y2, x1:x2] = blended

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'settings_toggle_btn'):
            self.settings_toggle_btn.move(self.width() - 46, 6)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self._next_structure()
        elif event.key() == Qt.Key_Left:
            self._prev_structure()
        elif event.key() == Qt.Key_S:
            self.settings_dock.setVisible(not self.settings_dock.isVisible())
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
        try:
            self.renderer.pipeline.remove_from_scene()
        except Exception:
            pass
        self.renderer = MoleculeRenderer(self.file_list[self.file_idx])
        self.mol_view.renderer = self.renderer
        self._mol_dirty = True
        self._is_high_quality = False
        self._hq_rerender_pending = True
        self._last_interaction_time = time.time()

        self.settings_panel.populate_elements(self.renderer.get_element_info())
        for name, cb in self.settings_panel.element_checkboxes.items():
            cb.stateChanged.connect(self._on_element_toggled)

        has_cell = self.renderer.has_cell()
        self.settings_panel.cb_cell.setEnabled(has_cell)
        self.settings_panel.cb_cell.setChecked(has_cell)
        self._update_structure_info()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_EXTENSIONS and path not in self.file_list:
                self.file_list.append(path)
                try:
                    r = MoleculeRenderer(path)
                    thumb = r.render(160, 100)
                    r.cleanup()
                    self.thumbnails.append(thumb)
                except Exception:
                    self.thumbnails.append(None)
                self.file_idx = len(self.file_list) - 1
                self._load_current_structure()

    def closeEvent(self, event):
        self._closing = True
        self.timer.stop()
        self.cam_thread.stop()
        self.cam_reader.stop()
        try:
            if self.cam and self.cam.isOpened():
                self.cam.release()
        except Exception:
            pass
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
    app.exec()
    os._exit(0)


if __name__ == "__main__":
    main()
