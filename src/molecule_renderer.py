"""
MoleculeRenderer v2: OVITO 파이프라인 래퍼.
궤도 카메라, 스타일 전환(ball&stick, spacefill, wireframe), 본드 자동 생성.
"""

import warnings
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import math
import numpy as np
import cv2
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import CreateBondsModifier, ExpressionSelectionModifier, DeleteSelectedModifier
from ovito.pipeline import Pipeline


def qimage_to_bgr(qimg):
    qimg = qimg.convertToFormat(qimg.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, 2::-1].copy()


class MoleculeRenderer:
    STYLES = ["ball_stick", "spacefill", "wireframe"]
    STYLE_LABELS = {
        "ball_stick": "Ball & Stick",
        "spacefill": "Space Filling",
        "wireframe": "Wireframe",
    }

    def __init__(self, filepath):
        self.filepath = filepath
        self.pipeline = import_file(str(filepath))

        self._hide_sel_mod = ExpressionSelectionModifier(expression='0')
        self._hide_sel_mod.enabled = False
        self._hide_del_mod = DeleteSelectedModifier()
        self._hide_del_mod.enabled = False
        self.pipeline.modifiers.append(self._hide_sel_mod)
        self.pipeline.modifiers.append(self._hide_del_mod)

        self.bonds_mod = CreateBondsModifier(mode=CreateBondsModifier.Mode.VdWRadius, vdw_prefactor=0.6)
        self.pipeline.modifiers.append(self.bonds_mod)

        self.pipeline.add_to_scene()

        data = self.pipeline.compute()
        self.num_atoms = data.particles.count
        self.num_frames = self.pipeline.source.num_frames
        self._current_frame = 0

        self._element_info = []
        self._hidden_elements = set()
        for pt in data.particles.particle_types.types:
            self._element_info.append({
                'id': pt.id, 'name': pt.name,
                'color': (pt.color[0], pt.color[1], pt.color[2]),
                'count': int(np.sum(np.array(data.particles.particle_types) == pt.id)),
            })

        self._has_cell = False
        self._cell_visible = False
        try:
            if data.cell is not None:
                cell_matrix = np.array(data.cell.matrix)
                vol = abs(np.linalg.det(cell_matrix[:3, :3]))
                if vol > 1.0:
                    self._has_cell = True
                    self._cell_visible = True
                    data.cell.vis.enabled = True
                else:
                    data.cell.vis.enabled = False
        except Exception:
            pass

        self.vp = Viewport(type=Viewport.Type.Perspective)
        self.vp.zoom_all()

        self._init_camera_from_viewport()
        self.distance *= 1.3
        right, up, _ = self.get_camera_vectors()
        self.center -= up * self.distance * 0.06

        self.renderer = TachyonRenderer()
        self.renderer.antialiasing_samples = 2
        self.renderer.ambient_occlusion = False
        self.renderer.shadows = False

        self.style_idx = 0
        self._apply_style()
        self._warm_up()

    def get_element_info(self):
        return self._element_info

    def set_hidden_elements(self, hidden_names):
        self._hidden_elements = set(hidden_names)
        if not self._hidden_elements:
            self._hide_sel_mod.enabled = False
            self._hide_del_mod.enabled = False
        else:
            hidden_ids = [e['id'] for e in self._element_info if e['name'] in self._hidden_elements]
            if hidden_ids:
                expr = ' || '.join(f'ParticleType=={tid}' for tid in hidden_ids)
                self._hide_sel_mod.expression = expr
                self._hide_sel_mod.enabled = True
                self._hide_del_mod.enabled = True
            else:
                self._hide_sel_mod.enabled = False
                self._hide_del_mod.enabled = False

    def has_cell(self):
        return self._has_cell

    def set_cell_visible(self, visible):
        self._cell_visible = visible
        if self._has_cell:
            try:
                data = self.pipeline.compute()
                if data.cell is not None:
                    data.cell.vis.enabled = visible
            except Exception:
                pass

    def get_camera_vectors(self):
        forward = np.array(self.vp.camera_dir)
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
        forward = forward / norm
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def set_quality(self, high):
        if high:
            self.renderer.antialiasing_samples = 4
            self.renderer.ambient_occlusion = True
            self.renderer.ambient_occlusion_brightness = 0.65
            self.renderer.shadows = False
        else:
            self.renderer.antialiasing_samples = 2
            self.renderer.ambient_occlusion = False
            self.renderer.shadows = False

    def _init_camera_from_viewport(self):
        pos = np.array(self.vp.camera_pos)
        direction = np.array(self.vp.camera_dir)
        self.center = pos + direction * np.linalg.norm(pos)
        self.distance = np.linalg.norm(pos - self.center)
        if self.distance < 1e-6:
            self.distance = 50.0
            self.center = pos + direction * self.distance

        offset = pos - self.center
        self.elevation = math.degrees(math.asin(np.clip(offset[2] / self.distance, -1, 1)))
        self.azimuth = math.degrees(math.atan2(offset[1], offset[0]))
        self.fov = self.vp.fov

    def _update_camera(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        x = self.distance * math.cos(el) * math.cos(az)
        y = self.distance * math.cos(el) * math.sin(az)
        z = self.distance * math.sin(el)
        pos = self.center + np.array([x, y, z])

        direction = self.center - pos
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm

        self.vp.camera_pos = tuple(pos)
        self.vp.camera_dir = tuple(direction)
        self.vp.fov = self.fov

    def _warm_up(self):
        self.vp.render_image(size=(100, 75), renderer=self.renderer)

    def _apply_style(self):
        data = self.pipeline.compute()
        pvis = data.particles.vis
        style = self.STYLES[self.style_idx]

        pvis.shape = pvis.Shape.Sphere

        if style == "ball_stick":
            pvis.radius = 0.5
            self.bonds_mod.enabled = True
            if data.particles.bonds is not None:
                bvis = data.particles.bonds.vis
                bvis.width = 0.25
                bvis.enabled = True

        elif style == "spacefill":
            pvis.radius = 1.2
            self.bonds_mod.enabled = False

        elif style == "wireframe":
            pvis.radius = 0.2
            self.bonds_mod.enabled = True
            if data.particles.bonds is not None:
                bvis = data.particles.bonds.vis
                bvis.width = 0.1
                bvis.enabled = True

    def render(self, width=800, height=600):
        self._update_camera()
        try:
            qimg = self.vp.render_image(size=(width, height), renderer=self.renderer)
            result = qimage_to_bgr(qimg)
            self._last_good_render = result
            return result
        except Exception:
            if hasattr(self, '_last_good_render') and self._last_good_render is not None:
                return cv2.resize(self._last_good_render, (width, height))
            return np.zeros((height, width, 3), dtype=np.uint8)

    def rotate(self, d_azimuth, d_elevation):
        self.azimuth += d_azimuth
        self.elevation = max(-89, min(89, self.elevation + d_elevation))

    def pan(self, dx, dy):
        right, up, _ = self.get_camera_vectors()
        scale = self.distance * 0.002
        self.center += right * (-dx * scale) + up * (dy * scale)

    def zoom(self, factor):
        self.distance *= factor
        self.distance = max(1.0, min(500.0, self.distance))

    def reset_camera(self):
        self.vp.zoom_all()
        self._init_camera_from_viewport()

    def next_style(self):
        self.style_idx = (self.style_idx + 1) % len(self.STYLES)
        self._apply_style()
        return self.STYLES[self.style_idx]

    def current_style(self):
        return self.STYLE_LABELS.get(self.STYLES[self.style_idx], self.STYLES[self.style_idx])

    def set_frame(self, frame):
        frame = max(0, min(frame, self.num_frames - 1))
        self._current_frame = frame

    def get_frame(self):
        return self._current_frame

    def next_frame(self):
        self._current_frame = min(self._current_frame + 1, self.num_frames - 1)
        return self._current_frame

    def prev_frame(self):
        self._current_frame = max(self._current_frame - 1, 0)
        return self._current_frame

    def advance_frame(self):
        self._current_frame += 1
        if self._current_frame >= self.num_frames:
            self._current_frame = 0
        return self._current_frame

    # === Edit Mode ===

    def get_atom_positions(self):
        data = self.pipeline.compute()
        return np.array(data.particles.positions)

    def get_atom_types(self):
        data = self.pipeline.compute()
        return np.array(data.particles.particle_types)

    def project_atoms(self, width, height):
        """3D 원자 좌표를 2D 스크린 좌표로 투영. (screen_x, screen_y, depth) 반환."""
        positions = self.get_atom_positions()
        if len(positions) == 0:
            return np.array([]), np.array([]), np.array([])

        self._update_camera()
        cam_pos = np.array(self.vp.camera_pos)
        forward = np.array(self.vp.camera_dir)
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        relative = positions - cam_pos
        z = relative @ forward
        x = relative @ right
        y = relative @ up

        valid = z > 0.1
        scale = np.where(valid, 1.0 / (np.tan(self.fov / 2) * z), 0)

        screen_x = (x * scale * height / 2) + width / 2
        screen_y = (-y * scale * height / 2) + height / 2

        return screen_x, screen_y, z

    def find_nearest_atom(self, click_x, click_y, width, height, max_dist_px=30):
        """클릭 위치에서 가장 가까운 원자 인덱스 반환. 없으면 -1."""
        sx, sy, sz = self.project_atoms(width, height)
        if len(sx) == 0:
            return -1

        dist = np.sqrt((sx - click_x) ** 2 + (sy - click_y) ** 2)
        visible = sz > 0.1
        dist = np.where(visible, dist, 1e6)

        min_idx = np.argmin(dist)
        if dist[min_idx] < max_dist_px:
            return int(min_idx)
        return -1

    def set_selection(self, indices):
        """원자 선택 설정. OVITO가 선택된 원자를 다른 색으로 표시."""
        src = self.pipeline.source.data
        n = src.particles.count
        sel = np.zeros(n, dtype=int)
        for idx in indices:
            if 0 <= idx < n:
                sel[idx] = 1
        src.particles_.selection_ = sel

    def move_atoms(self, indices, delta_3d):
        """선택된 원자들을 3D 델타만큼 이동 (소스 데이터 직접 수정)."""
        src = self.pipeline.source.data
        pos = np.array(src.particles.positions)
        for idx in indices:
            if 0 <= idx < len(pos):
                pos[idx] += delta_3d
        src.particles_.positions_ = pos

    def screen_to_world_delta(self, dx_px, dy_px, width, height):
        """2D 스크린 픽셀 변화량을 3D 월드 좌표 변화량으로 변환."""
        forward = np.array(self.vp.camera_dir)
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        scale = self.distance * np.tan(self.fov / 2) * 2 / height
        return right * dx_px * scale + up * (-dy_px) * scale

    def export_structure(self, output_path):
        """현재 구조를 파일로 내보내기."""
        from ovito.io import export_file
        export_file(self.pipeline, output_path, "xyz",
                    columns=["Particle Type", "Position.X", "Position.Y", "Position.Z"])

    def cleanup(self):
        self.pipeline.remove_from_scene()
