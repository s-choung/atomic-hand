# Gesture-Controlled Molecular Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** MacBook 카메라로 손 제스처를 인식해서 OVITO 기반 3D 분자 뷰포트를 실시간 제어하는 데스크톱 앱

**Architecture:** OVITO `Viewport.render_image()` → numpy → OpenCV 디스플레이. MediaPipe가 손 좌표를 읽고, 좌표 변화량을 OVITO 카메라 파라미터(camera_pos, camera_dir, fov)에 직접 매핑. 단일 OpenCV 윈도우에 카메라 피드 + 분자 뷰를 나란히 배치. 렌더 성능: 800x600에서 14-21ms (45+ fps), 카메라 이동 후 재렌더 7ms로 실시간 충분.

**Tech Stack:** OVITO 3.13.1, MediaPipe 0.10.14, OpenCV, ASE 3.27.0, numpy

---

## Phase 1: Minimal Molecule Viewer (OVITO → OpenCV)

### Task 1: OVITO 렌더 → OpenCV 디스플레이 파이프라인

**Files:**
- Create: `src/molecule_renderer.py`
- Create: `src/main.py`

**Step 1: MoleculeRenderer 클래스 작성**

`src/molecule_renderer.py` — OVITO 파이프라인을 감싸서 numpy 이미지를 반환하는 클래스.

```python
# 핵심 API:
# renderer = MoleculeRenderer(filepath)  # CIF/XYZ/POSCAR 등
# img = renderer.render(width, height)   # numpy BGR array
# renderer.set_camera(azimuth, elevation, distance)
# renderer.zoom(delta)
# renderer.next_frame() / prev_frame()   # 트래젝터리
```

기능:
- `import_file`로 구조 로드 (CIF, XYZ, POSCAR 지원)
- `Viewport`를 Perspective 모드로 생성, `zoom_all()`
- 궤도 카메라: azimuth, elevation, distance로 camera_pos 계산
- `render_image()` → QImage → numpy BGR 변환
- 배경색 검은색 (`renderer.background_color`)

**Step 2: 단독 테스트 — CIF/XYZ 파일 렌더 확인**

Run: `conda run -n base python -c "from src.molecule_renderer import MoleculeRenderer; ..."`

- CIF 파일 로드 후 render() → OpenCV imshow로 표시
- 키보드 A/D로 azimuth, W/S로 elevation, +/-로 zoom 동작 확인

**Step 3: Commit**

---

### Task 2: 키보드 컨트롤 뷰어 (OpenCV 윈도���)

**Files:**
- Create: `src/viewer.py` (메인 루프, OpenCV 윈도우)

**Step 1: 뷰어 메인루프 작성**

- OpenCV 윈도우에 렌더된 분자 이미지 표시
- 키보드 입력:
  - `A/D` → azimuth ±5°
  - `W/S` → elevation ±5°
  - `+/-` → zoom in/out
  - `[/]` → 다음/이전 프레임 (트래젝���리)
  - `R` → 카메라 리셋 (zoom_all)
  - `P` → 스크린샷 저장
  - `Q/ESC` → 종료
- 화면에 현재 카메라 파라미터 + 원자 수 + 파일명 오버레이

**Step 2: CIF, XYZ 파일 ��각 테스트**

Run: `conda run -n base python src/viewer.py <filepath>`

- 키보드로 회전/줌 동작 확인
- 렌더 FPS 표시로 성능 확인 (목표: 30+ fps)

**Step 3: Commit**

---

## Phase 2: 제스처 엔진 (MediaPipe)

### Task 3: 강화된 제스처 감지기

**Files:**
- Create: `src/gesture_engine.py`

**Step 1: GestureEngine 클래스 작성**

gesture_player.py의 GestureDetector를 확장. 추가 기능:

- **Palm drag tracking**: 손바닥 펴고 이동 시 (x, y) 델타 추적 → 연속 회전
- **Pinch detection**: 엄지-검지 거리 변화 → 줌 인/아웃
- 기존: Fist(토글), Swipe(이전/다음), Peace(스크린샷)

```python
# 핵심 API:
# engine = GestureEngine()
# result = engine.process(hand_landmarks)
# result.gesture: str          — "OPEN", "FIST", "PINCH", "POINT", "PEACE"
# result.action: str | None    — "TOGGLE", "SWIPE_LEFT", "SWIPE_RIGHT", "SCREENSHOT"
# result.drag_delta: (dx, dy)  — 손바닥 드래그 변화량 (0,0 if not dragging)
# result.pinch_delta: float    — 핀치 거리 변화량 (0 if not pinching)
```

제스처 상태 머신:
- `IDLE` → 손 감지 안 됨
- `OPEN_DRAG` → 손바닥 펴고 이동 중 (회전)
- `PINCH` → 엄지+검지 가까이 → 거리 변화로 줌
- `FIST` → 주먹 → open→fist 전환 시 토글 액션
- `PEACE` → 피스 → 스크린샷

**Step 2: 카메라 피드로 단독 테스트**

- OpenCV 카메라 + GestureEngine
- 화면에 gesture, drag_delta, pinch_delta 실시�� 표시
- 각 제스처 전환이 정확한지 확인

**Step 3: Commit**

---

### Task 4: 핀치 줌 구��

**Files:**
- Modify: `src/gesture_engine.py`

**Step 1: 핀치 감지 로직 추가**

```python
# 엄지 끝(4)과 검지 끝(8) 사이 유클리드 거리
dist = sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
# 임계값 이하면 PINCH 상태
# 이전 프레임 대비 거리 변화 = pinch_delta
```

- 핀치 시작: 엄지-검지 거리 < 0.08
- 핀치 중: 거리 변화량 추적
- 핀치 종료: 거리 > 0.12 또는 다른 제스처

**Step 2: ���치로 줌 테스트 (카메라 피드)**

**Step 3: Commit**

---

## Phase 3: 통합 — 제스처로 분자 제어

### Task 5: 메인 앱 통합

**Files:**
- Create: `src/gesture_mol_viewer.py` (최종 앱)

**Step 1: 카메라 + 분자 뷰 통합 루프**

레이아웃: `[카메라 피드 (480px)] [분자 뷰 (800px)]`

메인 루프:
```
1. 카메라 프레임 읽기
2. MediaPipe 손 감지
3. GestureEngine.process() → result
4. result에 따라 MoleculeRenderer 카메라 업데이트:
   - drag_delta → azimuth/elevation 변경
   - pinch_delta → zoom 변경
   - SWIPE → next/prev frame
   - FIST → 회전 모드/선택 모드 토글 (또는 렌더 스타일 전환)
   - PEACE → 스크��샷 저장
5. 분자 렌더링
6. 두 이미지 나란히 합쳐서 표시
```

**Step 2: ���스처 가이드 오버레이**

카메라 뷰 하단에 항상 보이는 가이드:
- 현재 인식된 제스처 하이라이트
- 액션 발생 시 팝업 피드백 (1.2초)
- 현재 모드 표시 (회전/줌)

**Step 3: CIF 파일로 테스트**

Run: `conda run -n base python src/gesture_mol_viewer.py <CIF path>`

- 손바닥 드래그 → 분자 회전 확인
- 핀치 → 줌 확인
- 주먹 → 토글 확인
- 피스 → 스크린��� 확인

**Step 4: XYZ 파일로 테스트 (2592 atoms)**

- 큰 구조에서도 30+ fps 유지하는지 확인

**Step 5: Commit**

---

### Task 6: 렌더 스타일 전환

**Files:**
- Modify: `src/molecule_renderer.py`

**Step 1: ��더 스타일 토글 구현**

주먹 쥐기로 스타일 순환:
1. **Ball & Stick** (기본) — 원자 구, 결합선
2. **Space Filling** — van der Waals 반경으로 큰 구
3. **Wireframe** — 결합만 선으로

OVITO에서:
```python
# particles vis
particles_vis = pipeline.source.data.particles.vis
particles_vis.radius = 0.3  # ball-stick
# or
particles_vis.radius = 1.0  # space-filling
```

**Step 2: 스타일 전환 테스트**

**Step 3: Commit**

---

## Phase 4: 파일 로딩 & UX 개선

### Task 7: 파일 브라우저 & 드래그 앤 드롭

**Files:**
- Modify: `src/gesture_mol_viewer.py`

**Step 1: 커맨드라인 + 파일 다이얼로그**

- 인자 없이 실행 시 `tkinter.filedialog`로 파일 선택 UI
- 지원 확장자: `.cif`, `.xyz`, `.pdb`, `.poscar`, `.vasp`, `.lammps`
- 파일 경로 표시 on screen

**Step 2: ��타임 파일 교체**

- `O` 키 → 새 파일 열기 다이얼로그
- 파일 교체 시 카메라 리셋

**Step 3: Commit**

---

### Task 8: 스크린샷 & 고해상도 ��더

**Files:**
- Modify: `src/molecule_renderer.py`

**Step 1: 스크��샷 기능**

- 피스 사인 → `screenshots/` 폴더에 고해상도(1920x1440) PNG 저장
- 파일명: `{구조명}_{timestamp}.png`
- 저장 시 화면에 "Screenshot saved!" 피드백

**Step 2: Commit**

---

## Phase 5: 패키징 & 런처

### Task 9: .command 런처 파일

**Files:**
- Create: `launch_gesture_viewer.command`

**Step 1: 더블클릭 실행 가능한 런처**

```bash
#!/bin/bash
cd "$(dirname "$0")"
conda run -n base python src/gesture_mol_viewer.py "$@"
```

- `chmod +x` 설정
- 더블클릭하면 Terminal.app이 열리면서 실행
- Terminal.app에 카메라 권한이 있으므로 권한 문제 해결

**Step 2: 파일 드래그 런처**

CIF/XYZ 파일을 .command 파일 위에 드래그 → 해당 구조로 뷰어 실행

**Step 3: Commit**

---

## 제스처 매핑 최종 정리

| 제스처 | 동작 | 분자 뷰어 기능 |
|--------|------|---------------|
| 손바닥 펴고 드래그 | palm (x,y) delta | 모델 회전 (azimuth, elevation) |
| 핀치 (엄지+검지) | 거리 변화 | 줌 인/아웃 |
| 주먹 쥐기 | open→fist 전환 | 렌더 스타일 순환 (B&S → SF → Wire) |
| 스와이프 좌/우 | 빠른 수평 이동 | 이전/다음 프레임 (트래젝터리) |
| 피스 사인 ✌️ | 2손가락 | 고해상도 스크린샷 저장 |
| 검지 포인팅 위/아래 | 검지 위치 | 원자 크기 +/- |

## 파일 구조

```
101_motion/
├── src/
│   ├── molecule_renderer.py   # OVITO 렌더 래퍼
│   ├── gesture_engine.py      # MediaPipe 제스처 엔진
│   ├── gesture_mol_viewer.py  # 메인 앱 (통합)
│   └── viewer.py              # 키보드 전용 뷰어 (디버그용)
├── screenshots/               # 피스 제스처 스크린샷
├── launch_gesture_viewer.command  # 더블클릭 런처
├── gesture_player.py          # 기존 영상 플레이어 (유지)
├── hand_gesture_controller.py # 기존 리모컨 모드 (유지)
└── docs/plans/
    └── 2026-04-27-gesture-molecular-viewer.md
```

## 구현 순서 & 예상 시간

| Phase | Tasks | 예상 |
|-------|-------|------|
| Phase 1 | Task 1-2: 키보드 뷰어 | ~15분 |
| Phase 2 | Task 3-4: 제스처 엔진 | ~15분 |
| Phase 3 | Task 5-6: 통합 + 스타일 | ~20분 |
| Phase 4 | Task 7-8: 파일 로딩 + 스크린샷 | ~10분 |
| Phase 5 | Task 9: 런처 | ~5분 |
