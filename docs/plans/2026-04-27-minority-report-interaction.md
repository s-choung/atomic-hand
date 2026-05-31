# Minority Report Style Natural Hand Interaction Plan

**Goal:** "제스처를 외워서 하는" 느낌이 아니라 "그냥 손으로 만지는" 느낌의 자연스러운 분자 제어

**핵심 철학:** 손의 상태가 곧 모드, 움직임이 곧 입력. 이산 제스처 -> 연속 매핑.

---

## 상태 머신

```
IDLE           -- 손 없음 / 가만히
ORBIT          -- 한 손 펴고 이동 -> 모델 회전 (트랙볼)
GRAB/PAN       -- 주먹 쥐고 이동 -> 모델 팬
PINCH_ZOOM     -- 핀치 거리 변화 -> 줌
SELECT/POINT   -- 검지만 펴짐 -> 원자 선택 (레이캐스팅)
TWO_HAND_SCALE -- 양손 거리 변화 -> 줌/스케일
TWO_HAND_TWIST -- 양손 각도 변화 -> Z축 회전
SLICE          -- 손바닥 뒤집기 -> 슬라이싱 모드
```

## Phase 1: 연속 매핑 기반 단일 손

### Task 1: 트랙볼 오빗 (Open Palm)
- 손바닥 중심(lm[0], lm[9] 중점)의 프레임간 delta
- delta_x -> azimuth, delta_y -> elevation
- **rolling average 5프레임** 적용으로 떨림 제거
- 손 멈추면 회전 멈춤 (관성 없음, 직접 매핑)
- 손 사라지면 마지막 위치 유지

### Task 2: 그랩 & 팬 (Fist)
- 주먹 상태에서 손 위치 delta -> 모델 평행 이동
- screen_to_world_delta()로 2D->3D 변환
- "잡고 놓기" 느낌: 쥐면 잡힘, 펴면 멈춤

### Task 3: 핀치 줌 (Thumb+Index)
- landmark 4(엄지)와 8(검지) 거리 연속 추적
- 거리 증가 = 줌인, 감소 = 줌아웃
- 핀치 풀면 줌 고정

### Task 4: 포인팅 원자 선택
- 검지만 펴진 상태 -> 포인터 모드
- 검지 끝(lm[8]) x,y -> 스크린 좌표 -> 레이캐스팅
- 가장 가까운 원자 하이라이트 (호버)
- 0.5초 dwell -> 원자 정보 팝업 (원소, 좌표)
- 클릭(주먹 전환) -> 선택 확정

### Task 5: 히스테리시스 + 스무딩
- 손가락 연장 판정: 확실히 펴짐(0.3 이하) / 확실히 접힘(0.7 이상)
- 중간 영역에서는 이전 상태 유지 (떨림 방지)
- 모든 좌표에 rolling average (3-5 프레임)
- 모드 전환 시 최소 0.3초 유지해야 전환 확정

## Phase 2: 양손 인터랙션

### Task 6: max_num_hands=2 전환
- MediaPipe 설정 변경
- 왼손/오른손 구분 (handedness)
- 한 손만 있을 때 = Phase 1 동작
- 양손 감지 시 = 양손 모드

### Task 7: 양손 줌 (스프레드)
- 양 손바닥 중심 사이 거리 추적
- 거리 벌어짐 = 줌인 (확대)
- 거리 좁아짐 = 줌아웃 (축소)
- 스마트폰 핀치줌과 동일한 메타포

### Task 8: 양손 회전 (트위스트)
- 양 손바닥 중심을 잇는 벡터의 각도 변화
- 각도 변화 = Z축 회전
- 양손을 비트는 동작 -> 모델 비틀기

### Task 9: 양손 잡고 이동 (Dual Grab)
- 양손 다 주먹 -> 양손 중점의 이동 = 모델 팬
- 한 손 주먹 + 한 손 오픈 = 한 손만 모드 적용

## Phase 3: 원자 편집 모드

### Task 10: 레이캐스팅 원자 매핑
- 카메라 매트릭스 역변환
- 2D 스크린 좌표 -> 3D 레이
- 레이와 원자 구 교차 판정
- 가장 가까운(depth 기준) 원자 반환

### Task 11: 원자 선택 & 이동
- 포인팅으로 원자 선택 (테두리 색 변경)
- 다중 선택: 포인팅 + dwell로 추가
- 선택된 원자들을 주먹으로 잡아서 이동
- 손 펴면 위치 고정
- 수정된 구조 export (XYZ/CIF)

### Task 12: 손바닥 뒤집기 모드 전환
- MediaPipe의 handedness + landmark z-depth로 palm facing 판별
- 손바닥 보임 = View 모드
- 손등 보임 = Edit 모드 (or 슬라이싱)

## 기술 요점

### Rolling Average 구현
```python
from collections import deque
class SmoothValue:
    def __init__(self, window=5):
        self.buffer = deque(maxlen=window)
    def update(self, val):
        self.buffer.append(val)
        return sum(self.buffer) / len(self.buffer)
```

### 히스테리시스 구현
```python
class HysteresisSwitch:
    def __init__(self, on_thresh=0.3, off_thresh=0.7):
        self.state = False
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
    def update(self, value):
        if self.state and value > self.off_thresh:
            self.state = False
        elif not self.state and value < self.on_thresh:
            self.state = True
        return self.state
```

### 양손 중심 거리
```python
def two_hand_distance(lm_left, lm_right):
    cx_l = (lm_left[0].x + lm_left[9].x) / 2
    cy_l = (lm_left[0].y + lm_left[9].y) / 2
    cx_r = (lm_right[0].x + lm_right[9].x) / 2
    cy_r = (lm_right[0].y + lm_right[9].y) / 2
    return math.sqrt((cx_l-cx_r)**2 + (cy_l-cy_r)**2)
```

## 예상 시간

| Phase | 예상 |
|-------|------|
| Phase 1: 단일 손 연속 매핑 | 2-3시간 |
| Phase 2: 양손 인터랙션 | 2-3시간 |
| Phase 3: 원자 편집 | 3-4시간 |
