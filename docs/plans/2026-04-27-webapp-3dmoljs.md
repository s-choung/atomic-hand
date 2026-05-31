# GestureMol Web App Implementation Plan (3Dmol.js)

**Goal:** 현재 데스크톱 GestureMol 앱을 웹 브라우저에서 동작하는 웹앱으로 구현.
MediaPipe Hands (JavaScript) + 3Dmol.js로 손 제스처 기반 분자 뷰어를 브라우저에서 실행.

**Architecture:**
- Frontend: HTML/CSS/JS (vanilla 또는 React)
- 3D 렌더링: 3Dmol.js (BSD, WebGL 기반, CIF/XYZ/PDB 지원)
- 손 인식: MediaPipe Hands JS (@mediapipe/hands)
- 카메라: WebRTC (getUserMedia)
- 배포: Vercel / Cloudflare Pages / GitHub Pages (정적 사이트)

**Tech Stack:** 3Dmol.js, @mediapipe/hands, WebRTC, HTML5 Canvas

---

## 데스크톱 vs 웹 비교

| 항목 | 데스크톱 (현재) | 웹앱 (계획) |
|------|----------------|-------------|
| 3D 렌더 | OVITO OpenGL | 3Dmol.js WebGL |
| 손 인식 | MediaPipe Python | MediaPipe JS |
| 카메라 | OpenCV | WebRTC |
| 파일 입력 | CLI 인자 / 파일 다이얼로그 | 드래그&드롭 / 파일 업로드 |
| 설치 | conda + pip | 없음 (브라우저) |
| 성능 | 네이티브 (빠름) | WebGL (충분) |
| 배포 | .command / .app | URL 공유 |

## 장점

1. **설치 불필요** — URL 접속만으로 동작
2. **크로스 플랫폼** — Windows/Mac/Linux/iPad 모두 가능
3. **3Dmol.js 내장 기능** — 회전/줌 마우스 인터랙션 기본 탑재, 스타일 전환 API 풍부
4. **공유 용이** — 구조 파일 URL로 공유 가능

## Phase 1: 기본 뷰어 (3Dmol.js + 파일 로드)

### Task 1: HTML 뼈대 + 3Dmol.js 뷰어

```html
<!-- 단일 HTML 파일로 시작 -->
<div id="mol-viewer" style="width:100%; height:80vh;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const viewer = $3Dmol.createViewer("mol-viewer", {backgroundColor: "0x1a1a2e"});
  // CIF/XYZ 파일 로드 → viewer.addModel(data, "cif")
  // viewer.setStyle({}, {stick:{}, sphere:{scale:0.3}})
  // viewer.zoomTo(); viewer.render();
</script>
```

- 드래그&드롭 파일 업로드 (CIF, XYZ, PDB)
- 스타일 전환 버튼 (Ball&Stick, Spacefill, Wireframe)
- 다크 배경, 반응형 레이아웃

### Task 2: MediaPipe Hands JS 통합

```javascript
import {Hands} from "@mediapipe/hands";
import {Camera} from "@mediapipe/camera_utils";

const hands = new Hands({locateFile: (file) => 
  `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands: 1, modelComplexity: 1});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => await hands.send({image: videoElement}),
  width: 640, height: 480
});
camera.start();
```

- 카메라 피드를 작은 canvas에 반투명으로 표시
- 손 랜드마크 감지 → 제스처 분류

### Task 3: 제스처 → 3Dmol.js 뷰포트 제어

3Dmol.js API:
```javascript
// 회전
viewer.rotate(angle, axis);  // axis: "x", "y", "z"

// 줌
viewer.zoom(factor);  // 1.1 = 줌인, 0.9 = 줌아웃

// 스타일 전환
viewer.setStyle({}, {stick:{}, sphere:{scale:0.3}});  // ball&stick
viewer.setStyle({}, {sphere:{scale:1.0}});              // spacefill
viewer.setStyle({}, {line:{}});                          // wireframe
viewer.render();
```

제스처 매핑 (데스크톱과 동일):
- 주먹 + 손 회전 → viewer.rotate()
- 핀치 → viewer.zoom()
- 손 펴기 → 트래젝터리 재생
- 손가락 1/2/3 → 스타일 전환

## Phase 2: UX 개선

### Task 4: UI 레이아웃

```
┌─────────────────────────────────────┐
│ [파일 드롭] [B&S] [SF] [Wire] [도움말] │  ← 상단 바
├─────────────────────────────────────┤
│                                     │
│         3Dmol.js 뷰어               │  ← 메인 (90%)
│                                     │
│                        ┌──────┐     │
│                        │카메라│     │  ← 우하단 반투명
│                        └──────┘     │
│  ┌──────────────┐                   │
│  │ 제스처 가이드 │                   │  ← 좌하단
│  └──────────────┘                   │
└─────────────────────────────────────┘
```

### Task 5: 모바일/태블릿 지원
- 터치 제스처 폴백 (MediaPipe 없을 때)
- 반응형 레이아웃

### Task 6: URL 파라미터로 구조 공유
- `?url=https://...file.cif` → 구조 파일 URL로 직접 로드
- `?style=spacefill` → 스타일 프리셋

## Phase 3: 고급 기능

### Task 7: 트래젝터리 재생
- 멀티프레임 XYZ → 애니메이션 재생/일시정지
- 3Dmol.js의 `viewer.animate()` 활용

### Task 8: 원자 편집 모드 (Advanced)
- 포인팅으로 원자 선택 → 하이라이트
- 주먹으로 잡아서 이동
- 수정된 구조 다운로드 (XYZ/CIF)

### Task 9: 배포
- Vercel 또는 Cloudflare Pages
- GitHub Pages로도 가능 (정적 사이트)
- PWA로 오프라인 지원 추가 가능

## 예상 시간

| Phase | 예상 |
|-------|------|
| Phase 1: 기본 뷰어 + 제스처 | 2-3시간 |
| Phase 2: UX 개선 | 1-2시간 |
| Phase 3: 고급 기능 | 3-4시간 |

## 참고 리소스

- 3Dmol.js docs: https://3dmol.csb.pitt.edu/doc/
- MediaPipe Hands JS: https://google.github.io/mediapipe/solutions/hands.html
- 3Dmol.js examples: https://3dmol.csb.pitt.edu/
