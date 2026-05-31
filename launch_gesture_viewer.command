#!/bin/bash
# Gesture Molecular Viewer Launcher
# 더블클릭하면 Terminal.app에서 실행됨 (카메라 권한 자동 사용)
# 구조 파일을 이 아이콘에 드래그&드롭하면 해당 파일로 열림

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Gesture Molecular Viewer"
echo "=========================================="
echo ""

if [ -n "$1" ]; then
    echo "File: $1"
    conda run -n base python "$DIR/src/gesture_mol_viewer.py" "$1"
else
    echo "파일 선택 다이얼로그가 열립니다..."
    conda run -n base python "$DIR/src/gesture_mol_viewer.py"
fi
