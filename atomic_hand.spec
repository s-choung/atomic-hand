# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for "Atomic Hand" (gesture-controlled OVITO molecule viewer).
# Build:  conda run -n base pyinstaller atomic_hand.spec --noconfirm
#
# Key risk: OVITO + PySide6 both rely on Qt. In this env OVITO was pip-installed
# and shares PySide6's Qt (ovito/qt_compat.py does `from PySide6 import ...`),
# so we collect a single Qt copy via PySide6 and let --collect-all ovito grab
# its native plugins/dylibs.

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []

for pkg in ("ovito", "mediapipe"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# mediapipe model assets (.tflite / .binarypb) sometimes missed by collect_all
datas += collect_data_files("mediapipe", include_py_files=False)

# OVITO loads its plugins dynamically; make sure submodules are pulled in.
hiddenimports += [
    "ovito.io", "ovito.vis", "ovito.modifiers", "ovito.pipeline", "ovito.data",
    "ovito._extensions.pyscript",
    "shiboken6",
    "PySide6.QtNetwork", "PySide6.QtXml", "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
]

# Bundle the structure files so the app has defaults to show.
datas += [("structures", "structures")]

block_cipher = None

a = Analysis(
    ["src/app_window.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Atomic Hand never imports these; they get swept in from the base env and
    # bloat the bundle by ~1 GB. Excluding them shrinks the .app to a few hundred MB.
    excludes=[
        "torch", "torchvision", "torchaudio", "sympy", "networkx",
        "matplotlib", "pandas", "scipy", "IPython", "jupyter", "notebook",
        "mediapipe.tasks.python.vision.face_landmarker",
        "mediapipe.tasks.python.vision.pose_landmarker",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="app_window",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # --windowed
    disable_windowed_traceback=False,
    argv_emulation=True,  # let macOS file-open events / drag-drop pass argv
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="app_window",
)

app = BUNDLE(
    coll,
    name="Atomic Hand.app",
    icon="icon.png",
    bundle_identifier="com.atomichand.viewer",
    info_plist={
        "NSCameraUsageDescription": "Atomic Hand uses the camera for hand-gesture control of the molecule viewer.",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
