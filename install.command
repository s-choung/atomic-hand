#!/bin/bash
# Atomic Hand — one-click installer for macOS.
# Double-click this file in Finder. It will:
#   1. install Miniconda if you don't have conda
#   2. create the "atomic-hand" environment (Python 3.11)
#   3. install all dependencies (OVITO, MediaPipe, etc.)
#   4. create a double-clickable "Atomic Hand.command" launcher
#
# No terminal knowledge required. If macOS says "unidentified developer",
# right-click this file -> Open -> Open.

set -u

# ---- pretty output ----
BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
say()  { printf "%s\n" "${BOLD}$*${RESET}"; }
ok()   { printf "%s\n" "${GREEN}✓ $*${RESET}"; }
warn() { printf "%s\n" "${YELLOW}! $*${RESET}"; }
die()  { printf "%s\n" "${RED}✗ $*${RESET}"; printf "\nPress Return to close.\n"; read -r _; exit 1; }

ENV_NAME="atomic-hand"
PY_VER="3.11"
REPO_URL="https://github.com/s-choung/atomic-hand.git"

clear
say "Atomic Hand installer"
echo "${DIM}This sets up everything for you. It can take a few minutes the first time.${RESET}"
echo

# ---- 0. locate the project (this script's folder, or clone it) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -f "$SCRIPT_DIR/src/app_window.py" ]; then
  REPO_DIR="$SCRIPT_DIR"
  ok "Found project at: $REPO_DIR"
else
  REPO_DIR="$HOME/atomic-hand"
  if [ -f "$REPO_DIR/src/app_window.py" ]; then
    ok "Using existing project at: $REPO_DIR"
  else
    say "Downloading Atomic Hand..."
    if ! command -v git >/dev/null 2>&1; then
      die "git is not installed. Run 'xcode-select --install' first, then re-run this."
    fi
    git clone "$REPO_URL" "$REPO_DIR" || die "Clone failed."
    ok "Downloaded to: $REPO_DIR"
  fi
fi

# ---- 1. find or install conda ----
find_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base 2>/dev/null && return 0
  fi
  for c in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" \
           "/opt/homebrew/Caskroom/miniconda/base" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -x "$c/bin/conda" ]; then echo "$c"; return 0; fi
  done
  return 1
}

say "Step 1/4 — checking for conda..."
CONDA_BASE="$(find_conda_base || true)"

if [ -z "${CONDA_BASE:-}" ]; then
  warn "conda not found. Installing Miniconda (this is safe and self-contained)."
  ARCH="$(uname -m)"
  if [ "$ARCH" = "arm64" ]; then
    MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
  else
    MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
  fi
  TMP_SH="$(mktemp -t miniconda).sh"
  curl -fsSL "$MC_URL" -o "$TMP_SH" || die "Could not download Miniconda."
  bash "$TMP_SH" -b -p "$HOME/miniconda3" || die "Miniconda install failed."
  rm -f "$TMP_SH"
  CONDA_BASE="$HOME/miniconda3"
  ok "Miniconda installed at $CONDA_BASE"
else
  ok "conda found at $CONDA_BASE"
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh" || die "Could not initialize conda."

# ---- 2. create the environment ----
say "Step 2/4 — creating the '$ENV_NAME' environment (Python $PY_VER)..."
if conda env list | grep -qE "^\s*$ENV_NAME\s"; then
  ok "Environment '$ENV_NAME' already exists — reusing it."
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER" || die "Could not create environment."
  ok "Environment created."
fi

# ---- 3. install dependencies ----
say "Step 3/4 — installing dependencies (OVITO first, to avoid Qt conflicts)..."
echo "${DIM}This is the long part. Grab a coffee.${RESET}"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip >/dev/null 2>&1
conda run -n "$ENV_NAME" pip install ovito || die "Failed to install OVITO."
conda run -n "$ENV_NAME" pip install "mediapipe==0.10.14" opencv-python PySide6 numpy || die "Failed to install dependencies."
ok "All dependencies installed."

# ---- 4. create a double-clickable launcher ----
say "Step 4/4 — creating the launcher..."
LAUNCHER="$REPO_DIR/Atomic Hand.command"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Double-click to launch Atomic Hand.
cd "$REPO_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
exec python src/app_window.py structures
EOF
chmod +x "$LAUNCHER"

# also drop a copy on the Desktop for easy access
if [ -d "$HOME/Desktop" ]; then
  cp "$LAUNCHER" "$HOME/Desktop/Atomic Hand.command" 2>/dev/null && chmod +x "$HOME/Desktop/Atomic Hand.command" 2>/dev/null
fi
ok "Launcher created: 'Atomic Hand.command' (in the project folder and on your Desktop)."

echo
say "${GREEN}Done!${RESET}"
echo "To start the app, double-click ${BOLD}Atomic Hand.command${RESET} (on your Desktop)."
echo "Grant camera permission when macOS asks."
echo
printf "Launch Atomic Hand now? [Y/n] "
read -r ans
case "${ans:-Y}" in
  [nN]*) echo "OK. Double-click the launcher whenever you're ready." ;;
  *) say "Launching..."; conda activate "$ENV_NAME"; cd "$REPO_DIR"; exec python src/app_window.py structures ;;
esac
