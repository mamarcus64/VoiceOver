#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VoiceOver Setup ==="
echo ""

# --- Git LFS: do not auto-download large files on pull/checkout (this clone only) ---
if [ -d .git ] && command -v git-lfs &>/dev/null; then
  echo "[0/4] Git LFS: lazy download (skip smudge on checkout/pull)..."
  git lfs install --local --skip-smudge --force
  echo "      Large files stay as pointers until: git lfs pull --include='…'"
  echo ""
fi

# --- Python virtual environment ---
# Require Python 3.10+ (3.9 is not supported by yt-dlp)
PYTHON_BIN=""
for candidate in python3.10 python3.11 python3.12 python3.13 python3; do
  if command -v "$candidate" &>/dev/null; then
    version=$("$candidate" -c 'import sys; print(sys.version_info[:2])')
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: Python 3.10+ is required but not found."
  echo "  - macOS: brew install python@3.10"
  echo "  - Ubuntu: sudo apt install python3.10 python3.10-venv"
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "[1/4] Creating Python virtual environment (using $PYTHON_BIN)..."
  "$PYTHON_BIN" -m venv venv
else
  echo "[1/4] Python venv already exists, skipping."
fi

echo "[2/4] Installing Python dependencies..."
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r backend/requirements.txt

# --- Node.js / Frontend ---
if ! command -v node &> /dev/null; then
  echo ""
  echo "ERROR: Node.js is not installed. Please install Node.js 18+ first."
  echo "  - macOS: brew install node"
  echo "  - Ubuntu: sudo apt install nodejs npm"
  echo "  - Or use nvm: https://github.com/nvm-sh/nvm"
  exit 1
fi

echo "[3/4] Installing frontend dependencies and building..."
cd frontend
npm install --silent
npm run build
cd ..

# --- Copy build to backend static dir ---
echo "[4/4] Deploying frontend build to backend/static/..."
rm -rf backend/static
cp -r frontend/dist backend/static
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To run VoiceOver:"
echo "  source venv/bin/activate"
echo "  cd backend"
echo "  uvicorn main:app --host 0.0.0.0 --port 1945"
echo ""
echo "Then open http://localhost:1945 in your browser."
echo ""
echo "For development (with hot reload):"
echo "  Terminal 1: source venv/bin/activate && cd backend && uvicorn main:app --reload --port 1945"
echo "  Terminal 2: cd frontend && npm run dev"
echo "  Then open http://localhost:5173"
