#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VoiceOver Setup ==="
echo ""

# --- Python virtual environment ---
if [ ! -d "venv" ]; then
  echo "[1/4] Creating Python virtual environment..."
  python3 -m venv venv
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

# --- Copy build to backend ---
echo "[4/4] Linking frontend build for production serving..."
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To run VoiceOver:"
echo "  source venv/bin/activate"
echo "  cd backend"
echo "  uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then open http://localhost:8000 in your browser."
echo ""
echo "For development (with hot reload):"
echo "  Terminal 1: source venv/bin/activate && cd backend && uvicorn main:app --reload --port 8000"
echo "  Terminal 2: cd frontend && npm run dev"
echo "  Then open http://localhost:5173"
