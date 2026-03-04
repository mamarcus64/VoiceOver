#!/usr/bin/env bash
set -e

# kill any existing processes on port 1945
if lsof -i :1945 | awk 'NR!=1 {print $2}' | xargs kill -9 2>/dev/null; then
    echo "Killed existing processes on port 1945"
fi

PORT=1945

echo "==> Building frontend..."
cd frontend
npm run build
cd ..

echo "==> Starting server on port $PORT"
source venv/bin/activate
cd backend
uvicorn main:app --host 0.0.0.0 --port $PORT
