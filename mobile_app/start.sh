#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo " Starting Schwab Token Update Mobile App"
echo "=========================================="

# Use project virtual environment if available
if [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

cd "$SCRIPT_DIR/backend"

echo "Installing/checking backend dependencies..."
$PYTHON_BIN -m pip install -q -r requirements.txt

# Ensure frontend build exists
if [ ! -d "$SCRIPT_DIR/frontend/dist" ]; then
    if command -v npm &> /dev/null; then
        echo "Building frontend PWA assets..."
        (cd "$SCRIPT_DIR/frontend" && npm install && npm run build)
    else
        echo "Warning: frontend/dist not found and npm is not installed."
    fi
fi

echo "Starting FastAPI server on http://127.0.0.1:8001 ..."
echo "------------------------------------------"

$PYTHON_BIN -m uvicorn main:app --host 127.0.0.1 --port 8001
