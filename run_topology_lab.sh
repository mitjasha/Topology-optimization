#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=python3
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
fi
if ! "$PYTHON" -c "import streamlit" >/dev/null 2>&1; then
    echo "Install GUI requirements first:"
    echo "    $PYTHON -m pip install -r requirements-gui.txt"
    exit 1
fi
exec "$PYTHON" -m streamlit run topopt/mlnext/gui.py --server.address 127.0.0.1
