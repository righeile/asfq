#!/bin/bash
# Double-click to start ASFQ.  The first run builds a Python
# environment; later runs reuse it, refresh it when requirements.txt changes,
# and make sure the browser opens onto the copy this script just started.
cd "$(dirname "$0")"
trap 'echo; read -r -p "Press Enter to close this window..."' EXIT
set -e

VENV=.venv
PORT=${PORT:-8765}

# --- the interpreter ---------------------------------------------------------
# A Mac does not come with a Python this app can use.  `python3` is either
# missing entirely or Xcode's 3.9, and finding that out inside pip costs
# several minutes and reads like a dependency problem.  Ask each candidate for
# its version instead, and take the first one that is new enough.
PY=""
for cand in python3 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$cand" >/dev/null 2>&1 &&
       "$cand" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    echo "This app needs Python 3.10 or newer, and this machine has none on PATH."
    echo "Install it from https://www.python.org/downloads/ (the .pkg installer),"
    echo "then double-click this file again."
    exit 1
fi

# --- the environment ---------------------------------------------------------
if [ ! -d "$VENV" ]; then
    echo "First run: setting up a Python environment with $PY (a few minutes)..."
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
fi

# An interrupted first run leaves a .venv that exists and holds nothing.
if [ ! -x "$VENV/bin/python" ]; then
    echo "The Python environment in $VENV is incomplete.  Delete that folder and"
    echo "double-click this file again."
    exit 1
fi

# A .venv that exists is not the same as a .venv that is current.  The old
# script installed requirements once and never looked again, so a dependency
# added later was simply absent, and the only symptom was a crash partway
# through whatever needed it.  Keep a copy of the requirements the environment
# was built from and reinstall whenever the real one differs.
if ! cmp -s requirements.txt "$VENV/.requirements.txt"; then
    echo "Installing dependencies..."
    "$VENV/bin/python" -m pip install -q -r requirements.txt
    cp requirements.txt "$VENV/.requirements.txt"
fi

# --- the port ----------------------------------------------------------------
listeners() { lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null; }

# Ask the occupant who it is.  /api/health is the direct answer, but a copy
# started before that route existed cannot give it -- and a server too old to
# introduce itself is exactly the one worth replacing.  It still serves this
# app's own index page off disk, so the page is the fallback identification.
is_us() {
    if curl -fsS -m 2 "http://127.0.0.1:$1/api/health" 2>/dev/null \
       | grep -qE 'asfq|fibrosis-quantifier'; then return 0; fi
    curl -fsS -m 2 "http://127.0.0.1:$1/" 2>/dev/null \
       | grep -qi "fibrosis quantifier"
}

for _ in 1 2 3 4 5; do
    if [ -z "$(listeners "$PORT")" ]; then break; fi

    if is_us "$PORT"; then
        # Almost always this app, still running from an earlier day.  It keeps
        # serving the current HTML off disk with its own older Python behind
        # it, which is why a stale copy looks like a broken one.  Stop it.
        echo "Stopping the copy already running on port $PORT (pid $(listeners "$PORT" | tr '\n' ' '))."
        listeners "$PORT" | xargs kill 2>/dev/null || true
        sleep 1
        if [ -n "$(listeners "$PORT")" ]; then
            listeners "$PORT" | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
        continue
    fi

    echo "Port $PORT is taken by something else; trying $((PORT + 1))."
    PORT=$((PORT + 1))
done

if [ -n "$(listeners "$PORT")" ]; then
    echo "Could not free port $PORT.  Close whatever is using it and try again."
    exit 1
fi

# --- start, then open --------------------------------------------------------
# Open the browser only once the server actually answers, not after a fixed
# wait, so a failed launch does not still end up showing a page.
"$VENV/bin/python" open_when_ready.py "$PORT" 60 &

PORT="$PORT" "$VENV/bin/python" app.py
