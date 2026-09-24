#!/usr/bin/env python3
"""
open_when_ready.py -- open the browser once the server is actually answering.

Both launchers used to open the browser after a fixed two-second wait, whether
or not anything had started.  When the port was already taken the new server
never bound, the wait elapsed regardless, and the browser landed on whatever
else was listening -- usually a copy of this app left over from another day,
serving the current interface off disk with older code behind it.  Waiting for
a real answer from *this* app is the whole job, so it lives here rather than
being written twice in two shell dialects.

    python open_when_ready.py 8765 [timeout_seconds]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import webbrowser


def ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
            # "fibrosis-quantifier" is what a copy started before the
            # rename answers, and that is the copy worth replacing.
            return json.load(response).get("app") in ("asfq", "fibrosis-quantifier")
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    deadline = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 60.0)
    while time.time() < deadline:
        if ready(port):
            webbrowser.open(f"http://127.0.0.1:{port}")
            return 0
        time.sleep(0.4)
    print(f"the server never answered on port {port}; nothing opened")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
