"""
Saving a whole working session, not just what it exported.

A section takes real handwork before it measures correctly: tiles dragged into
place, a trace redirected around a branch, marks dropped on features, foci
clicked away, a rule chosen, a figure styled.  All of that lived only in the
browser tab, so closing it threw the work away and left the PNGs behind with no
way back to the state that made them.  This keeps the state.

Two decisions shape the format.

**The pictures are referenced, never copied.**  A session records the source
folder and where the outputs were written, not the tiles and not the tables.
Sixty CZIs are a gigabyte and a half; the arrangement of them is a few hundred
poses.  So a session file stays small enough to keep dozens of, re-running an
analysis into the same folder updates every session that points at it, and a
session and its outputs cannot drift into disagreeing about the numbers -- the
only copy of a number is the one in the results folder.

**The client's own state is stored opaquely.**  Everything under ``ui`` is
written and read back without the server looking inside it.  The server knows
about tiles and poses because it holds them; it has no business knowing that
the aorta tab has a lesion threshold.  Adding a control later is then a change
to one file instead of three, and an older session simply restores the controls
it knew about.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

#: What a session file says it is.  The app was called "Fibrosis Quantifier"
#: until it grew the straightener's half of its name, and a file written then
#: is still one of ours -- so reading accepts either and only writing moves on.
APP_ID = "asfq"
APP_IDS = (APP_ID, "fibrosis-quantifier")

_STORE = Path.home() / ".asfq"
if not _STORE.exists():                       # sessions saved under the old name
    try:
        Path.home().joinpath(".fibrosis_quantifier").rename(_STORE)
    except OSError:                           # absent, or not ours to move
        pass
SESSION_DIR = _STORE / "sessions"
VERSION = 1


def sanitize(name: str) -> str:
    """
    A session name that is safe as a file name, and honest about being changed.

    A name with a slash in it does not reach a server-side check at all -- the
    router rejects the path before any handler runs -- so the caller has to be
    told what the name became rather than discovering a 404.  The client does
    the same substitution and says so; this is the backstop.
    """
    clean = re.sub(r"[^\w\-. ]+", "-", str(name or "").strip())
    clean = re.sub(r"\s+", " ", clean).strip(" .-")
    return clean[:80] or "session"


def _path(name: str) -> Path:
    return SESSION_DIR / f"{sanitize(name)}.json"


def save(name: str, ui: dict[str, Any], stitch: dict[str, Any] | None = None) -> Path:
    """Write one session.  Returns the path, whose stem is the cleaned name."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    payload = {
        "app": APP_ID,
        "version": VERSION,
        "name": path.stem,
        "saved": time.time(),
        "stitch": stitch or None,
        "ui": ui or {},
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load(name: str) -> dict[str, Any]:
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(f"no saved session called {path.stem!r}")
    data = json.loads(path.read_text())
    if data.get("app") not in APP_IDS:
        raise ValueError(f"{path.name} is not an ASFQ session")
    return data


def delete(name: str) -> bool:
    path = _path(name)
    if not path.exists():
        return False
    path.unlink()
    return True


def listing() -> list[dict[str, Any]]:
    """
    Every saved session, newest first, with enough to choose between them.

    The source folder is reported as it was recorded *and* as it stands now: a
    folder that has been moved or renamed is the one failure this feature has,
    and it is worth seeing before opening rather than as an error afterwards.
    """
    if not SESSION_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(SESSION_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if d.get("app") not in APP_IDS:
                continue
            folder = ((d.get("stitch") or {}).get("folder")) or None
            ui = d.get("ui") or {}
            out.append({
                "name": p.stem,
                "saved": d.get("saved"),
                "folder": folder,
                "folder_exists": bool(folder and Path(folder).exists()),
                "n_tiles": len((d.get("stitch") or {}).get("poses") or []),
                "tab": ui.get("tab"),
                "bytes": p.stat().st_size,
            })
        except Exception:
            continue          # a file we cannot read is not a session
    out.sort(key=lambda r: r.get("saved") or 0, reverse=True)
    return out


def stitch_state(sess: dict[str, Any]) -> dict[str, Any]:
    """
    The part of a stitching session worth keeping: where the tiles ended up.

    The tiles themselves are named, not stored, and matched back by name on
    open, so a layout survives a folder that has gained or lost a file.  The
    accepted matches come too -- they are what "re-solve, keep my locks" solves
    from, and without them reopening a session would silently turn that button
    into a full re-match.
    """
    ts = sess["ts"]
    return {
        "folder": str(sess.get("folder") or (ts.folder or "")),
        "params": asdict(sess["params"]) if sess.get("params") is not None else None,
        "tiles": [t.name for t in ts.tiles],
        "poses": [p.to_dict() for p in sess["poses"]],
        "matches": [m.to_dict() for m in (sess.get("matches") or [])],
    }
