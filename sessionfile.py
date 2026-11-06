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

**A session is a file where the user put it.**  There is no store of sessions
by name under a dot-folder in ``$HOME``: saving asks where, every time, the way
a document does.  A folder of sessions is then something a person can see in
Finder, back up, and hand to someone else with the results beside it -- none of
which is true of ``~/.asfq/sessions``.  The file dialog is the list, so this
module has no ``listing`` and no ``delete``: the one that used to exist is gone
and so is the select box it fed.

**The client's own state is stored opaquely.**  Everything under ``ui`` is
written and read back without the server looking inside it.  The server knows
about tiles and poses because it holds them; it has no business knowing that
the aorta tab has a lesion threshold.  Adding a control later is then a change
to one file instead of three, and an older session simply restores the controls
it knew about.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

#: What a session file says it is.  The app was called "Fibrosis Quantifier"
#: until it grew the straightener's half of its name, and a file written then
#: is still one of ours -- so reading accepts either and only writing moves on.
APP_ID = "asfq"
APP_IDS = (APP_ID, "fibrosis-quantifier")

VERSION = 1

#: What the save dialog puts on a name that has no suffix, and what the open
#: dialog filters its listing by.  Doubled up deliberately: ``.json`` alone
#: would offer every layout, every results table and every figure preset in the
#: same folder as the session to open.
SUFFIX = ".asfq.json"

#: Sessions were kept here when the app managed them by name.  Nothing is
#: written to it any more and nothing is moved out of it; it is only where the
#: open dialog starts, so a session saved before the change is one click away
#: instead of lost.
OLD_STORE = Path.home() / ".asfq" / "sessions"


def old_store() -> Path | None:
    """The retired store, if it still holds anything worth opening."""
    try:
        return OLD_STORE if any(OLD_STORE.glob("*.json")) else None
    except OSError:
        return None


def save(path: str | Path, ui: dict[str, Any],
         stitch: dict[str, Any] | None = None) -> Path:
    """
    Write one session to the path the user chose.

    The name is theirs, not sanitized into something else: it arrives from the
    file dialog, where a slash makes a folder rather than a bad file name, and
    silently renaming a file someone has just named is how a save ends up
    somewhere they do not look for it.  The suffix is the one thing added, and
    only when it is missing.
    """
    path = Path(path).expanduser()
    if not path.name.lower().endswith(SUFFIX):
        path = path.with_name(path.name + SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "app": APP_ID,
        "version": VERSION,
        "name": path.name[: -len(SUFFIX)],
        "saved": time.time(),
        "stitch": stitch or None,
        "ui": ui or {},
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load(path: str | Path) -> dict[str, Any]:
    path = Path(str(path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there")
    data = json.loads(path.read_text())
    if data.get("app") not in APP_IDS:
        raise ValueError(f"{path.name} is not an ASFQ session")
    data.setdefault("name", path.name[: -len(SUFFIX)] if path.name.endswith(SUFFIX)
                    else path.stem)
    return data


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
