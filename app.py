#!/usr/bin/env python3
"""
app.py -- ASFQ: the browser front end for the tile assembler
and the fibrosis analyses.

Run it with::

    python app.py

and open http://127.0.0.1:8765 .  Everything stays on this machine; the server
only reads the folders you point it at.

One tab assembles; the rest measure, one organ each:

* **Assemble** -- generic.  Point it at a folder of overlapping fields, press
  *Auto-stitch*, then drag whatever the software could not place and press
  *Align* to let correlation finish the job.  Nothing here knows or cares what
  the specimen is.
* **Aorta** -- the wall analysis: straighten the media, collagen per unit
  length of wall (cumulative and local), and blue staining through the muscle
  layer.
* **Heart**, **Kidney** -- fibrosis by area.  There is no wall to follow in
  either: interstitial fibrosis is diffuse, scattered between myocytes or
  around tubules, so the question becomes how much of the tissue is collagen.

Each organ tab carries its own batch at the bottom, because a tray of heart
sections and a tray of aortas are different runs with different readouts and
nothing is gained by making one page mean both.

The heavy lifting lives in ``stitch_core``, ``wall_analysis`` and
``fibrosis``; this file is transport and state.
"""

from __future__ import annotations

import io
import json
import math
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

import matplotlib
matplotlib.use("Agg")   # figures are written to files, never to a window

import stitch_core as SC
import wall_analysis as WA
import cohort as CO
import fibrosis as FB
import sessionfile as SF
from labkit import browse as LKB
from labkit import schemes as LKSC   # LKS is labkit.stats in cohort.py
from labkit.plots import PlotSpec

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.config["JSON_SORT_KEYS"] = False

SESSIONS: dict[str, dict[str, Any]] = {}
JOBS: dict[str, dict[str, Any]] = {}
# The WallResult and overview image behind each "..._rectangular_channels.png"
# a run has produced, keyed by that path -- so the figure can be redrawn with
# a different rotation, size or type scale without repeating the analysis
# that produced it. Keyed by the figure's own output path because the client
# already holds that string; nothing new to hand out or look up.
FIGURE_CACHE: dict[str, dict[str, Any]] = {}
STARTED = time.time()
_LOCK = threading.Lock()


# =============================================================================
# Jobs
# =============================================================================


def start_job(fn, label: str) -> str:
    """Run *fn(progress)* on a worker thread and hand back a job id."""
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"id": jid, "label": label, "progress": 0.0, "message": "starting",
                 "done": False, "error": None, "result": None, "t0": time.time()}

    def progress(frac: float, msg: str = "") -> None:
        job = JOBS.get(jid)
        if job is not None:
            job["progress"] = float(np.clip(frac, 0.0, 1.0))
            if msg:
                job["message"] = msg

    def run() -> None:
        try:
            JOBS[jid]["result"] = fn(progress)
            JOBS[jid]["progress"] = 1.0
            JOBS[jid]["message"] = "finished"
        except Exception as exc:  # surfaced to the browser, not swallowed
            JOBS[jid]["error"] = f"{type(exc).__name__}: {exc}"
            JOBS[jid]["traceback"] = traceback.format_exc()
        finally:
            JOBS[jid]["done"] = True
            JOBS[jid]["seconds"] = round(time.time() - JOBS[jid]["t0"], 1)

    threading.Thread(target=run, daemon=True).start()
    return jid


@app.route("/api/job/<jid>")
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    out = {k: v for k, v in job.items() if k not in ("result",)}
    if job["done"] and job["error"] is None:
        out["result"] = job["result"]
    return jsonify(out)


# =============================================================================
# Helpers
# =============================================================================


def _sess(sid: str) -> dict[str, Any]:
    s = SESSIONS.get(sid)
    if s is None:
        raise KeyError(f"session {sid} is not open")
    return s


def _poses_out(sess: dict[str, Any]) -> list[dict]:
    return [p.to_dict() for p in sess["poses"]]


def _tiles_out(sess: dict[str, Any]) -> list[dict]:
    ts: SC.TileSet = sess["ts"]
    return [t.to_dict() for t in ts.tiles]


def _qc_out(sess: dict[str, Any]) -> dict[str, Any]:
    ts: SC.TileSet = sess["ts"]
    df = SC.layout_qc(ts, sess["poses"])
    pairs = df.attrs.get("pairs", pd.DataFrame())
    conflicts = []
    if len(pairs):
        for r in pairs[pairs.conflict].itertuples():
            conflicts.append({"i": int(r.i), "j": int(r.j),
                              "ncc": None if not np.isfinite(r.ncc) else round(float(r.ncc), 3),
                              "mask_iou": None if not np.isfinite(r.mask_iou) else round(float(r.mask_iou), 3)})
    return {
        "tiles": [{"index": int(r["index"]), "score": None if not np.isfinite(r["score"]) else round(float(r["score"]), 3),
                   "n_neighbours": int(r["n_neighbours"]), "n_conflicts": int(r["n_conflicts"]),
                   "island": int(r["island"])}
                  for r in df.to_dict("records")],
        "conflicts": conflicts,
    }


def _png_response(arr: np.ndarray, fmt: str = "JPEG", quality: int = 88):
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, quality=quality)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg" if fmt == "JPEG" else "image/png")


def _params_from(d: dict, base: SC.StitchParams | None = None) -> SC.StitchParams:
    p = base or SC.StitchParams()
    known = {f for f in asdict(p)}
    clean = {}
    for k, v in (d or {}).items():
        if k in known and v is not None and v != "":
            cur = getattr(p, k)
            try:
                clean[k] = type(cur)(v) if cur is not None else v
            except (TypeError, ValueError):
                clean[k] = v
    return p.copy_with(**clean)


# =============================================================================
# Static
# =============================================================================


@app.route("/")
def index():
    return send_from_directory(str(HERE / "static"), "index.html")


@app.route("/api/health")
def health():
    """
    Say who is answering on this port, and since when.

    The launcher asks this before it starts anything.  A busy port is usually
    this app left running from a previous day: it goes on serving today's HTML
    and JavaScript straight off disk while running that day's Python behind it,
    so the interface looks current and then fails on a route the old server
    never had.  That reads as a broken app rather than a stale one.  A name and
    a start time are enough to tell the two apart, and to tell either from a
    genuinely different server that happens to want the same port.
    """
    return jsonify({"app": "asfq",
                    "pid": os.getpid(),
                    "started": STARTED,
                    "uptime_s": round(time.time() - STARTED, 1)})


# =============================================================================
# Browsing the filesystem (local app; read-only listing)
# =============================================================================


# The listing itself is labkit's, so this dialog is the same dialog as in the
# other five tools.  The blueprint answers GET and POST alike, which is what
# let the front end move to the shared browser without the endpoint changing
# shape underneath it.
app.register_blueprint(LKB.flask_blueprint())
# Colours by genotype, saved under ~/.labkit and shared with the other tools:
# "the KO is this red" is a decision about a paper, and a paper's panels come
# out of more than one of these programs.
app.register_blueprint(LKSC.flask_blueprint())


# =============================================================================
# Sessions
# =============================================================================


@app.route("/api/session/create", methods=["POST"])
def session_create():
    body = request.json or {}
    folder = Path(str(body.get("folder", ""))).expanduser()
    if not folder.is_dir():
        return jsonify({"error": f"{folder} is not a folder"}), 400
    params = _params_from(body.get("params", {}))
    sid = uuid.uuid4().hex[:12]

    def work(progress):
        ts = SC.TileSet(params).load_folder(folder, progress=progress)
        with _LOCK:
            SESSIONS[sid] = {
                "id": sid, "ts": ts, "folder": str(folder), "params": params,
                "poses": [SC.Pose(x=0.0, y=0.0, island=i, placed=True) for i in range(len(ts))],
                "matches": [], "stats": {}, "analysis": None, "mosaic_cache": {},
            }
        # Lay the unplaced tiles out in a grid so the canvas is not a pile.
        _grid_layout(SESSIONS[sid])
        return {"sid": sid, "n_tiles": len(ts), "px_um": ts.px_um,
                "tile_w": ts.tiles[0].width, "tile_h": ts.tiles[0].height}

    return jsonify({"job": start_job(work, f"loading {folder.name}"), "sid": sid})


def _grid_layout(sess: dict[str, Any]) -> None:
    ts: SC.TileSet = sess["ts"]
    n = len(ts)
    cols = max(1, int(np.ceil(np.sqrt(n))))
    for i, t in enumerate(ts.tiles):
        r, c = divmod(i, cols)
        sess["poses"][i] = SC.Pose(x=c * t.width * 1.06, y=r * t.height * 1.06,
                                   island=i, placed=True)


@app.route("/api/session/<sid>")
def session_get(sid: str):
    try:
        sess = _sess(sid)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    ts: SC.TileSet = sess["ts"]
    return jsonify({
        "sid": sid, "folder": sess["folder"], "n_tiles": len(ts),
        "px_um": ts.px_um, "tiles": _tiles_out(sess), "poses": _poses_out(sess),
        "stats": sess.get("stats", {}), "params": asdict(sess["params"]),
        "has_matches": bool(sess.get("matches")),
    })


@app.route("/api/session/<sid>/thumb/<int:idx>.jpg")
def session_thumb(sid: str, idx: int):
    sess = _sess(sid)
    ts: SC.TileSet = sess["ts"]
    if not (0 <= idx < len(ts)):
        return jsonify({"error": "bad tile index"}), 404
    return _png_response(ts.tiles[idx].thumb, "JPEG", quality=82)


@app.route("/api/session/<sid>/autostitch", methods=["POST"])
def session_autostitch(sid: str):
    sess = _sess(sid)
    body = request.json or {}
    params = _params_from(body.get("params", {}), sess["params"])
    sess["params"] = params
    keep_locked = bool(body.get("keep_locked", True))
    ts: SC.TileSet = sess["ts"]

    def work(progress):
        def sub(lo, hi):
            return lambda f, m: progress(lo + (hi - lo) * f, m)

        matches = sess.get("matches") or []
        if not matches or body.get("rematch", True):
            matches = SC.match_all_pairs(ts, params, progress=sub(0.0, 0.55))
            matches = SC.refine_matches(ts, matches, params, progress=sub(0.55, 0.85))
            sess["matches"] = matches
        progress(0.88, "solving the layout")
        locked = None
        if keep_locked:
            locked = {i: p for i, p in enumerate(sess["poses"]) if p.locked}
            locked = locked or None
        poses, stats = SC.solve_layout(ts, matches, params, locked=locked)
        sess["poses"] = poses
        sess["stats"] = stats
        sess["mosaic_cache"] = {}
        progress(0.97, "checking the result")
        qc = _qc_out(sess)
        return {"poses": _poses_out(sess), "stats": _jsonable(stats), "qc": qc}

    return jsonify({"job": start_job(work, "auto-stitching")})


def _jsonable(x: Any) -> Any:
    """
    Make a value safe for `jsonify`.

    Non-finite floats become null.  This is not cosmetic: Python writes them as
    the bare tokens `NaN` and `Infinity`, which are not valid JSON, so a single
    NaN anywhere in a result makes `response.json()` throw in the browser and the
    whole page silently shows nothing.
    """
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


@app.route("/api/session/<sid>/poses", methods=["POST"])
def session_set_poses(sid: str):
    """Accept the browser's canvas state after a drag."""
    sess = _sess(sid)
    body = request.json or {}
    incoming = body.get("poses") or []
    for d in incoming:
        i = int(d["index"])
        if 0 <= i < len(sess["poses"]):
            sess["poses"][i] = SC.Pose(
                x=float(d.get("x", 0.0)), y=float(d.get("y", 0.0)),
                theta=float(d.get("theta", 0.0)), island=int(d.get("island", 0)),
                placed=bool(d.get("placed", True)), locked=bool(d.get("locked", False)),
            )
    sess["mosaic_cache"] = {}
    if body.get("qc"):
        return jsonify({"ok": True, "qc": _qc_out(sess)})
    return jsonify({"ok": True})


@app.route("/api/session/<sid>/snap", methods=["POST"])
def session_snap(sid: str):
    """Automatic alignment after a manual drag -- the other half of the ask."""
    sess = _sess(sid)
    body = request.json or {}
    ts: SC.TileSet = sess["ts"]
    indices = body.get("indices")
    search_frac = float(body.get("search_frac", 0.22))
    passes = int(body.get("passes", 2))

    def work(progress):
        if indices:
            cur, detail = SC.snap_selection(
                ts, sess["poses"], [int(i) for i in indices], sess["params"],
                search_frac=search_frac, passes=3, progress=progress)
            sess["poses"] = cur
            info_all = _jsonable(detail)
        else:
            sess["poses"] = SC.snap_all(ts, sess["poses"], sess["params"],
                                        passes=passes, search_frac=min(search_frac, 0.12),
                                        progress=progress)
            info_all = []
        sess["mosaic_cache"] = {}
        return {"poses": _poses_out(sess), "qc": _qc_out(sess), "detail": info_all}

    return jsonify({"job": start_job(work, "aligning")})


@app.route("/api/session/<sid>/snap_island", methods=["POST"])
def session_snap_island(sid: str):
    """Settle a hand-dropped island as a rigid body against what surrounds it."""
    sess = _sess(sid)
    body = request.json or {}
    ts: SC.TileSet = sess["ts"]
    island = int(body.get("island", 0))
    search_frac = float(body.get("search_frac", 0.22))

    def work(progress):
        progress(0.1, f"aligning island {island}")
        poses, info = SC.snap_island(ts, sess["poses"], island, sess["params"],
                                     search_frac=search_frac)
        sess["poses"] = poses
        sess["mosaic_cache"] = {}
        progress(0.9, "checking the result")
        return {"poses": _poses_out(sess), "qc": _qc_out(sess), "info": _jsonable(info)}

    return jsonify({"job": start_job(work, "aligning island")})


@app.route("/api/session/<sid>/snap_islands", methods=["POST"])
def session_snap_islands(sid: str):
    """Settle every island against the others, each staying rigid."""
    sess = _sess(sid)
    body = request.json or {}
    ts: SC.TileSet = sess["ts"]
    search_frac = float(body.get("search_frac", 0.12))
    passes = int(body.get("passes", 4))

    def work(progress):
        poses, info = SC.snap_islands(ts, sess["poses"], sess["params"],
                                      passes=passes, search_frac=search_frac,
                                      progress=progress)
        sess["poses"] = poses
        sess["mosaic_cache"] = {}
        return {"poses": _poses_out(sess), "qc": _qc_out(sess), "info": _jsonable(info)}

    return jsonify({"job": start_job(work, "settling islands")})


@app.route("/api/session/<sid>/qc")
def session_qc(sid: str):
    return jsonify(_qc_out(_sess(sid)))


@app.route("/api/session/<sid>/suggestions", methods=["POST"])
def session_suggestions(sid: str):
    """Merges the assembler believes in but will not perform unasked."""
    sess = _sess(sid)
    ts: SC.TileSet = sess["ts"]
    if not sess.get("matches"):
        return jsonify({"suggestions": []})
    poses = [SC.Pose(**asdict(p)) for p in sess["poses"]]
    _p, info = SC.merge_islands(ts, poses, sess["matches"], sess["params"], apply=False)
    return jsonify({"suggestions": _jsonable(info.get("suggestions", []))})


@app.route("/api/session/<sid>/apply_shift", methods=["POST"])
def session_apply_shift(sid: str):
    """Move one island bodily -- used by the suggestion panel and by island drags."""
    sess = _sess(sid)
    body = request.json or {}
    island = int(body.get("island", -1))
    dx, dy = float(body.get("dx", 0.0)), float(body.get("dy", 0.0))
    dth = float(body.get("dtheta", 0.0))
    into = body.get("into_island")
    for p in sess["poses"]:
        if p.island == island and not p.locked:
            p.x += dx
            p.y += dy
            p.theta += dth
    if into is not None:
        for p in sess["poses"]:
            if p.island == island:
                p.island = int(into)
    sess["mosaic_cache"] = {}
    return jsonify({"poses": _poses_out(sess), "qc": _qc_out(sess)})


@app.route("/api/session/<sid>/merge_islands", methods=["POST"])
def session_merge_islands(sid: str):
    """
    Declare that these islands are one piece, leaving every tile where it is.

    The automatic merge is a judgement -- `snap_island` joins an island only
    when the result contradicts nothing, and the suggestion panel offers a
    shift it worked out itself.  Neither can help when the evidence is
    genuinely ambiguous and you can nonetheless see where the piece goes.
    This is the override: position by hand, then say so.  Nothing moves, so
    there is nothing for it to get wrong; it only relabels, and the quality
    check still reports whatever conflicts the result has rather than being
    silenced by the decision.
    """
    sess = _sess(sid)
    body = request.json or {}
    islands = {int(i) for i in (body.get("islands") or [])}
    if not islands:
        islands = {int(sess["poses"][int(i)].island)
                   for i in (body.get("indices") or [])
                   if 0 <= int(i) < len(sess["poses"])}
    if len(islands) < 2:
        return jsonify({"error": "select tiles from at least two islands to merge"}), 400

    counts: dict[int, int] = {}
    for p in sess["poses"]:
        if p.island in islands:
            counts[int(p.island)] = counts.get(int(p.island), 0) + 1
    # The biggest piece keeps its name, so the mosaic's main island stays
    # island 0 and the export and analysis pickers do not renumber underneath.
    target = max(counts, key=lambda k: (counts[k], -k))
    moved = 0
    for p in sess["poses"]:
        if p.island in islands and int(p.island) != target:
            p.island = target
            moved += 1
    sess["mosaic_cache"] = {}
    return jsonify({"poses": _poses_out(sess), "qc": _qc_out(sess),
                    "info": {"target": target, "merged": sorted(islands - {target}),
                             "n_tiles_moved": moved}})


@app.route("/api/session/<sid>/mosaic.jpg")
def session_mosaic(sid: str):
    sess = _sess(sid)
    ts: SC.TileSet = sess["ts"]
    scale = float(request.args.get("scale", 0.12))
    island = request.args.get("island")
    mode = request.args.get("mode", "feather")
    idxs = None
    if island not in (None, "", "all"):
        want = int(island)
        idxs = [i for i, p in enumerate(sess["poses"]) if p.island == want]
    key = (round(scale, 4), island, mode)
    cached = sess["mosaic_cache"].get(key)
    if cached is None:
        cached = SC.render_mosaic(ts, sess["poses"], scale=scale, mode=mode, indices=idxs)
        sess["mosaic_cache"][key] = cached
    return _png_response(cached, "JPEG", quality=90)


@app.route("/api/session/<sid>/export", methods=["POST"])
def session_export(sid: str):
    sess = _sess(sid)
    body = request.json or {}
    ts: SC.TileSet = sess["ts"]
    out_dir = Path(str(body.get("out_dir") or (Path(sess["folder"]) / "stitched"))).expanduser()
    name = str(body.get("name") or Path(sess["folder"]).name).replace("/", "_")
    scale = float(body.get("scale", 0.5))
    island = body.get("island")
    mode = str(body.get("mode", "feather"))
    fmt = str(body.get("format", "png")).lower()

    def work(progress):
        out_dir.mkdir(parents=True, exist_ok=True)
        idxs = None
        if island not in (None, "", "all"):
            idxs = [i for i, p in enumerate(sess["poses"]) if p.island == int(island)]
        mo = SC.render_mosaic(ts, sess["poses"], scale=scale, mode=mode, indices=idxs,
                              progress=progress)
        img_path = out_dir / f"{name}_mosaic.{'tif' if fmt in ('tif', 'tiff') else 'png'}"
        arr = (np.clip(mo, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_path)
        lay = SC.save_layout(out_dir / f"{name}_layout.json", ts, sess["poses"],
                             sess.get("matches"), sess["params"],
                             extra={"stats": _jsonable(sess.get("stats", {}))})
        qc = SC.layout_qc(ts, sess["poses"])
        qc.to_csv(out_dir / f"{name}_tile_quality.csv", index=False)
        info = SC.mosaic_info(ts, sess["poses"], scale, idxs)
        (out_dir / f"{name}_mosaic_info.json").write_text(json.dumps(_jsonable(info), indent=1))
        return {"mosaic": str(img_path), "layout": str(lay),
                "quality": str(out_dir / f"{name}_tile_quality.csv"),
                "px_um": (ts.px_um / scale) if ts.px_um else None,
                "shape": [int(mo.shape[0]), int(mo.shape[1])]}

    return jsonify({"job": start_job(work, "exporting mosaic")})


@app.route("/api/session/<sid>/save_layout", methods=["POST"])
def session_save_layout(sid: str):
    sess = _sess(sid)
    body = request.json or {}
    path = Path(str(body.get("path") or (Path(sess["folder"]) / "layout.json"))).expanduser()
    p = SC.save_layout(path, sess["ts"], sess["poses"], sess.get("matches"), sess["params"])
    return jsonify({"path": str(p), "csv": str(Path(p).with_suffix(".csv"))})


@app.route("/api/session/<sid>/load_layout", methods=["POST"])
def session_load_layout(sid: str):
    sess = _sess(sid)
    body = request.json or {}
    # Either from a layout file, or handed the arrangement directly -- which is
    # what reopening a saved session does, and it should not have to write a
    # temporary file to say something it is already holding.
    if body.get("layout"):
        data = dict(body["layout"])
        data["poses"] = [SC.Pose(**p) for p in data.get("poses", [])]
        data["matches"] = [SC.PairMatch(**m) for m in data.get("matches", [])]
        names = data.get("tiles") or []
        data["tiles"] = [{"name": n} if isinstance(n, str) else n for n in names]
    else:
        path = Path(str(body.get("path", ""))).expanduser()
        if not path.exists():
            return jsonify({"error": f"{path} not found"}), 400
        data = SC.load_layout(path)
    ts: SC.TileSet = sess["ts"]
    by_name = {t.name: t.index for t in ts.tiles}
    poses = list(sess["poses"])
    n_set = 0
    for tile, pose in zip(data.get("tiles", []), data.get("poses", [])):
        idx = by_name.get(tile.get("name"))
        if idx is not None:
            poses[idx] = pose
            n_set += 1
    sess["poses"] = poses
    if data.get("matches"):
        sess["matches"] = data["matches"]
    sess["mosaic_cache"] = {}
    return jsonify({"poses": _poses_out(sess), "n_restored": n_set, "qc": _qc_out(sess)})


# =============================================================================
# Saved sessions
# =============================================================================
#
# A layout is where the tiles are.  A *session* is that plus everything a person
# decided afterwards -- the redirected trace, the marks, the foci clicked away,
# which rule the media was measured by, how the figure was styled -- and until
# now none of it outlived the tab.  See `sessionfile` for what is kept and what
# is only pointed at.


@app.route("/api/sessions")
def sessions_list():
    return jsonify({"sessions": SF.listing(), "dir": str(SF.SESSION_DIR)})


@app.route("/api/sessions/save", methods=["POST"])
def sessions_save():
    body = request.json or {}
    name = str(body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "a session needs a name"}), 400
    sid = body.get("sid")
    stitch = None
    if sid:
        try:
            stitch = SF.stitch_state(_sess(sid))
        except KeyError:
            stitch = None      # the tab outlived the server; keep the rest
    path = SF.save(name, body.get("ui") or {}, stitch)
    return jsonify({"path": str(path), "name": path.stem,
                    "renamed": path.stem != name, "bytes": path.stat().st_size})


@app.route("/api/sessions/open", methods=["POST"])
def sessions_open():
    body = request.json or {}
    try:
        data = SF.load(str(body.get("name") or ""))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    st = data.get("stitch") or {}
    folder = st.get("folder")
    # Said plainly and up front, because it is the one thing that goes wrong:
    # the images are referenced, so a folder that has moved cannot be restored
    # and the client must not start clearing the tab on the strength of it.
    if folder and not Path(folder).is_dir():
        return jsonify({"error": f"the images have moved: {folder} is not there any more",
                        "folder": folder, "ui": data.get("ui") or {}}), 409
    return jsonify({"name": data.get("name"), "saved": data.get("saved"),
                    "stitch": st or None, "ui": data.get("ui") or {}})


@app.route("/api/sessions/delete", methods=["POST"])
def sessions_delete():
    body = request.json or {}
    ok = SF.delete(str(body.get("name") or ""))
    return jsonify({"deleted": bool(ok)})


# =============================================================================
# Wall analysis
# =============================================================================


def _outline_for_editor(res: Any, paths: dict[str, Any], max_points: int = 700
                        ) -> dict[str, Any]:
    """
    What the outline editor -- and the figure canvas -- need: the picture, the
    curve on it, the scale that relates the two, and enough of what the map
    figure draws (arc length, branches, the other lumina) to redraw the same
    annotations in the browser instead of only in the saved PNG.

    The curve is thinned to something a browser can drag around -- a 4 mm wall
    is four thousand points at the analysis step, and none of that resolution
    survives being drawn by hand anyway.  Everything goes out in microns, which
    is the one frame that means the same thing in the editor, in the analysis
    and in the next run at a different render scale.
    """
    xy = np.asarray(res.centerline_xy, float) * res.px_um
    step = max(1, len(xy) // max_points)
    arc_um = np.asarray(res.arc_um, float)

    this_rank = int(res.totals.get("lumen_rank", -1))
    others = [
        {
            "xy": [float(l["x_px"]) * res.px_um, float(l["y_px"]) * res.px_um],
            "label": f"vessel {int(l['rank']) + 1}" if l.get("vessel") else "not a vessel",
            "vessel": bool(l.get("vessel")),
        }
        for l in (getattr(res, "lumina", None) or [])
        if int(l.get("rank", -1)) != this_rank
    ]
    branches_um = [[float(b["x_px"]) * res.px_um, float(b["y_px"]) * res.px_um]
                   for b in (getattr(res, "branches", None) or [])]

    # A crop around the traced vessel, in microns -- a section is mostly empty
    # slide, and a map of the whole mosaic leaves the wall too small to read.
    # The client clamps this to the picture it actually has, so an unusually
    # small or offset crop here can never ask it to draw outside the image.
    pad_um = max(float(np.nanmax(res.tissue_outer_um)) * 1.6, 300.0)
    x0 = float(xy[:, 0].min()) - pad_um
    x1 = float(xy[:, 0].max()) + pad_um
    y0 = float(xy[:, 1].min()) - pad_um
    y1 = float(xy[:, 1].max()) + pad_um

    return {
        "um": _jsonable(xy[::step]),
        "arc_mm": _jsonable(arc_um[::step] / 1000.0),
        "closed": bool(res.closed),
        "image": str(paths.get("overview", "")),
        "image_px_um": float(paths.get("overview_px_um", res.px_um)),
        "crop_um": [x0, y0, x1, y1],
        "branches_um": _jsonable(branches_um),
        "others": _jsonable(others),
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Straighten a vessel wall and measure it.

    The image can come from the current stitching session (rendered at the
    requested scale) or from any file on disk, so the analysis is usable on
    single fields and on mosaics made elsewhere.
    """
    body = request.json or {}
    sid = body.get("sid")
    wp = WA.WallParams()
    for k, v in (body.get("params") or {}).items():
        if k in asdict(wp) and v not in (None, ""):
            cur = getattr(wp, k)
            try:
                setattr(wp, k, type(cur)(v) if cur is not None else float(v))
            except (TypeError, ValueError):
                setattr(wp, k, v)
    name = str(body.get("name") or "wall")
    out_dir = Path(str(body.get("out_dir") or (HERE / "analysis_out"))).expanduser()
    render_scale = float(body.get("render_scale", 0.35))
    island = body.get("island", 0)
    # A hand-drawn outline, in microns from the top-left of the image.  Microns
    # rather than pixels so the drawing survives a change of render scale or of
    # analysis resolution between the run that produced it and the run that uses
    # it -- the editor works on a shrunken overview, and pixels there mean
    # nothing anywhere else.
    # One outline names one wall.  Several outlines name several, and are
    # measured exactly like several detected vessels -- each gets its own
    # lumen_rank, so the vessel picker, the per-vessel file names and the
    # cohort pooling all treat a hand-drawn second vessel as a vessel rather
    # than as a special case.  A lone `guide` is still accepted as it was.
    guides = body.get("guides") or None
    if not guides and body.get("guide"):
        guides = [{"um": body["guide"], "closed": body.get("guide_closed")}]
    guides = [g if isinstance(g, dict) else {"um": g, "closed": None}
              for g in (guides or [])]
    guides = [g for g in guides if g.get("um")]
    # The vessels the editor is *not* touching.  They are re-detected rather
    # than re-drawn from the outlines already on screen: a curve does not
    # survive a round trip through the re-centring untouched -- feeding one
    # straight back moved its traced length by 5-11 % and its blue area by a
    # quarter -- so an untouched vessel would silently change its numbers every
    # time its neighbour was edited.  Running it again automatically is the
    # same call that produced it, so it comes back identical.
    keep_ranks = sorted({int(k) for k in (body.get("keep_ranks") or [])})
    # Marks placed by hand, in the same microns as a drawn outline, and the two
    # knobs the lesion search has.  Both travel with the analysis rather than
    # being a second pass, because a lesion is measured off the same maps the
    # wall is.
    marks = body.get("points") or None
    lesion_k = float(body.get("lesion_k_mad") or 3.0)
    lesion_min = float(body.get("lesion_min_area_um2") or 2000.0)
    # A person's own verdict on a candidate lesion, kept apart from the knobs
    # above on purpose: those change what the detector looks for everywhere in
    # the section, and a click here removes one specific patch someone has
    # actually looked at without moving anything else.
    reject_lesions = body.get("exclude_lesions_at") or None

    def work(progress):
        if sid:
            sess = _sess(sid)
            ts: SC.TileSet = sess["ts"]
            idxs = None
            if island not in (None, "", "all"):
                idxs = [i for i, p in enumerate(sess["poses"]) if p.island == int(island)]
            progress(0.05, "rendering the mosaic")
            rgb = SC.render_mosaic(ts, sess["poses"], scale=render_scale, indices=idxs)
            px_um = (ts.px_um or 1.0) / render_scale
            title = Path(sess["folder"]).name
        else:
            path = Path(str(body.get("image", ""))).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            arr, px = SC.read_rgb(path)
            rgb = SC.to_float_rgb(arr)
            px_um = float(body.get("px_um") or px or 1.0)
            title = path.name

        progress(0.15, "analysing the wall")
        # Every vessel the section encloses, not just the biggest: a section
        # that catches the aorta twice has two walls in it, and measuring one
        # and silently dropping the other is how half a section goes missing.
        if guides:
            # A drawn outline is the enumeration for the vessel it names; the
            # rest of the section is still found the usual way.  Editing one
            # wall used to delete every other wall in the section, because the
            # outlines were taken as the whole answer.
            #
            # `keep_all_vessels` because that is what `analyze_vessels` below
            # runs with, and a kept vessel has to come back the same as it did
            # there -- the band scoring is what decides whether its component
            # is in the media mask at all.
            jobs = [(int(g.get("rank", gi)), g) for gi, g in enumerate(guides)]
            jobs += [(k, None) for k in keep_ranks
                     if k not in {r for r, _ in jobs}]
            jobs.sort(key=lambda j: j[0])
            wp_all = wp.copy_with(keep_all_vessels=True)
            results = []
            for gi, (rank, g) in enumerate(jobs):
                lo = 0.15 + 0.65 * gi / len(jobs)
                hi = 0.15 + 0.65 * (gi + 1) / len(jobs)
                results.append(WA.analyze_wall(
                    rgb, px_um, wp_all, lumen_rank=rank,
                    guide_um=np.asarray(g["um"], dtype=float) if g else None,
                    guide_closed=g.get("closed") if g else None,
                    points_um=marks, lesion_k_mad=lesion_k,
                    lesion_min_area_um2=lesion_min,
                    exclude_lesions_at=reject_lesions,
                    progress=lambda f, m, lo=lo, hi=hi: progress(lo + (hi - lo) * f, m)))
        else:
            results = WA.analyze_vessels(
                rgb, px_um, wp, points_um=marks, lesion_k_mad=lesion_k,
                lesion_min_area_um2=lesion_min, exclude_lesions_at=reject_lesions,
                progress=lambda f, m: progress(0.15 + 0.65 * f, m))
        progress(0.85, "drawing")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for res_i in results:
            rank = int(res_i.totals.get("lumen_rank", 0))
            stem = name if rank == 0 else f"{name}_vessel{rank + 1}"
            ov_scale = min(1.0, 900.0 / max(res_i.image_shape))
            overview = SC.resize(rgb, (px_um / res_i.px_um) * ov_scale, "area")
            overview = np.clip(overview, 0, 1)
            pth = WA.save_results(res_i, out_dir, stem, overview_rgb=overview)
            FIGURE_CACHE[pth["rectangular_channels"]] = {"res": res_i, "overview_rgb": overview}
            # The same overview the figure uses, saved on its own so the outline
            # editor can draw on the picture rather than on a blank canvas.
            ov_path = out_dir / f"{stem}_overview.jpg"
            Image.fromarray((overview * 255).astype(np.uint8)).save(ov_path, quality=88)
            pth["overview"] = str(ov_path)
            pth["overview_px_um"] = res_i.px_um / max(ov_scale, 1e-9)
            out.append((res_i, pth, rank))
        res, paths, _ = out[0]

        prof = res.profile
        step = max(1, len(prof) // 1200)
        series = _jsonable({
            "arc_mm": prof.arc_um.values[::step] / 1000.0,
            "thickness_um": prof.media_thickness_um.values[::step],
            "collagen_per_length": prof.collagen_od_um.values[::step],
            "collagen_adventitia": prof.collagen_adventitia_od_um.values[::step],
            "cum_collagen": prof.cum_collagen_od_um2.values[::step] / 1000.0,
            "blue_area_pct": prof.collagen_area_fraction.values[::step] * 100,
        })
        depth_rel = _jsonable({
            "x": res.depth_profile_rel.depth_relative.values,
            "muscle": res.depth_profile_rel.muscle_mean.values,
            "collagen": res.depth_profile_rel.collagen_mean.values,
        })
        vessels = [{
            "rank": rank,
            "label": f"Vessel {rank + 1}",
            "totals": _jsonable(r_i.totals),
            "lesions": _jsonable(r_i.lesions),
            "lesion_outlines": _jsonable(
                FB.lesion_outlines(r_i.lesion_labels, r_i.px_um)
                if r_i.lesion_labels is not None else []),
            "points": _jsonable(r_i.points),
            "paths": {k: str(v) for k, v in p_i.items()},
            "rectangular_channels_png": str(p_i["rectangular_channels"]),
            "rectangular_channels_svg": str(p_i["rectangular_channels_svg"]),
            "straight_png": str(p_i["straightened"]),
            "outline": _outline_for_editor(r_i, p_i),
        } for r_i, p_i, rank in out]
        return {
            "totals": _jsonable(res.totals), "paths": {k: str(v) for k, v in paths.items()},
            "lesions": _jsonable(res.lesions), "points": _jsonable(res.points),
            "lesion_outlines": _jsonable(
                FB.lesion_outlines(res.lesion_labels, res.px_um)
                if res.lesion_labels is not None else []),
            "series": series, "depth_relative": depth_rel,
            "straight_png": str(paths["straightened"]),
            "rectangular_png": str(paths["rectangular"]),
            "rectangular_channels_png": str(paths["rectangular_channels"]),
            "rectangular_channels_svg": str(paths["rectangular_channels_svg"]),
            "vessels": vessels,
            "lumina": _jsonable(getattr(res, "lumina", []) or []),
            "outline": _outline_for_editor(res, paths),
            "title": title,
        }

    return jsonify({"job": start_job(work, "wall analysis")})


@app.route("/api/replot", methods=["POST"])
def replot():
    """
    Redraw a figure an analysis run already produced, with a different
    rotation, size or type scale -- without repeating the analysis.

    Restyling is synchronous rather than a job: plot_rectangular only reads
    fields a finished WallResult already has, so this is matplotlib drawing
    time, not measurement time, and it stays fast enough not to need progress
    reporting or a job id of its own.
    """
    body = request.json or {}
    path = str(body.get("path", ""))
    entry = FIGURE_CACHE.get(path)
    if entry is None:
        return jsonify({
            "error": "This figure's data isn't cached any more (the server "
                     "restarted, or it's from an old run) -- rerun the analysis "
                     "to restyle it.",
        }), 404

    style = body.get("style") or {}

    def fnum(key: str, default: float) -> float:
        v = style.get(key)
        if v in (None, ""):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Display-only stain gains: the picture is redrawn with a stain turned
    # down, and nothing measured goes near them.  Three numbers so the same
    # control works for a trichrome's third channel if it is ever wanted;
    # in practice only the first, the red, gets moved.
    gain = (fnum("gain_muscle", 1.0), fnum("gain_collagen", 1.0), fnum("gain_nuclei", 1.0))

    # The control reads in points of the largest text in the figure (the
    # panel titles) rather than as a bare multiplier, so font_scale is
    # derived from it here rather than sent as its own field.
    title_pt = max(4.0, fnum("title_pt", WA.FS_TITLE))
    font_scale = title_pt / WA.FS_TITLE

    import matplotlib.pyplot as plt

    # An area analysis is a different figure with the same question behind the
    # controls: turn the section, set the type scale, keep or drop the boxes.
    if "fib" in entry:
        p = Path(path)
        kw = dict(
            rotation_deg=fnum("rotation_deg", 0.0),
            font_scale=max(4.0, fnum("title_pt", WA.FS_TITLE)) / WA.FS_TITLE,
            show_grid=bool(style.get("show_grid", True)),
            show_spines=bool(style.get("show_spines", True)),
            panel_in=fnum("panel_in", 3.0),
            stain_gain=gain,
        )
        fig = FB.plot_fibrosis(entry["fib"], title=entry.get("title", ""),
                               organ=entry.get("organ", ""), **kw)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        with plt.rc_context({"svg.fonttype": "none"}):
            fig = FB.plot_fibrosis(entry["fib"], title=entry.get("title", ""),
                                   organ=entry.get("organ", ""), svg=True, **kw)
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
            plt.close(fig)
        return jsonify({"ok": True})

    kw = dict(
        rotation_deg=fnum("rotation_deg", 0.0),
        font_scale=font_scale,
        show_grid=bool(style.get("show_grid", True)),
        show_spines=bool(style.get("show_spines", True)),
        row_height=fnum("row_height_in", 1.7),
        left_width=fnum("left_width_in", 9.0),
        right_width=fnum("right_width_in", 2.5),
        show_trace=bool(style.get("show_trace", True)),
        stain_gain=gain,
    )
    p = Path(path)
    fig = WA.plot_rectangular(entry["res"], overview_rgb=entry["overview_rgb"], **kw)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    # See save_results for why the SVG is a separate render rather than a
    # second savefig of the same figure.
    fig_svg = WA.plot_rectangular(entry["res"], overview_rgb=entry["overview_rgb"], svg=True, **kw)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig_svg.savefig(p.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig_svg)
    return jsonify({"ok": True})


@app.route("/api/file")
def get_file():
    """Serve a produced figure or CSV back to the page."""
    path = Path(str(request.args.get("path", ""))).expanduser()
    if not path.exists() or not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(path))


# =============================================================================
# Batch
# =============================================================================


@app.route("/api/cohort/scan", methods=["POST"])
def cohort_scan():
    """
    Every section folder under one parent, with what its name says about it.

    The tray is the natural unit: one parent folder, one sub-folder per
    section.  What the parser made of each name comes back with it, so a
    mis-read is something you see before the batch runs rather than something
    you discover in the group means.
    """
    body = request.json or {}
    raw = body.get("path") or str(Path.home())
    organ = str(body.get("organ") or "aorta")
    parent = Path(raw).expanduser()
    if not parent.exists():
        return jsonify({"error": f"{parent} does not exist"}), 400
    if parent.is_file():
        parent = parent.parent
    # Only the parent itself being unreadable is fatal.  A sub-folder that
    # refuses to be listed is reported and stepped over: every home directory
    # has a .Trash or a cloud-drive mount, and one of those must not be able to
    # stop you browsing to your data.
    skipped: list[str] = []
    try:
        dirs = CO.list_children(parent, list_images=SC.list_images,
                                skipped=skipped, organ=organ)
    except PermissionError:
        return jsonify({"error": f"cannot read {parent} — permission denied"}), 400
    except OSError as exc:
        return jsonify({"error": f"cannot read {parent} — {exc.strerror or exc}"}), 400
    return jsonify({
        "path": str(parent),
        "parent": str(parent.parent),
        "has_metadata_csv": (parent / "metadata.csv").exists(),
        # The vocabulary the names were read against, so the browser can offer
        # it when a region is typed in by hand instead of read out of a name.
        "segments": CO.segment_names(organ),
        "dirs": dirs,
        "sections": [d for d in dirs if d["is_section"]],
        "skipped": skipped,
    })


@app.route("/api/cohort/meta", methods=["POST"])
def cohort_meta():
    """
    Say by hand what a folder name did not: the region these sections came from.

    Written into the ``metadata.csv`` beside each section, which every later
    step already reads, so a region typed here reaches the scan, the batch and
    the pooling by the one path the file-based overrides always took.  Sections
    ticked across several trays are written to each tray's own file.

    A blank value clears the override rather than setting an empty region --
    the folder name gets its say back.
    """
    body = request.json or {}
    folders = [Path(str(f)).expanduser() for f in (body.get("folders") or [])]
    fields = {str(k): str(v) for k, v in (body.get("fields") or {}).items()
              if k in CO.OVERRIDE_FIELDS}
    if not folders:
        return jsonify({"error": "no sections chosen"}), 400
    if not fields:
        return jsonify({"error": f"nothing to set: {', '.join(CO.OVERRIDE_FIELDS)}"}), 400
    by_parent: dict[Path, dict[str, dict[str, str]]] = {}
    for folder in folders:
        by_parent.setdefault(folder.parent, {})[folder.name] = fields
    try:
        written = [str(CO.write_overrides(parent, updates))
                   for parent, updates in by_parent.items()]
    except OSError as exc:
        return jsonify({"error": f"cannot write metadata.csv — {exc.strerror or exc}"}), 400
    return jsonify({"files": written, "n": len(folders), "fields": fields})


COHORT_TABLES = ("sections", "by_mouse_segment", "by_mouse", "by_genotype",
                 "comparison", "depth_relative")


@app.route("/api/cohort/plot", methods=["POST"])
def cohort_plot():
    """
    Draw the cohort tables, reading them back from the batch's own CSVs.

    Plotting from the files rather than from the batch's return value means the
    controls can be turned without re-stitching anything: the measurement is
    the expensive part and it is already done and on disk.
    """
    body = request.json or {}
    out_dir = Path(str(body.get("out_dir") or (HERE / "batch_out"))).expanduser()
    organ = str(body.get("organ") or "aorta")
    tables = {}
    for name in COHORT_TABLES:
        f = out_dir / f"cohort_{name}.csv"
        if f.exists() and f.stat().st_size:
            try:
                tables[name] = pd.read_csv(f)
            except Exception:
                pass
    if not tables:
        return jsonify({"error": f"no cohort_*.csv in {out_dir} — run a batch first"}), 400
    # Everything about how the figure looks -- and whether p-values are printed
    # on it -- arrives as the shared PlotSpec, so the panel of controls here is
    # the panel of controls in the other tools.
    spec = PlotSpec.from_dict(body.get("spec") or {})
    try:
        paths = CO.save_cohort_plot(
            tables, out_dir, name="cohort",
            readouts=body.get("readouts") or list(CO.scheme_for(organ).readouts),
            unit=str(body.get("unit") or "mouse_segment"),
            spec=spec,
            title=str(body.get("title") or ""),
            # The grid, sized a panel at a time.  This figure is the one that
            # grows -- one panel per readout per region -- so a fixed figure
            # width means a different panel size on every run, and panels that
            # have to sit beside each other on a page need to be set directly.
            organ=organ,
            panel_mm=float(body.get("panel_mm") or 0.0),
            panel_h_mm=float(body.get("panel_h_mm") or 0.0),
            ncol=int(body.get("ncol") or 0),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    # The curve through the wall is drawn from its own table and is not one of
    # the readouts, so it gets its own figure rather than a panel in the grid.
    # A batch without it -- an area organ, or the wall analysis switched off --
    # simply returns the one figure.
    if "depth_relative" in tables:
        try:
            paths.update({f"depth_{k}": v for k, v in CO.save_depth_plot(
                tables["depth_relative"], out_dir, spec=spec,
                title=str(body.get("title") or ""),
                counterstain=FB.counterstain_name(organ),
                stains_together=bool(body.get("stains_together")),
                panel_mm=float(body.get("panel_mm") or 0.0),
                panel_h_mm=float(body.get("panel_h_mm") or 0.0)).items()})
            # The source data for that figure is the file it was just drawn
            # from, not a second copy of the same numbers: every point on it is
            # a row of this table, mean and SD and n together.
            paths["depth_source_csv"] = str(out_dir / "cohort_depth_relative.csv")
        except ValueError:
            pass
    # The genotypes that were actually drawn, so the panel can offer a colour
    # box per genotype without guessing at names the figure does not carry.
    paths["groups"] = CO.genotypes_in(tables, str(body.get("unit") or "mouse_segment"))
    return jsonify(paths)


# =============================================================================
# Fibrosis by area -- heart, kidney, and anything else without a wall to follow
# =============================================================================


def _fib_params(d: dict) -> FB.FibrosisParams:
    fp = FB.FibrosisParams()
    for k, v in (d or {}).items():
        if k in asdict(fp) and v not in (None, ""):
            cur = getattr(fp, k)
            try:
                setattr(fp, k, type(cur)(v) if cur is not None else float(v))
            except (TypeError, ValueError):
                setattr(fp, k, v)
    return fp


def _fib_image(body: dict, progress) -> tuple[np.ndarray, float, np.ndarray | None, str]:
    """The image to measure, from the stitching session or from a file."""
    sid = body.get("sid")
    if sid:
        sess = _sess(str(sid))
        ts: SC.TileSet = sess["ts"]
        island = body.get("island", 0)
        idxs = None
        if island not in (None, "", "all"):
            idxs = [i for i, q in enumerate(sess["poses"]) if q.island == int(island)]
        scale = float(body.get("render_scale", 0.35))
        progress(0.05, "rendering the mosaic")
        # The coverage mask comes back with the mosaic because canvas no tile
        # ever painted is not blank slide.  Counting it as slide would inflate
        # the denominator with area the microscope never looked at, and every
        # number here is per unit tissue area.
        rgb, cov, _ = SC.render_mosaic(ts, sess["poses"], scale=scale, indices=idxs,
                                       return_coverage=True)
        return rgb, (ts.px_um or 1.0) / scale, cov, Path(sess["folder"]).name
    path = Path(str(body.get("image", ""))).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    arr, px = SC.read_rgb(path)
    return SC.to_float_rgb(arr), float(body.get("px_um") or px or 1.0), None, path.name


@app.route("/api/fibrosis", methods=["POST"])
def fibrosis_one():
    """
    How much of this tissue is collagen -- one section, heart or kidney.

    No wall is straightened here because there is none to follow: interstitial
    fibrosis is diffuse, and the question is simply how much of the tissue is
    collagen.  Everything is reported per unit *tissue* area, never per image
    area, because these sections are half empty slide and how much varies with
    where the field was taken.
    """
    body = request.json or {}
    organ = str(body.get("organ") or "heart")
    fp = _fib_params(body.get("params") or {})
    name = str(body.get("name") or organ)
    out_dir = Path(str(body.get("out_dir") or (HERE / "analysis_out"))).expanduser()
    # Hand-drawn polygons, in microns from the top-left of the image, so a
    # drawing made on the overview survives a change of render scale or of
    # analysis resolution between the run that drew it and the run that uses
    # it.  `exclude` comes out of the tissue; `regions` are measured on their
    # own as well as being part of the whole.
    exclude = body.get("exclude") or None
    regions = body.get("regions") or None
    # Marks, in the same microns as everything else drawn on the overview.
    points = body.get("points") or None
    lesion_k = float(body.get("lesion_k_mad") or 3.0)
    lesion_min = float(body.get("lesion_min_area_um2") or 2000.0)
    # A person's own verdict on a candidate lesion, kept apart from the two
    # knobs above: those change what the detector looks for everywhere in the
    # section, and a click here removes one specific patch someone has
    # actually looked at without moving anything else.
    reject_lesions = body.get("exclude_lesions_at") or None

    def work(progress):
        rgb, px_um, cov, title = _fib_image(body, progress)
        res = FB.analyze_fibrosis(rgb, px_um, fp, coverage=cov, organ=organ,
                                  exclude_um=exclude, regions_um=regions,
                                  points_um=points, lesion_k_mad=lesion_k,
                                  lesion_min_area_um2=lesion_min,
                                  exclude_lesions_at=reject_lesions,
                                  progress=lambda f, m: progress(0.1 + 0.75 * f, m))
        progress(0.9, "drawing")
        paths = FB.save_results(res, out_dir, name, title=title, organ=organ)
        FIGURE_CACHE[paths["figure"]] = {"fib": res, "title": title, "organ": organ}
        return {"organ": organ, "title": title,
                "counterstain": res.totals.get("counterstain", "muscle"),
                "totals": _jsonable(res.totals),
                "regions": _jsonable(res.regions),
                "lesions": _jsonable(res.lesions),
                "lesion_outlines": _jsonable(
                    FB.lesion_outlines(res.lesion_labels, res.px_um)
                    if res.lesion_labels is not None else []),
                "points": _jsonable(res.points),
                "paths": {k: str(v) for k, v in paths.items()},
                "figure_png": paths["figure"], "figure_svg": paths["figure_svg"],
                "overview": paths["overview"],
                "overview_px_um": float(paths["overview_px_um"]),
                "stains": _jsonable(np.asarray(res.stain_matrix).tolist())}

    return jsonify({"job": start_job(work, f"{organ} fibrosis")})


@app.route("/api/zones", methods=["POST"])
def zones_suggest():
    """
    A texture guide for the corticomedullary axis, and a first guess at zones.

    Two things with very different standing, returned together and labelled as
    such.  The *guide* asserts no boundary -- it is the texture gradient drawn
    as a picture, and it is monotonic across a whole kidney section (coherence
    0.41 at the papilla falling to 0.22 at the cortex).  The *polygons* are a
    clustering of that gradient, and they are poor: checked against the region
    lists in the folder names they got the number of zones right in 5 of 17
    sections, finding three zones in crops cut entirely from cortex.  They are
    a starting point for the editor, nothing more, and the note says so.
    """
    body = request.json or {}
    organ = str(body.get("organ") or "kidney")
    out_dir = Path(str(body.get("out_dir") or (HERE / "analysis_out"))).expanduser()
    name = str(body.get("name") or organ)

    def work(progress):
        import matplotlib.pyplot as plt
        import zones as ZN
        rgb, px_um, _cov, title = _fib_image(body, progress)
        progress(0.35, "measuring texture")
        guide, gpx, stats = ZN.texture_guide(rgb, px_um)
        out_dir.mkdir(parents=True, exist_ok=True)
        gpath = out_dir / f"{name}_texture_guide.png"
        plt.imsave(gpath, np.clip(guide, 0, 1))
        progress(0.7, "proposing zones")
        polys, note, k = [], "", 0
        try:
            res = ZN.detect_kidney_zones(rgb, px_um)
            polys = ZN.zone_polygons(res)
            k, note = res.n_zones, res.note
        except Exception as exc:
            note = f"no proposal: {type(exc).__name__}: {exc}"
        return {"guide": str(gpath), "guide_px_um": float(gpx),
                "polygons": _jsonable(polys), "n_zones": k,
                "note": note, "stats": _jsonable(stats), "title": title}

    return jsonify({"job": start_job(work, f"{organ} zones")})


@app.route("/api/fibrosis/batch", methods=["POST"])
def fibrosis_batch():
    """
    Stitch and measure a tray of heart or kidney sections without supervision.

    The same ladder as the aorta batch -- sections into mouse-and-region, mice
    into genotypes, the mouse as the unit -- with area integrals in place of
    the wall's length integrals.  Region matters more here than segment does
    for an aorta: cortex, medulla and papilla have visibly different baseline
    collagen, so pooling across them would compare tissue mixtures rather than
    genotypes.  It is kept as its own column for exactly that reason.
    """
    body = request.json or {}
    organ = str(body.get("organ") or "heart")
    folders = [Path(str(f)).expanduser() for f in (body.get("folders") or [])]
    parent = Path(str(body["parent"])).expanduser() if body.get("parent") else None
    if parent and not folders:
        folders = [Path(m.folder) for m in
                   CO.scan_tray(parent, list_images=SC.list_images, organ=organ)]
    out_root = Path(str(body.get("out_dir") or (HERE / "batch_out"))).expanduser()
    sp = _params_from(body.get("stitch_params", {}))
    fp = _fib_params(body.get("params") or {})
    render_scale = float(body.get("render_scale", 0.35))
    do_analysis = bool(body.get("analyze", True))
    # One stain matrix for the whole run, estimated from it, instead of the
    # fixed trichrome vectors.  Off by default: per-image estimation called
    # healthy myocardium 23 % fibrotic on this data, and a matrix estimated
    # across a run of nearly-pure muscle drifts the same way, only less.
    shared = bool(body.get("shared_stains", False))

    def work(progress):
        out_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        mosaics: list[tuple[Path, np.ndarray, float, np.ndarray, dict[str, Any]]] = []
        n = max(len(folders), 1)

        for k, folder in enumerate(folders):
            base, span = k / n, 1.0 / n
            meta = CO.parse_section_name(folder.name, folder=str(folder), organ=organ)
            over = dict(CO.read_overrides(parent).get(folder.name, {})) if parent else {}
            over.update(CO.read_overrides(folder.parent).get(folder.name, {}))
            for fld, value in over.items():
                setattr(meta, fld, value)
            row: dict[str, Any] = {k2: v for k2, v in meta.as_dict().items()
                                   if k2 != "missing"}
            try:
                ts, poses, matches, stats = SC.auto_stitch(
                    folder, sp,
                    progress=lambda f, m: progress(base + span * 0.6 * f, f"{folder.name}: {m}"))
                qc = SC.layout_qc(ts, poses)
                row.update({"n_tiles": len(ts), "n_islands": stats["n_islands"],
                            "largest_island": stats["island_sizes"][0],
                            "conflicts": int(qc.n_conflicts.sum()),
                            "median_agreement": round(float(qc.score.median()), 3)})
                # A file that would not decode is named rather than folded into
                # a smaller tile count: a section that came out thin and one
                # that was thin are different problems.
                if getattr(ts, "unreadable", None):
                    row["unreadable"] = "; ".join(ts.unreadable)
                d = out_root / folder.name
                d.mkdir(parents=True, exist_ok=True)
                SC.save_layout(d / "layout.json", ts, poses, matches, sp)
                idxs = [i for i, q in enumerate(poses) if q.island == 0]
                mo, cov, _ = SC.render_mosaic(ts, poses, scale=render_scale,
                                              indices=idxs, return_coverage=True)
                Image.fromarray((np.clip(mo, 0, 1) * 255).astype(np.uint8)).save(d / "mosaic.png")
                row["ok"] = True
                if do_analysis:
                    mosaics.append((d, mo, (ts.px_um or 1.0) / render_scale, cov, row))
                    progress(base + span, f"{folder.name}: stitched")
                    continue
            except Exception as exc:
                row["ok"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            progress(base + span, f"{folder.name}: done")

        # The stain matrix is chosen once, for the whole run, and written down
        # beside the numbers.  Two sections deconvolved with different vectors
        # are not comparable, and the only way to notice that afterwards is to
        # have recorded which were used.
        M = None
        if shared and mosaics:
            progress(0.9, "estimating stain vectors across the run")
            M = FB.shared_stain_matrix([m[1] for m in mosaics])
        for j, (d, mo, px_um, cov, row) in enumerate(mosaics):
            progress(0.9 + 0.09 * j / max(len(mosaics), 1), f"{d.name}: measuring")
            try:
                res = FB.analyze_fibrosis(mo, px_um, fp, coverage=cov,
                                          stain_matrix=M, organ=organ)
                FB.save_results(res, d, d.name, title=d.name, organ=organ)
                r = dict(row)
                r.update({k2: v for k2, v in res.totals.items()})
                r["ok"] = True
                rows.append(r)
            except Exception as exc:
                r = dict(row)
                r["ok"] = False
                r["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(r)

        df = pd.DataFrame(rows)
        csv = out_root / "batch_summary.csv"
        df.to_csv(csv, index=False)
        if M is not None:
            pd.DataFrame(M, columns=["R", "G", "B"],
                         index=["muscle", "collagen", "residual"]).to_csv(
                out_root / "stain_vectors.csv")

        cohort: dict[str, Any] = {}
        paths: dict[str, str] = {}
        measured = df[df.get("ok", False).astype(bool)] if "ok" in df else df
        if do_analysis and len(measured) and "collagen_od_um2_total" in measured.columns:
            try:
                tables = CO.summarise(measured, organ=organ)
                paths = CO.write_summary(tables, out_root)
                cohort = {k: _jsonable(v.to_dict(orient="records"))
                          for k, v in tables.items() if k != "sections"}
            except Exception as exc:   # a cohort is a bonus; the batch still stands
                cohort = {"error": f"{type(exc).__name__}: {exc}"}
        return {"organ": organ, "rows": _jsonable(rows), "csv": str(csv),
                "cohort": cohort, "cohort_csv": paths,
                "readouts": list(CO.scheme_for(organ).readouts)}

    return jsonify({"job": start_job(work, f"{organ} batch")})


@app.route("/api/batch", methods=["POST"])
def batch():
    """Stitch and (optionally) analyse a list of folders without supervision."""
    body = request.json or {}
    folders = [Path(str(f)).expanduser() for f in (body.get("folders") or [])]
    # "Point me at the tray": a parent folder stands in for its sub-folders.
    parent = Path(str(body["parent"])).expanduser() if body.get("parent") else None
    if parent and not folders:
        folders = [Path(m.folder) for m in CO.scan_tray(parent, list_images=SC.list_images)]
    out_root = Path(str(body.get("out_dir") or (HERE / "batch_out"))).expanduser()
    do_analysis = bool(body.get("analyze", True))
    sp = _params_from(body.get("stitch_params", {}))
    wp = WA.WallParams()
    for k, v in (body.get("wall_params") or {}).items():
        if k in asdict(wp) and v not in (None, ""):
            cur = getattr(wp, k)
            try:
                setattr(wp, k, type(cur)(v) if cur is not None else float(v))
            except (TypeError, ValueError):
                setattr(wp, k, v)
    render_scale = float(body.get("render_scale", 0.35))

    def work(progress):
        rows = []
        # Panel E, section by section.  save_results already writes each one
        # next to its own mosaic, but the question that panel exists to answer
        # -- does the profile through the media differ by genotype -- is asked
        # across sections, and answering it from fifty separate files is
        # nobody's idea of an analysis.  Collected here in long form, one row
        # per depth per vessel, carrying the same identity columns as the
        # summary so it groups the same way.
        depth_frames: list[pd.DataFrame] = []
        out_root.mkdir(parents=True, exist_ok=True)
        for k, folder in enumerate(folders):
            base = k / max(len(folders), 1)
            span = 1.0 / max(len(folders), 1)
            meta = CO.parse_section_name(folder.name, folder=str(folder))
            # Sections can now be ticked across several trays, so overrides come
            # from each folder's own parent; the browsed tray is only a fallback
            # for a folder listed by hand from somewhere else.
            over = dict(CO.read_overrides(parent).get(folder.name, {})) if parent else {}
            over.update(CO.read_overrides(folder.parent).get(folder.name, {}))
            for field, value in over.items():
                setattr(meta, field, value)
            row: dict[str, Any] = {k2: v for k2, v in meta.as_dict().items()
                                   if k2 != "missing"}
            try:
                ts, poses, matches, stats = SC.auto_stitch(
                    folder, sp,
                    progress=lambda f, m: progress(base + span * 0.6 * f, f"{folder.name}: {m}"))
                qc = SC.layout_qc(ts, poses)
                row.update({"n_tiles": len(ts), "n_islands": stats["n_islands"],
                            "largest_island": stats["island_sizes"][0],
                            "conflicts": int(qc.n_conflicts.sum()),
                            "median_agreement": round(float(qc.score.median()), 3)})
                # A file that would not decode is named rather than folded into
                # a smaller tile count: a section that came out thin and one
                # that was thin are different problems.
                if getattr(ts, "unreadable", None):
                    row["unreadable"] = "; ".join(ts.unreadable)
                d = out_root / folder.name
                d.mkdir(parents=True, exist_ok=True)
                SC.save_layout(d / "layout.json", ts, poses, matches, sp)
                idxs = [i for i, p in enumerate(poses) if p.island == 0]
                mo = SC.render_mosaic(ts, poses, scale=render_scale, indices=idxs)
                Image.fromarray((np.clip(mo, 0, 1) * 255).astype(np.uint8)).save(d / "mosaic.png")
                if do_analysis:
                    progress(base + span * 0.75, f"{folder.name}: wall analysis")
                    # Every vessel in the frame, not just the biggest.  An arch
                    # section routinely catches the vessel twice; measuring one
                    # and dropping the other throws away half the wall, and the
                    # cohort step pools them back together by segment anyway.
                    results = WA.analyze_vessels(mo, (ts.px_um or 1.0) / render_scale, wp)
                    ov = np.clip(SC.resize(mo, 0.3), 0, 1)
                    for res in results:
                        rank = int(res.totals.get("lumen_rank", 0))
                        stem = folder.name if rank == 0 else f"{folder.name}_vessel{rank + 1}"
                        saved = WA.save_results(res, d, stem, overview_rgb=ov)
                        r = dict(row)
                        r.update({k2: v for k2, v in res.totals.items()})
                        # Where this vessel's per-position table went, so the
                        # cohort step can rebuild the media-quality medians
                        # from positions instead of from section medians.
                        r[CO.PROFILE_COLUMN] = saved.get("profile", "")
                        r["vessel"] = rank + 1
                        r["n_vessels"] = len(results)
                        r["ok"] = True
                        rows.append(r)

                        ident = {k2: v for k2, v in meta.as_dict().items()
                                 if k2 != "missing"}
                        dr = res.depth_profile_rel.copy()
                        dr.insert(0, "vessel", rank + 1)
                        for pos, (k2, v) in enumerate(ident.items()):
                            dr.insert(pos, k2, v)
                        depth_frames.append(dr)
                    if results:
                        progress(base + span, f"{folder.name}: done")
                        continue
                row["ok"] = True
            except Exception as exc:
                row["ok"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            progress(base + span, f"{folder.name}: done")
        df = pd.DataFrame(rows)
        csv = out_root / "batch_summary.csv"
        df.to_csv(csv, index=False)

        depth_csv = depth_avg_csv = ""
        if depth_frames:
            depth_long = pd.concat(depth_frames, ignore_index=True)
            depth_csv = str(out_root / "batch_depth_relative.csv")
            depth_long.to_csv(depth_csv, index=False)
            try:
                avg = CO.average_depth_profile(depth_long)
                if len(avg):
                    depth_avg_csv = str(out_root / "cohort_depth_relative.csv")
                    avg.to_csv(depth_avg_csv, index=False)
                    CO.save_depth_plot(avg, out_root,
                                       title=parent.name if parent else "")
            except Exception as exc:   # an average is a bonus; the batch still stands
                depth_avg_csv = f"error: {type(exc).__name__}: {exc}"

        # Sections -> segments within a mouse -> mice -> genotypes.  Only the
        # sections that actually measured take part; a failed stitch must not
        # quietly become a smaller denominator.
        cohort: dict[str, Any] = {}
        paths: dict[str, str] = {}
        measured = df[df.get("ok", False).astype(bool)] if "ok" in df else df
        if do_analysis and len(measured) and "collagen_od_um2_total" in measured.columns:
            try:
                # The profiles were written a moment ago and their paths are
                # in the rows, so the exact rebuild is available here and
                # costs one pass over files already on disk.  Any group it
                # cannot read falls back, and says so in `medians_from`.
                tables = CO.summarise(measured, medians="exact")
                paths = CO.write_summary(tables, out_root)
                cohort = {k: _jsonable(v.to_dict(orient="records"))
                          for k, v in tables.items() if k != "sections"}
            except Exception as exc:   # a cohort is a bonus; the batch still stands
                cohort = {"error": f"{type(exc).__name__}: {exc}"}
        return {"rows": _jsonable(rows), "csv": str(csv),
                "depth_relative_csv": depth_csv,
                "depth_relative_avg_csv": depth_avg_csv,
                "cohort": cohort, "cohort_csv": paths}

    return jsonify({"job": start_job(work, "batch")})


# =============================================================================


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("FIBROSIS_PORT") or 8765)
    print("\n  Aortic Straightener and Fibrosis Quantifier")
    print(f"  open  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
