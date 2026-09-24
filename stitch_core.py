"""
stitch_core.py -- a generic tile-mosaic assembler for microscopy fields.

Nothing in this module knows about aortas.  It takes a folder of image tiles
that overlap each other in an unknown arrangement and works out where each one
belongs, so the same engine serves vessel walls, kidney sections, brain slices,
gut, or any other specimen photographed as a set of overlapping snapshots.

Why not plain correlation
-------------------------
A plain masked correlation of percentile-stretched RGB is the obvious score for
a candidate placement, and it fails here.  On trichrome histology most of a
tile is empty white slide, so background-against-background scores ~0.99 and
beats the true placement, even on pairs with zero geometric inliers.  Searching
translations on a coarse integer grid (step 16 at 0.1x, i.e. 160
full-resolution pixels) is both slow and blind to the correct peak.

This module instead uses

*  an **optical-density** registration signal that is exactly zero on blank
   slide, high-pass filtered so uneven illumination between snapshots cannot
   drive the match;
*  **masked normalised cross-correlation evaluated by FFT** (Padfield 2012), so
   every integer translation is scored at once, in O(n log n), with the
   normalisation computed only over the pixels where *both* tiles carry tissue.
   Background contributes nothing, because it is outside the mask;
*  a **confidence model** -- peak height, peak-to-second-peak ratio, overlapping
   tissue area -- so the assembler can tell "this is certainly the right
   placement" from "this pair is ambiguous, ask the human".

Public entry points
-------------------
``TileSet``                load a folder, build pyramids and registration signals
``match_all_pairs``        score every tile pair (coarse) and refine the good ones
``solve_layout``           robust global placement, split into islands
``snap_tile`` / ``snap_all``  local re-alignment after a manual drag
``render_mosaic``          feathered mosaic at any scale
``save_layout`` / ``load_layout``   layout persistence (JSON + CSV)
"""

from __future__ import annotations

import json
import math
import os
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import ndimage as ndi

try:  # optional, only used for fast resizing / warping
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from czifile import CziFile, imread as _czi_imread
except ImportError:  # pragma: no cover
    CziFile = None
    _czi_imread = None

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

from PIL import Image

# A tissue mask is a tissue mask whether it is being registered or measured,
# and wall_analysis already has the OpenCV morphology that matches scipy
# pixel for pixel (see test_fast_paths.py).  It imports nothing from here.
from wall_analysis import gaussian, square_morph

Image.MAX_IMAGE_PIXELS = None


IMAGE_EXTS = (".czi", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Anything below this many overlapping tissue pixels is not evidence of a match,
# it is a statistical accident.  Expressed at the coarse working scale.
MIN_OVERLAP_PX_COARSE = 400

# How much a TileSet may hold in cached registration levels.  A session is
# kept for as long as the server runs, so this is a permanent cost per folder
# and it is set at the knee rather than at the optimum: on a 60-tile section
# refinement takes 7.8 s holding 128 MB against 6.8 s holding all 272 MB, and
# 14.8 s holding none.
LEVEL_CACHE_BYTES = 128 << 20


# =============================================================================
# 1.  Image loading
# =============================================================================


def list_images(folder: str | Path, exts: Sequence[str] = IMAGE_EXTS) -> list[Path]:
    """Every readable image in *folder*, sorted naturally by name."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in folder.iterdir():
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return sorted(out, key=_natural_key)


def _natural_key(p: Path) -> tuple:
    import re

    parts = re.split(r"(\d+)", p.name)
    return tuple(int(s) if s.isdigit() else s.lower() for s in parts)


def read_rgb(path: str | Path) -> tuple[np.ndarray, float | None]:
    """
    Read one tile as an ``(H, W, 3)`` array plus its pixel size in microns.

    The array keeps its native dtype; ``to_float_rgb`` normalises it later.
    Returns ``pixel_size_um = None`` when the format carries no scale.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".czi":
        return _read_czi(path)
    if suffix in (".tif", ".tiff"):
        return _read_tiff(path)

    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr, None


def _read_czi(path: Path) -> tuple[np.ndarray, float | None]:
    if _czi_imread is None:
        raise ImportError("Reading .czi needs the `czifile` package (pip install czifile).")
    arr = np.squeeze(_czi_imread(str(path)))
    arr = _as_rgb(arr)

    px_um = None
    try:
        with CziFile(str(path)) as czi:
            root = ET.fromstring(czi.metadata())
        node = root.find('.//Scaling//Distance[@Id="X"]/Value')
        if node is not None and node.text:
            px_um = float(node.text) * 1e6
    except Exception:
        px_um = None
    return arr, px_um


def _read_tiff(path: Path) -> tuple[np.ndarray, float | None]:
    if tifffile is None:
        arr = np.asarray(Image.open(path).convert("RGB"))
        return arr, None
    with tifffile.TiffFile(str(path)) as tf:
        arr = np.squeeze(tf.asarray())
        px_um = None
        try:
            page = tf.pages[0]
            tags = page.tags
            unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
            xres = tags["XResolution"].value if "XResolution" in tags else None
            if xres:
                num, den = (xres if isinstance(xres, tuple) else (xres, 1))
                if num:
                    per_unit = float(den) / float(num)
                    # 2 = inch, 3 = centimetre
                    if unit == 2:
                        px_um = per_unit * 25400.0
                    elif unit == 3:
                        px_um = per_unit * 10000.0
        except Exception:
            px_um = None
    return _as_rgb(arr), px_um


def _as_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce whatever came out of the reader into ``(H, W, 3)``."""
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            return arr
        if arr.shape[-1] == 4:
            return arr[..., :3]
        if arr.shape[0] == 3:
            return np.moveaxis(arr, 0, -1)
        if arr.shape[0] == 4:
            return np.moveaxis(arr[:3], 0, -1)
        # more than 4 planes: treat the first three as RGB
        if arr.shape[0] < arr.shape[-1]:
            return np.moveaxis(arr[:3], 0, -1)
        return arr[..., :3]
    raise ValueError(f"Cannot interpret an array of shape {arr.shape} as one RGB tile.")


def to_float_rgb(arr: np.ndarray) -> np.ndarray:
    """Scale to ``[0, 1]`` floats using the dtype's full range where sensible."""
    img = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        top = float(info.max)
        # 12-bit and 14-bit cameras store into uint16; using 65535 would make
        # everything black, so fall back to the observed maximum when the data
        # occupies a small part of the container.
        obs = float(img.max()) if img.size else 1.0
        if top > 0 and obs < top * 0.35:
            top = max(obs, 1.0)
        return np.clip(img / max(top, 1.0), 0.0, 1.0)
    top = float(np.nanmax(img)) if img.size else 1.0
    if top <= 1.5:
        return np.clip(img, 0.0, 1.0)
    return np.clip(img / max(top, 1e-6), 0.0, 1.0)


def resize(img: np.ndarray, scale: float, interp: str = "area") -> np.ndarray:
    """Scale an image by *scale* (<=1 shrinks).  Uses OpenCV when available."""
    if scale == 1.0:
        return img.copy()
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if cv2 is not None:
        flag = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
        }[interp]
        if img.dtype == bool:
            out = cv2.resize(img.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
            return out > 0
        return cv2.resize(img, (nw, nh), interpolation=flag)
    zoom = [nh / h, nw / w] + ([1] * (img.ndim - 2))
    order = 0 if interp == "nearest" or img.dtype == bool else 1
    return ndi.zoom(img.astype(np.float32), zoom, order=order).astype(img.dtype)


# =============================================================================
# 2.  Registration signal: optical density, tissue mask, high-pass
# =============================================================================


def optical_density(rgb01: np.ndarray, white: np.ndarray | None = None) -> np.ndarray:
    """
    Beer-Lambert optical density, ``OD = -log10(I / I_white)``.

    On blank slide ``I == I_white`` so ``OD == 0``.  That is the whole point:
    the registration signal vanishes where there is nothing to register.
    """
    if white is None:
        white = estimate_white(rgb01)
    white = np.asarray(white, dtype=np.float32).reshape(1, 1, -1)
    ratio = np.clip(rgb01.astype(np.float32) / np.maximum(white, 1e-4), 1e-4, 1.0)
    return -np.log10(ratio)


def estimate_white(rgb01: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """
    Per-channel white point of a tile, taken as a high percentile.

    Using a percentile rather than the maximum keeps hot pixels and specular
    dust from setting the reference.
    """
    flat = rgb01.reshape(-1, rgb01.shape[-1])
    if flat.shape[0] > 400_000:
        step = flat.shape[0] // 400_000 + 1
        flat = flat[::step]
    w = np.percentile(flat, pct, axis=0).astype(np.float32)
    return np.maximum(w, 1e-3)


def auto_od_threshold(od_sum: np.ndarray, lo: float = 0.12, hi: float = 0.80) -> float:
    """
    Pick the background/tissue cut from the tile's own optical-density histogram.

    Brightfield histology gives a tall narrow background spike near OD 0 and a
    long tail of tissue; that is the exact shape the *triangle* threshold was
    designed for, and unlike Otsu it does not need the two classes to be
    balanced -- which matters here, because a tile can be 5 % tissue or 95 %.

    A fixed threshold cannot work: measured background sits at OD ~0.10-0.13 on
    this scanner because the slide is faintly tinted and the field vignettes, so
    anything below that swallows the whole frame.  The clamp keeps a pathological
    histogram (all background, or all tissue) from producing a silly answer.
    """
    try:
        from skimage.filters import threshold_triangle
        thr = float(threshold_triangle(od_sum))
    except Exception:
        thr = float(np.percentile(od_sum, 60.0))
    if not np.isfinite(thr):
        thr = lo
    return float(np.clip(thr, lo, hi))


def tissue_mask_from_od(
    od_sum: np.ndarray,
    od_threshold: float | None = None,
    min_area_frac: float = 0.0008,
    close_px: int = 2,
) -> np.ndarray:
    """
    Where is there actually something on the slide?

    ``od_threshold`` is in optical-density units summed over RGB.  Leave it at
    ``None`` to let :func:`auto_od_threshold` read it off the tile.
    """
    if od_threshold is None:
        od_threshold = auto_od_threshold(od_sum)
    mask = od_sum > float(od_threshold)
    if close_px > 0:
        mask = square_morph(mask, close_px * 2 + 1, "close")
    mask = square_morph(mask, 3, "open")
    if min_area_frac > 0:
        min_px = max(16, int(min_area_frac * mask.size))
        lab, n = ndi.label(mask)
        if n:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= min_px
            keep[0] = False
            mask = keep[lab]
    return mask.astype(bool)


def high_pass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Remove smooth illumination differences, keep texture."""
    if sigma <= 0:
        return img.astype(np.float32)
    a = img.astype(np.float32)
    return a - gaussian(a, sigma)


@dataclass
class RegLevel:
    """One pyramid level of one tile, ready for masked correlation."""

    scale: float
    signal: np.ndarray  # float32, zero outside tissue
    mask: np.ndarray  # bool
    shape: tuple[int, int]
    od_sum: np.ndarray | None = None  # kept so the mask can be redone later
    threshold: float = 0.0

    @property
    def tissue_px(self) -> int:
        return int(self.mask.sum())


def build_reg_level(
    rgb01: np.ndarray,
    scale: float,
    od_threshold: float | None = None,
    highpass_sigma: float = 6.0,
    white: np.ndarray | None = None,
) -> RegLevel:
    """
    Turn a tile into (signal, mask) at *scale*.

    The signal is the high-pass filtered total optical density, zeroed outside
    the tissue mask.  The mask is what masked NCC normalises over.
    """
    small = resize(rgb01, scale, "area") if scale != 1.0 else rgb01
    od = optical_density(small, white=white if white is not None else estimate_white(rgb01, 99.5))
    od_sum = od.sum(axis=2).astype(np.float32)
    thr = auto_od_threshold(od_sum) if od_threshold is None else float(od_threshold)
    lvl = RegLevel(scale=float(scale), signal=np.empty(0, np.float32),
                   mask=np.empty(0, bool), shape=od_sum.shape[:2],
                   od_sum=od_sum, threshold=thr)
    return remask_level(lvl, thr, highpass_sigma)


def remask_level(level: RegLevel, threshold: float, highpass_sigma: float = 6.0) -> RegLevel:
    """Rebuild a level's mask and signal at a different optical-density cut."""
    od_sum = level.od_sum
    if od_sum is None:
        return level
    mask = tissue_mask_from_od(od_sum, od_threshold=threshold)
    sig = high_pass(od_sum, highpass_sigma)
    sig = np.where(mask, sig, 0.0).astype(np.float32)
    # A slight dilation of the mask keeps the tissue edge -- a very informative
    # feature -- inside the normalisation window.
    mask = ndi.binary_dilation(mask, structure=np.ones((3, 3)), iterations=2)
    return RegLevel(scale=level.scale, signal=sig, mask=mask, shape=sig.shape[:2],
                    od_sum=od_sum, threshold=float(threshold))


# =============================================================================
# 3.  Masked normalised cross-correlation by FFT  (Padfield 2012)
# =============================================================================


def _fast_shape(h: int, w: int) -> tuple[int, int]:
    return (sfft.next_fast_len(h), sfft.next_fast_len(w))


def _pack_fft(level: RegLevel, fshape: tuple[int, int], workers: int = -1) -> dict[str, np.ndarray]:
    """The three transforms a tile needs, whichever side of a pair it is on."""
    f = level.signal
    m = level.mask.astype(np.float32)
    return {
        "F1": sfft.rfft2(f, fshape, workers=workers).astype(np.complex64),
        "F2": sfft.rfft2(f * f, fshape, workers=workers).astype(np.complex64),
        "F3": sfft.rfft2(m, fshape, workers=workers).astype(np.complex64),
    }


def masked_ncc_map(
    pack_a: dict[str, np.ndarray],
    pack_b: dict[str, np.ndarray],
    fshape: tuple[int, int],
    min_overlap: int = MIN_OVERLAP_PX_COARSE,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Masked NCC for every integer shift, plus the overlap count at each shift.

    Index convention: entry ``[ky, kx]`` is the score for placing tile B at
    translation ``(tx, ty)`` relative to tile A, where ``t = k`` for
    ``k < fshape/2`` and ``t = k - fshape`` otherwise (standard FFT wrap-around).
    ``shift_from_index`` converts.

    Derivation follows D. Padfield, "Masked object registration in the Fourier
    domain", IEEE TIP 21(5), 2012: every first and second moment needed for the
    Pearson correlation over the *overlapping masked region* is itself a
    correlation, so all of them come from six inverse transforms.
    """
    ir = lambda X: sfft.irfft2(X, fshape, workers=workers)

    A1, A2, A3 = pack_a["F1"], pack_a["F2"], pack_a["F3"]
    B1, B2, B3 = pack_b["F1"], pack_b["F2"], pack_b["F3"]

    n = ir(A3 * np.conj(B3))  # overlapping mask area
    fg = ir(A1 * np.conj(B1))  # sum f*g
    f_1 = ir(A1 * np.conj(B3))  # sum f  over overlap
    g_1 = ir(A3 * np.conj(B1))  # sum g  over overlap
    f2 = ir(A2 * np.conj(B3))  # sum f^2
    g2 = ir(A3 * np.conj(B2))  # sum g^2

    n = np.maximum(n, 0.0)
    valid = n >= float(min_overlap)
    safe_n = np.where(valid, n, 1.0)

    num = fg - f_1 * g_1 / safe_n
    den_f = f2 - f_1 * f_1 / safe_n
    den_g = g2 - g_1 * g_1 / safe_n
    den = np.sqrt(np.maximum(den_f, 0.0) * np.maximum(den_g, 0.0))

    ncc = np.where(valid & (den > 1e-8), num / np.maximum(den, 1e-8), 0.0)
    # Numerical slop can push a perfectly-overlapping shift a hair past 1.
    ncc = np.clip(ncc, -1.0, 1.0)
    return ncc.astype(np.float32), n.astype(np.float32)


def shift_from_index(ky: int, kx: int, fshape: tuple[int, int]) -> tuple[int, int]:
    """FFT bin -> signed translation ``(tx, ty)`` of B relative to A."""
    fy, fx = fshape
    ty = ky if ky < fy // 2 else ky - fy
    tx = kx if kx < fx // 2 else kx - fx
    return int(tx), int(ty)


def index_from_shift(tx: int, ty: int, fshape: tuple[int, int]) -> tuple[int, int]:
    fy, fx = fshape
    return int(ty % fy), int(tx % fx)


def _valid_shift_window(
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    fshape: tuple[int, int],
    max_shift_frac: float,
) -> np.ndarray:
    """
    Boolean map of shifts we are willing to consider.

    Shifts beyond ``max_shift_frac`` of the tile size mean the tiles barely
    touch; those are exactly where masked NCC gets noisy, so we exclude them
    before searching for a peak instead of filtering afterwards.
    """
    ha, wa = shape_a
    hb, wb = shape_b
    fy, fx = fshape
    ty = np.where(np.arange(fy) < fy // 2, np.arange(fy), np.arange(fy) - fy)
    tx = np.where(np.arange(fx) < fx // 2, np.arange(fx), np.arange(fx) - fx)
    lim_y = max_shift_frac * max(ha, hb)
    lim_x = max_shift_frac * max(wa, wb)
    ok_y = np.abs(ty) <= lim_y
    ok_x = np.abs(tx) <= lim_x
    return ok_y[:, None] & ok_x[None, :]


def _crest(weighted: np.ndarray, ky: int, kx: int, frac: float) -> np.ndarray:
    """The winner's own high ground: everything connected to it above *frac*.

    Rolled so the winner sits in the middle, because the correlation map wraps
    and a ridge that runs off one edge continues on the other.
    """
    fy, fx = weighted.shape
    w = np.roll(np.roll(weighted, fy // 2 - ky, 0), fx // 2 - kx, 1)
    lab, _ = ndi.label(w >= frac * w[fy // 2, fx // 2])
    comp = lab == lab[fy // 2, fx // 2]
    return np.roll(np.roll(comp, ky - fy // 2, 0), kx - fx // 2, 1)


def peak_with_confidence(
    ncc: np.ndarray,
    counts: np.ndarray,
    fshape: tuple[int, int],
    allowed: np.ndarray,
    exclusion_px: int = 12,
    saturation_overlap: float = 0.0,
    ridge_frac: float = 0.6,
    ridge_below: float = 1.6,
) -> dict[str, Any]:
    """
    Best shift, and how much we should believe it.

    Two guards matter here.

    *Small-overlap saturation.*  Masked NCC is a correlation over however many
    pixels the two masks share, so a shift that makes forty pixels overlap can
    reach exactly 1.0 by accident.  Scores are therefore weighted by
    ``sqrt(n / saturation_overlap)`` while the peak is being *found*; the
    unweighted NCC is what gets reported.  Without this the search happily
    walks off to a corner where two specks touch.

    *Ridges.*  ``peak_ratio`` compares the winner with the best value well away
    from it.  A pair overlapping along a long featureless structure -- a vessel
    wall photographed end to end -- produces a correlation *ridge*: slide one
    tile along the wall and the score barely drops, so the best value a long way
    off is nearly the winner's and a box-shaped exclusion puts the ratio at 1.

    But a ridge and a rival are not the same doubt.  A rival is a *different*
    placement that fits about as well, and one of the two is wrong.  A ridge is
    one placement whose position along the wall is loose: every point on it puts
    the same tissue on the same tissue.  Refusing both throws away every join
    across a thinly-sampled stretch of wall -- which is exactly where the join
    is needed, because that is where the tiles stop overlapping properly.

    So the runner-up is taken from outside the winner's own crest, not merely
    outside a small box.  Walking the crest costs an extra labelling of the
    map, so it is only done when the box already says the match is
    unconvincing.
    """
    if saturation_overlap and saturation_overlap > 0:
        w = np.sqrt(np.clip(counts / float(saturation_overlap), 0.0, 1.0))
    else:
        w = 1.0
    weighted = ncc * w

    scores = np.where(allowed, weighted, -2.0)
    k = int(np.argmax(scores))
    ky, kx = np.unravel_index(k, scores.shape)
    if float(scores[ky, kx]) <= -1.5:
        return {"ok": False}

    # Second peak, outside a small exclusion box around the winner.
    fy, fx = fshape
    yy = (np.arange(fy)[:, None] - ky + fy // 2) % fy - fy // 2
    xx = (np.arange(fx)[None, :] - kx + fx // 2) % fx - fx // 2
    far = (np.abs(yy) > exclusion_px) | (np.abs(xx) > exclusion_px)
    best_w = float(weighted[ky, kx])
    second = float(np.max(np.where(allowed & far, weighted, -2.0)))
    second = max(second, 1e-3)
    ratio = best_w / second

    if ratio < ridge_below:
        crest = _crest(np.where(allowed, weighted, 0.0), int(ky), int(kx), ridge_frac)
        second = max(float(np.max(np.where(allowed & far & ~crest, weighted, -2.0))), 1e-3)
        ratio = best_w / second

    tx, ty = shift_from_index(int(ky), int(kx), fshape)
    sub_dx, sub_dy = _subpixel_offset(ncc, int(ky), int(kx))
    return {
        "ok": True,
        "tx": float(tx + sub_dx),
        "ty": float(ty + sub_dy),
        "tx_int": int(tx),
        "ty_int": int(ty),
        "ncc": float(ncc[ky, kx]),
        "second": second,
        "peak_ratio": float(ratio),
        "n_overlap": float(counts[ky, kx]),
    }


def _subpixel_offset(ncc: np.ndarray, ky: int, kx: int) -> tuple[float, float]:
    """Parabolic fit through the three samples around the peak, per axis."""
    fy, fx = ncc.shape

    def par(m: float, c: float, p: float) -> float:
        denom = (m - 2 * c + p)
        if abs(denom) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (m - p) / denom, -0.75, 0.75))

    dy = par(ncc[(ky - 1) % fy, kx], ncc[ky, kx], ncc[(ky + 1) % fy, kx])
    dx = par(ncc[ky, (kx - 1) % fx], ncc[ky, kx], ncc[ky, (kx + 1) % fx])
    return dx, dy


# =============================================================================
# 4.  Tiles and tile sets
# =============================================================================


@dataclass
class Tile:
    index: int
    path: Path
    name: str
    height: int
    width: int
    px_um: float | None = None
    white: np.ndarray | None = None
    tissue_frac: float = 0.0
    thumb: np.ndarray | None = None  # uint8 RGB, display resolution
    coarse: RegLevel | None = None
    _fft: dict[str, np.ndarray] | None = None
    _thumb_raw: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "path": str(self.path),
            "name": self.name,
            "height": self.height,
            "width": self.width,
            "px_um": self.px_um,
            "tissue_frac": self.tissue_frac,
        }


@dataclass
class StitchParams:
    """Everything the assembler can be told to do differently."""

    coarse_max_dim: int = 320  # working size for the all-pairs sweep
    refine_max_dim: int = 1100  # working size for the verification pass
    display_max_dim: int = 700  # thumbnail size sent to the browser
    od_threshold: float | None = None  # None = per-tile automatic (triangle)
    highpass_sigma: float = 6.0
    max_shift_frac: float = 0.92  # reject placements that barely touch
    min_overlap_frac: float = 0.06  # of the smaller tile's tissue area
    min_ncc: float = 0.30
    min_peak_ratio: float = 1.12
    rotation_search_deg: float = 0.0  # 0 disables; e.g. 4 searches +/-4 deg
    rotation_step_deg: float = 1.0
    neighbour_window: int = 0  # 0 = compare all pairs; k = only |i-j| <= k
    chain_window: int = 3  # lay tiles adjacent in acquisition order first; 0 = off
    refine_top_k: int = 6  # per tile, how many coarse candidates to verify
    flatfield: bool = True
    white_balance: bool = True
    max_workers: int = max(1, (os.cpu_count() or 4) - 1)

    def copy_with(self, **kw) -> "StitchParams":
        d = asdict(self)
        d.update({k: v for k, v in kw.items() if k in d and v is not None})
        return StitchParams(**d)


class TileSet:
    """
    A folder of tiles, loaded once and kept ready for matching and rendering.

    Full-resolution pixels are *not* held in memory; tiles are re-read on demand
    when a mosaic is rendered.  What is cached is the display thumbnail, the
    coarse registration level and its FFTs, which is a few megabytes per tile.
    """

    def __init__(self, params: StitchParams | None = None):
        self.params = params or StitchParams()
        self.tiles: list[Tile] = []
        self.unreadable: list[str] = []   # files that would not decode, by name
        self.fshape: tuple[int, int] = (0, 0)
        self.folder: Path | None = None
        self.od_threshold: float | None = None
        self._flat: np.ndarray | None = None
        self._white_ref: np.ndarray | None = None
        self._levels: "OrderedDict[tuple, RegLevel]" = OrderedDict()
        self._levels_pending: "dict[tuple, Future]" = {}
        self._levels_bytes = 0
        self._levels_lock = threading.Lock()

    # -- loading ----------------------------------------------------------

    def load_folder(
        self,
        folder: str | Path,
        files: Sequence[str] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> "TileSet":
        folder = Path(folder).expanduser()
        self.folder = folder
        paths = [folder / f for f in files] if files else list_images(folder)
        if not paths:
            raise FileNotFoundError(f"No readable images in {folder}")
        return self.load_paths(paths, progress=progress)

    def load_paths(
        self,
        paths: Sequence[str | Path],
        progress: Callable[[float, str], None] | None = None,
    ) -> "TileSet":
        paths = [Path(p) for p in paths]
        n = len(paths)
        p = self.params
        self.tiles = []

        # Pass 1: read each tile, keep a thumbnail and the coarse level.
        whites = []
        bg_accum: np.ndarray | None = None
        bg_n = 0
        # A file that will not decode costs you that file, not the section.  One
        # corrupt snapshot in sixty is a real thing -- a CZI written with no
        # scene in it raises from inside the reader -- and losing the other
        # fifty-nine to it is a bad trade.  The names are kept so a section
        # that came out thin can be told from one that was.
        self.unreadable = []

        def read_one(path: Path):
            """Everything one tile needs, with no full-resolution pixels kept."""
            arr, px_um = read_rgb(path)
            rgb01 = to_float_rgb(arr)
            h, w = rgb01.shape[:2]
            white = estimate_white(rgb01)
            cs = min(1.0, p.coarse_max_dim / max(h, w))
            ds = min(1.0, p.display_max_dim / max(h, w))
            coarse = build_reg_level(
                rgb01, cs, od_threshold=p.od_threshold,
                highpass_sigma=p.highpass_sigma, white=white,
            )
            thumb_small = resize(rgb01, ds, "area")
            # Contribution to the flat-field estimate: the tile divided by its
            # own white point, averaged over all tiles, is dominated by the
            # illumination profile because tissue moves between snapshots.
            small = (resize(rgb01 / white.reshape(1, 1, 3), 128 / max(h, w), "area")
                     if p.flatfield else None)
            return path, px_um, h, w, white, coarse, thumb_small, small

        def read_guarded(path: Path):
            try:
                return read_one(path)
            except Exception as exc:
                return f"{path.name}: {type(exc).__name__}"

        # Reading is four fifths of the load and every tile is independent, so
        # it goes wide.  The pool is capped well below the core count because
        # each worker holds one full-resolution tile while it works -- sixty
        # megabytes apiece on this scanner -- and the disk stops being the
        # limit long before fifteen of those are worth holding at once.
        with ThreadPoolExecutor(max_workers=max(1, min(p.max_workers, 8))) as ex:
            for i, got in enumerate(ex.map(read_guarded, paths)):
                if progress:
                    progress(0.05 + 0.60 * i / max(n, 1), f"reading {paths[i].name}")
                if isinstance(got, str):
                    self.unreadable.append(got)
                    continue
                path, px_um, h, w, white, coarse, thumb_small, small = got
                whites.append(white)
                if small is not None:
                    bg_accum = small if bg_accum is None else bg_accum + small
                    bg_n += 1
                tile = Tile(
                    index=len(self.tiles), path=path, name=path.name, height=h, width=w,
                    px_um=px_um, white=white,
                    tissue_frac=float(coarse.mask.mean()),
                    thumb=None, coarse=coarse,
                )
                tile._thumb_raw = thumb_small
                self.tiles.append(tile)

        if bg_accum is not None and bg_n:
            flat = bg_accum / bg_n
            flat = ndi.gaussian_filter(flat, (6, 6, 0))
            flat = flat / np.maximum(np.percentile(flat, 99.5), 1e-4)
            self._flat = np.clip(flat, 0.25, 1.6).astype(np.float32)

        if not self.tiles:
            raise ValueError(
                "no readable images in this folder" +
                (" (" + "; ".join(self.unreadable) + ")" if self.unreadable else ""))
        self._white_ref = np.median(np.stack(whites), axis=0).astype(np.float32)

        # One tissue threshold for the whole set.  Choosing it per tile looks
        # harmless and is not: the cut then depends on how much tissue happens to
        # be in that frame, so two correctly-placed neighbours can disagree about
        # where the tissue is and be reported as contradicting each other.  On a
        # synthetic set with known positions, per-tile thresholds produced
        # exactly that -- mask agreement 0.25 between two tiles placed to within
        # 0.2 px of truth.  The threshold is a property of the stain and the
        # scanner, not of the field of view.
        if p.od_threshold is None:
            thrs = [t.coarse.threshold for t in self.tiles if t.coarse is not None]
            self.od_threshold = float(np.median(thrs)) if thrs else 0.3
            for t in self.tiles:
                t.coarse = remask_level(t.coarse, self.od_threshold, p.highpass_sigma)
                t.tissue_frac = float(t.coarse.mask.mean())
        else:
            self.od_threshold = float(p.od_threshold)

        # Thumbnails are made only now, because flat-fielding needs the whole
        # set: the canvas should show what the mosaic will look like, not the
        # raw frames, or a placement that only looks wrong because of vignetting
        # will be second-guessed by eye.
        for t in self.tiles:
            raw = getattr(t, "_thumb_raw", None)
            if raw is None:
                continue
            t.thumb = np.clip(self.correct(raw, t.index) * 255.0, 0, 255).astype(np.uint8)
            t._thumb_raw = None

        # Pass 2: FFTs, on a shape big enough for the largest pair.
        max_h = max(t.coarse.shape[0] for t in self.tiles)
        max_w = max(t.coarse.shape[1] for t in self.tiles)
        self.fshape = _fast_shape(2 * max_h, 2 * max_w)
        if progress:
            progress(0.70, "preparing correlation transforms")
        with ThreadPoolExecutor(max_workers=max(1, min(p.max_workers, 8))) as ex:
            for t, f in zip(self.tiles, ex.map(
                    lambda t: _pack_fft(t.coarse, self.fshape, workers=1), self.tiles)):
                t._fft = f
        if progress:
            progress(0.80, f"{n} tiles ready")
        return self

    # -- convenience ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def px_um(self) -> float | None:
        vals = [t.px_um for t in self.tiles if t.px_um and np.isfinite(t.px_um)]
        return float(np.median(vals)) if vals else None

    @property
    def coarse_scale(self) -> float:
        return float(self.tiles[0].coarse.scale) if self.tiles else 1.0

    def full_rgb(self, index: int, scale: float = 1.0, corrected: bool = True) -> np.ndarray:
        """Re-read a tile at *scale*, optionally flat-fielded and white-balanced."""
        t = self.tiles[index]
        arr, _ = read_rgb(t.path)
        rgb = to_float_rgb(arr)
        if scale != 1.0:
            rgb = resize(rgb, scale, "area")
        if corrected:
            rgb = self.correct(rgb, index)
        return rgb

    def correct(self, rgb01: np.ndarray, index: int) -> np.ndarray:
        """
        Make blank slide read as blank slide, in every tile and every corner.

        Two separate defects are corrected.  *Vignetting* is a per-pixel gain
        that is the same for every tile, estimated from the average of all tiles
        divided by their own white points -- tissue moves between snapshots, so
        it averages out and what survives is the illumination profile.  The
        *white point* is a per-tile scalar per channel, absorbing lamp drift,
        exposure differences and the faint tint of the slide itself.

        Dividing by both puts empty background at 1.0 in all three channels, so
        the mosaic has no visible tile rectangles and, more importantly, so that
        optical density downstream is measured against a consistent white.
        """
        p = self.params
        out = rgb01.astype(np.float32)
        gain = np.ones((1, 1, 3), dtype=np.float32)
        if p.flatfield and self._flat is not None:
            flat = _resize_to(self._flat, out.shape[:2])
            out = out / np.maximum(flat, 0.2)
        if p.white_balance:
            w = self.tiles[index].white
            if w is not None:
                gain = 1.0 / np.maximum(w.reshape(1, 1, 3), 1e-4)
            out = out * gain
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def level_at(self, index: int, scale: float,
                 params: StitchParams | None = None) -> RegLevel:
        """
        The registration level of one tile at *scale*, built once and kept.

        Refinement and snapping both work pair by pair, so the same tile is
        asked for as many times as it has neighbours: a 60-tile section
        refines ~200 pairs, which is ~400 requests for 60 distinct levels,
        each costing a re-read of the file and a quarter of a second of
        filtering.  Least-recently-used with a byte budget, because the levels
        are a hundred times bigger than the coarse ones and a 200-tile section
        would otherwise fill the machine.

        The two settings that change what a level *is* -- the tissue cut and
        the high-pass -- are part of the key, so a second run with a knob moved
        rebuilds rather than quietly reusing the first run's answer.
        """
        p = params or self.params
        thr = self.od_threshold if self.od_threshold is not None else p.od_threshold
        sigma = p.highpass_sigma * scale / max(self.coarse_scale, 1e-6)
        key = (int(index), round(float(scale), 6), round(float(sigma), 6),
               None if thr is None else round(float(thr), 6))
        with self._levels_lock:
            lvl = self._levels.pop(key, None)
            if lvl is not None:
                self._levels[key] = lvl  # freshest end
                return lvl
            pending = self._levels_pending.get(key)
            mine = pending is None
            if mine:
                pending = self._levels_pending[key] = Future()
        if not mine:
            # Neighbouring pairs go to different threads, so half a dozen of
            # them ask for the same tile within the same instant.  Waiting for
            # the one build costs less than running six: 60 tiles were built
            # 161 times before this.
            return pending.result()

        try:
            lvl = build_reg_level(self.full_rgb(index, scale), 1.0, thr, sigma)
            # Only remask_level reads od_sum, and nothing remasks a refine
            # level; dropping it halves what the cache holds per tile.
            lvl.od_sum = None
        except BaseException as exc:
            with self._levels_lock:
                self._levels_pending.pop(key, None)
            pending.set_exception(exc)
            raise
        size = int(lvl.signal.nbytes + lvl.mask.nbytes)
        with self._levels_lock:
            self._levels_pending.pop(key, None)
            while self._levels and self._levels_bytes + size > LEVEL_CACHE_BYTES:
                _, stale = self._levels.popitem(last=False)
                self._levels_bytes -= int(stale.signal.nbytes + stale.mask.nbytes)
            self._levels[key] = lvl
            self._levels_bytes += size
        pending.set_result(lvl)
        return lvl


def _resize_to(img: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if img.shape[:2] == tuple(shape_hw):
        return img
    if cv2 is not None:
        return cv2.resize(img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
    zoom = [shape_hw[0] / img.shape[0], shape_hw[1] / img.shape[1]] + [1] * (img.ndim - 2)
    return ndi.zoom(img, zoom, order=1)


# =============================================================================
# 5.  Pairwise matching
# =============================================================================


@dataclass
class PairMatch:
    i: int
    j: int
    tx: float  # translation of tile j relative to tile i, FULL-resolution px
    ty: float
    theta: float = 0.0  # rotation of j relative to i, degrees
    ncc: float = 0.0
    peak_ratio: float = 1.0
    n_overlap: float = 0.0
    overlap_frac: float = 0.0
    quality: float = 0.0
    stage: str = "coarse"

    def to_dict(self) -> dict:
        return asdict(self)


def _quality(ncc: float, peak_ratio: float, overlap_frac: float, p: StitchParams) -> float:
    """
    One number in [0, 1] combining the three independent pieces of evidence.

    Each factor saturates, so a spectacular value on one axis cannot rescue a
    failure on another.
    """
    a = np.clip((ncc - p.min_ncc) / (0.85 - p.min_ncc), 0.0, 1.0)
    b = np.clip((peak_ratio - 1.0) / 0.8, 0.0, 1.0)
    c = np.clip(overlap_frac / 0.35, 0.0, 1.0)
    return float((a ** 0.8) * (0.35 + 0.65 * b) * (0.45 + 0.55 * c))


def _rotate_level(level: RegLevel, deg: float) -> RegLevel:
    """Rotate a registration level about its centre, keeping the canvas size."""
    if abs(deg) < 1e-6:
        return level
    h, w = level.shape
    if cv2 is not None:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
        sig = cv2.warpAffine(level.signal, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
        msk = cv2.warpAffine(level.mask.astype(np.uint8), M, (w, h),
                             flags=cv2.INTER_NEAREST, borderValue=0) > 0
    else:
        sig = ndi.rotate(level.signal, deg, reshape=False, order=1, cval=0.0)
        msk = ndi.rotate(level.mask.astype(np.uint8), -0.0 + deg, reshape=False, order=0, cval=0) > 0
    return RegLevel(scale=level.scale, signal=sig.astype(np.float32), mask=msk, shape=(h, w))


def match_pair(
    ts: TileSet,
    i: int,
    j: int,
    params: StitchParams | None = None,
) -> PairMatch | None:
    """Score one tile pair at the coarse level, optionally over rotations."""
    p = params or ts.params
    ti, tj = ts.tiles[i], ts.tiles[j]
    allowed = _valid_shift_window(ti.coarse.shape, tj.coarse.shape, ts.fshape, p.max_shift_frac)
    min_tissue = min(ti.coarse.tissue_px, tj.coarse.tissue_px)
    min_overlap = max(MIN_OVERLAP_PX_COARSE, int(p.min_overlap_frac * min_tissue))
    sat_overlap = max(3.0 * min_overlap, 0.20 * min_tissue)

    angles = [0.0]
    if p.rotation_search_deg > 0:
        step = max(0.25, p.rotation_step_deg)
        k = int(round(p.rotation_search_deg / step))
        angles = [step * a for a in range(-k, k + 1)]

    best: PairMatch | None = None
    for ang in angles:
        if abs(ang) < 1e-9:
            pack_j = tj._fft
        else:
            pack_j = _pack_fft(_rotate_level(tj.coarse, ang), ts.fshape, workers=1)
        ncc, counts = masked_ncc_map(ti._fft, pack_j, ts.fshape, min_overlap=min_overlap, workers=1)
        pk = peak_with_confidence(ncc, counts, ts.fshape, allowed,
                                  saturation_overlap=sat_overlap)
        if not pk.get("ok"):
            continue
        overlap_frac = pk["n_overlap"] / max(min_tissue, 1)
        q = _quality(pk["ncc"], pk["peak_ratio"], overlap_frac, p)
        inv = 1.0 / max(ts.coarse_scale, 1e-6)
        cand = PairMatch(
            i=i, j=j,
            tx=pk["tx"] * inv, ty=pk["ty"] * inv, theta=float(ang),
            ncc=pk["ncc"], peak_ratio=pk["peak_ratio"],
            n_overlap=pk["n_overlap"], overlap_frac=float(overlap_frac),
            quality=q, stage="coarse",
        )
        if best is None or cand.quality > best.quality:
            best = cand

    if best is None:
        return None
    if best.ncc < p.min_ncc or best.peak_ratio < p.min_peak_ratio:
        best.stage = "coarse-weak"
    return best


def match_all_pairs(
    ts: TileSet,
    params: StitchParams | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[PairMatch]:
    """
    Score every candidate tile pair.

    With ``neighbour_window = 0`` this is all ``N(N-1)/2`` pairs, which is the
    honest thing to do when the acquisition order says nothing about geometry
    (manual snapshots usually do not).  Set a window when the tiles came off a
    serpentine stage scan and adjacency is known.
    """
    p = params or ts.params
    n = len(ts)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if p.neighbour_window and (j - i) > p.neighbour_window:
                continue
            pairs.append((i, j))

    out: list[PairMatch] = []
    done = 0
    total = max(len(pairs), 1)

    def work(pair: tuple[int, int]) -> PairMatch | None:
        return match_pair(ts, pair[0], pair[1], p)

    with ThreadPoolExecutor(max_workers=p.max_workers) as ex:
        for res in ex.map(work, pairs):
            done += 1
            if res is not None:
                out.append(res)
            if progress and done % 25 == 0:
                progress(done / total, f"matched {done}/{total} pairs")
    if progress:
        progress(1.0, f"matched {total} pairs")
    return out


def good_pairs(matches: Iterable[PairMatch], params: StitchParams) -> list[PairMatch]:
    return [
        m for m in matches
        if m.ncc >= params.min_ncc
        and m.peak_ratio >= params.min_peak_ratio
        and m.overlap_frac >= params.min_overlap_frac
    ]


# -- verification / refinement at higher resolution ---------------------------


def refine_pair(
    ts: TileSet,
    m: PairMatch,
    params: StitchParams | None = None,
    search_px: int | None = None,
) -> PairMatch:
    """
    Re-solve one placement at ``refine_max_dim`` inside a window around the
    coarse answer.

    Only the predicted overlap is loaded, so this stays cheap even at 4x the
    coarse resolution -- and 4x resolution is what turns a +/-10 full-res pixel
    coarse answer into a sub-pixel one.
    """
    p = params or ts.params
    ti = ts.tiles[m.i]
    scale = min(1.0, p.refine_max_dim / max(ti.height, ti.width))
    if scale <= ts.coarse_scale * 1.05:
        return m

    lvl_i = ts.level_at(m.i, scale, p)
    lvl_j = ts.level_at(m.j, scale, p)
    if abs(m.theta) > 1e-6:
        lvl_j = _rotate_level(lvl_j, m.theta)

    tx0 = m.tx * scale
    ty0 = m.ty * scale
    win = search_px if search_px is not None else int(max(12, 3.0 / ts.coarse_scale * scale))

    hi, wi = lvl_i.shape
    hj, wj = lvl_j.shape
    # Crop to the predicted common region plus the search window.
    ax0 = int(np.floor(max(0, tx0 - win)))
    ay0 = int(np.floor(max(0, ty0 - win)))
    ax1 = int(np.ceil(min(wi, tx0 + wj + win)))
    ay1 = int(np.ceil(min(hi, ty0 + hj + win)))
    if ax1 - ax0 < 24 or ay1 - ay0 < 24:
        return m
    bx0 = int(np.floor(max(0, -tx0 - win)))
    by0 = int(np.floor(max(0, -ty0 - win)))
    bx1 = int(np.ceil(min(wj, wi - tx0 + win)))
    by1 = int(np.ceil(min(hj, hi - ty0 + win)))
    if bx1 - bx0 < 24 or by1 - by0 < 24:
        return m

    sub_i = RegLevel(1.0, lvl_i.signal[ay0:ay1, ax0:ax1], lvl_i.mask[ay0:ay1, ax0:ax1],
                     (ay1 - ay0, ax1 - ax0))
    sub_j = RegLevel(1.0, lvl_j.signal[by0:by1, bx0:bx1], lvl_j.mask[by0:by1, bx0:bx1],
                     (by1 - by0, bx1 - bx0))
    fshape = _fast_shape(sub_i.shape[0] + sub_j.shape[0], sub_i.shape[1] + sub_j.shape[1])
    pa = _pack_fft(sub_i, fshape, workers=1)
    pb = _pack_fft(sub_j, fshape, workers=1)
    min_tissue = min(sub_i.tissue_px, sub_j.tissue_px)
    min_overlap = max(200, int(0.15 * min_tissue))
    ncc, counts = masked_ncc_map(pa, pb, fshape, min_overlap=min_overlap, workers=1)
    sat_overlap = max(3.0 * min_overlap, 0.30 * min_tissue)

    # Restrict to the search window around the coarse prediction.
    fy, fx = fshape
    ty_axis = np.where(np.arange(fy) < fy // 2, np.arange(fy), np.arange(fy) - fy)
    tx_axis = np.where(np.arange(fx) < fx // 2, np.arange(fx), np.arange(fx) - fx)
    # Where sub_j's origin sits inside sub_i.  Tile j starts at tx0 in tile i;
    # the crops then move both origins, j's out by bx0 and i's in by ax0.
    pred_tx = tx0 + bx0 - ax0
    pred_ty = ty0 + by0 - ay0
    allowed = (np.abs(ty_axis - pred_ty)[:, None] <= win) & (np.abs(tx_axis - pred_tx)[None, :] <= win)
    pk = peak_with_confidence(ncc, counts, fshape, allowed, exclusion_px=max(4, win // 4),
                              saturation_overlap=sat_overlap)
    if not pk.get("ok"):
        return m

    tx_full = (pk["tx"] - bx0 + ax0) / scale
    ty_full = (pk["ty"] - by0 + ay0) / scale
    overlap_frac = pk["n_overlap"] / max(min(lvl_i.tissue_px, lvl_j.tissue_px), 1)
    return PairMatch(
        i=m.i, j=m.j, tx=float(tx_full), ty=float(ty_full), theta=m.theta,
        # The coarse ratio is kept, not remeasured.  peak_ratio asks whether
        # some *other* place matches nearly as well, and the coarse pass asked
        # that over the whole search space.  Here the search is a +-12 px box,
        # so the runner-up is five pixels down the same hill and the ratio is
        # ~1.05 however good the match is -- half the good pairs used to be
        # thrown out for the crime of having been refined.
        ncc=float(pk["ncc"]), peak_ratio=m.peak_ratio,
        n_overlap=float(pk["n_overlap"]), overlap_frac=float(overlap_frac),
        quality=_quality(pk["ncc"], m.peak_ratio, overlap_frac, p),
        stage="refined",
    )


def refine_matches(
    ts: TileSet,
    matches: Sequence[PairMatch],
    params: StitchParams | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[PairMatch]:
    """Refine the best few edges per tile; leave the rest at coarse accuracy."""
    p = params or ts.params
    by_tile: dict[int, list[PairMatch]] = {}
    for m in matches:
        by_tile.setdefault(m.i, []).append(m)
        by_tile.setdefault(m.j, []).append(m)
    chosen: dict[tuple[int, int], PairMatch] = {}
    for idx, lst in by_tile.items():
        for m in sorted(lst, key=lambda x: -x.quality)[: p.refine_top_k]:
            chosen[(m.i, m.j)] = m

    todo = list(chosen.values())
    out: list[PairMatch] = []
    done = 0

    def work(m: PairMatch) -> PairMatch:
        try:
            return refine_pair(ts, m, p)
        except Exception:
            return m

    # Ten, not one per core: a refining thread holds two cropped levels and
    # nine transforms of twice their size, so this is the point where the
    # memory bus runs out before the cores do (7 -> 10 threads buys 0.7 s on
    # a 60-tile section, 10 -> 15 buys 0.2 s for another 600 MB).
    with ThreadPoolExecutor(max_workers=max(1, min(p.max_workers, 10))) as ex:
        for res in ex.map(work, todo):
            done += 1
            out.append(res)
            if progress and done % 5 == 0:
                progress(done / max(len(todo), 1), f"refined {done}/{len(todo)} pairs")

    refined = {(m.i, m.j): m for m in out}
    final = []
    for m in matches:
        final.append(refined.get((m.i, m.j), m))
    if progress:
        progress(1.0, f"refined {len(todo)} pairs")
    return final


# =============================================================================
# 6.  Global layout
# =============================================================================


@dataclass
class Pose:
    """Where a tile sits in the mosaic, in full-resolution pixels."""

    x: float = 0.0  # position of the tile's top-left corner (before rotation)
    y: float = 0.0
    theta: float = 0.0  # degrees, about the tile centre
    island: int = 0
    placed: bool = False
    locked: bool = False  # a human put it there; auto-solve must not move it

    def to_dict(self) -> dict:
        return asdict(self)

def _coarse_of(ts: "TileSet", idx: int, theta: float, _cache: dict) -> RegLevel:
    """Coarse registration level of a tile at a given rotation, memoised."""
    key = (idx, round(float(theta), 3))
    lvl = _cache.get(key)
    if lvl is None:
        lvl = ts.tiles[idx].coarse
        if abs(theta) > 1e-6:
            lvl = _rotate_level(lvl, theta)
        _cache[key] = lvl
    return lvl


def agreement(
    ts: "TileSet",
    i: int,
    j: int,
    dx_full: float,
    dy_full: float,
    theta_i: float = 0.0,
    theta_j: float = 0.0,
    _cache: dict | None = None,
) -> dict[str, float]:
    """
    How well do two tiles agree *at a placement we already have in mind*?

    This is not a search -- it scores one specific relative offset.  It is the
    test that catches the failure mode a spanning tree cannot see: a chain of
    individually-plausible matches that walks the layout around a loop and lands
    a piece of vessel on top of a different piece of vessel.  Those two tiles
    never got compared, because no edge joins them; here they do.

    Returns the correlation over the shared tissue, the tissue areas, and the
    mask agreement (a tile claiming tissue where another claims empty slide is a
    contradiction even if the correlation is undefined).
    """
    cache = _cache if _cache is not None else {}
    A = _coarse_of(ts, i, theta_i, cache)
    B = _coarse_of(ts, j, theta_j, cache)
    cs = ts.coarse_scale
    dx, dy = int(round(dx_full * cs)), int(round(dy_full * cs))

    ha, wa = A.shape
    hb, wb = B.shape
    ax0, ay0 = max(0, dx), max(0, dy)
    ax1, ay1 = min(wa, dx + wb), min(ha, dy + hb)
    if ax1 - ax0 < 8 or ay1 - ay0 < 8:
        return {"area": 0.0, "ncc": np.nan, "n_both": 0.0, "n_a": 0.0, "n_b": 0.0, "mask_iou": np.nan}

    bx0, by0 = ax0 - dx, ay0 - dy
    hh, ww = ay1 - ay0, ax1 - ax0
    sa = A.signal[ay0:ay1, ax0:ax1]
    sb = B.signal[by0:by0 + hh, bx0:bx0 + ww]
    ma = A.mask[ay0:ay1, ax0:ax1]
    mb = B.mask[by0:by0 + hh, bx0:bx0 + ww]

    both = ma & mb
    n_both = float(both.sum())
    n_a, n_b = float(ma.sum()), float(mb.sum())
    union = n_a + n_b - n_both
    iou = n_both / union if union > 0 else np.nan

    ncc = np.nan
    if n_both >= 250:
        a = sa[both].astype(np.float64)
        b = sb[both].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if denom > 1e-9:
            ncc = float((a * b).sum() / denom)
    return {"area": float(hh * ww), "ncc": ncc, "n_both": n_both,
            "n_a": n_a, "n_b": n_b, "mask_iou": float(iou) if np.isfinite(iou) else np.nan}


def _contradicts(
    ag: dict[str, float],
    accept_ncc: float = 0.35,
    min_iou: float = 0.55,
    min_tissue: float = 600.0,
    min_area: float = 1200.0,
) -> bool:
    """
    Is this pair of placements mutually impossible?

    Two tiles placed in a mosaic make a claim about the same physical piece of
    slide, so they must agree about it.  Disagreement comes in two flavours and
    both are checked, because either alone lets a bad layout through.

    *Where is the tissue?*  ``mask_iou`` compares the two tiles' tissue masks
    over the region they share.  On hand-verified true overlaps of these
    trichrome aorta tiles it runs 0.81-0.99; on the collisions a spanning-tree
    layout produces it runs 0.15-0.22.  Nothing lands in between.  This is the
    stronger test, and it works even when the shared tissue is too small to
    correlate -- the case where one tile shows wall and the other shows empty
    slide, which correlation cannot score at all because there is nothing to
    correlate.

    *Is it the same tissue?*  Where enough tissue is shared, the correlation of
    the two registration signals must be high: 0.45-0.94 on true pairs against
    -0.02-0.38 on collisions.

    Overlaps too small or too empty to carry information are not judged; they
    return ``False`` and the decision rests on other neighbours.
    """
    n_both, n_a, n_b = ag["n_both"], ag["n_a"], ag["n_b"]
    big = max(n_a, n_b)
    if ag["area"] < min_area or big < min_tissue:
        return False
    measurable = n_both >= min_tissue and np.isfinite(ag["ncc"])
    # A convincing correlation settles it.  Mask disagreement can also come from
    # the two tiles thresholding faint material differently, so it must not
    # override direct evidence that the shared tissue is the same tissue.
    if measurable and ag["ncc"] >= max(0.55, accept_ncc + 0.15):
        return False
    if np.isfinite(ag["mask_iou"]) and ag["mask_iou"] < min_iou:
        return True
    if measurable and ag["ncc"] < accept_ncc:
        return True
    return False


def _solve_layout_once(
    ts: "TileSet",
    matches: Sequence[PairMatch],
    params: StitchParams | None = None,
    locked: dict[int, Pose] | None = None,
    accept_ncc: float = 0.35,
) -> tuple[list[Pose], dict[str, Any]]:
    """
    Place every tile, refusing any placement that contradicts one already made.

    The assembly is greedy but *verified*.  Tiles enter the mosaic in order of
    match quality; before a tile is accepted, it is compared against every tile
    already placed whose footprint it would overlap -- not just the one that
    proposed it.  A proposal that would drop this piece of wall on top of a
    different piece of wall is rejected, and the tile waits for a better offer.

    This is the difference that matters.  A maximum spanning tree uses exactly
    one edge per tile and therefore never notices that the resulting layout is
    self-contradictory; on the 54-tile aortic arch it silently stacked five
    tiles from one limb of the arch onto four from another, because each
    individual match in the chain looked convincing.

    Afterwards, positions are polished by robust least squares over all edges
    that survived verification, so the layout uses every consistent measurement
    rather than only the tree.

    Tiles nobody could place consistently become their own island, laid out
    beside the mosaic for the user to drag into position by hand.
    """
    p = params or ts.params
    n_tiles = len(ts)
    good = sorted(good_pairs(matches, p), key=lambda m: -m.quality)

    adj: dict[int, list[tuple[int, PairMatch, int]]] = {i: [] for i in range(n_tiles)}
    for m in good:
        adj[m.i].append((m.j, m, +1))
        adj[m.j].append((m.i, m, -1))

    poses: list[Pose | None] = [None] * n_tiles
    island_of: dict[int, int] = {}
    cache: dict = {}
    accepted_edges: list[PairMatch] = []
    n_rejected = 0

    def propose(from_idx: int, m: PairMatch, sign: int) -> Pose:
        base = poses[from_idx]
        assert base is not None
        return Pose(
            x=base.x + sign * m.tx, y=base.y + sign * m.ty,
            theta=base.theta + sign * m.theta, placed=True,
        )

    def conflicts(idx: int, cand: Pose, members: Sequence[int]) -> tuple[bool, float]:
        """Check a candidate placement against everything already down."""
        worst = 1.0
        for k in members:
            if k == idx or poses[k] is None:
                continue
            if _overlap_area(poses[k], ts.tiles[k], cand, ts.tiles[idx]) < 400:
                continue
            ag = agreement(ts, k, idx, cand.x - poses[k].x, cand.y - poses[k].y,
                           poses[k].theta, cand.theta, cache)
            if _contradicts(ag, accept_ncc):
                return True, float(ag["ncc"]) if np.isfinite(ag["ncc"]) else -1.0
            if np.isfinite(ag["ncc"]) and ag["n_both"] > 900:
                worst = min(worst, float(ag["ncc"]))
        return False, worst

    # Hand-placed tiles are truth: they seed the first island and never move.
    island = 0
    if locked:
        members: list[int] = []
        for idx, lp in locked.items():
            if 0 <= idx < n_tiles:
                poses[idx] = Pose(**{**asdict(lp), "placed": True, "locked": True, "island": 0})
                island_of[idx] = 0
                members.append(idx)
        if members:
            island = 1

    while True:
        unplaced = [i for i in range(n_tiles) if poses[i] is None]
        if not unplaced:
            break
        unplaced_set = set(unplaced)

        # Grow whatever is already down before starting a new island.
        members = [i for i in range(n_tiles) if poses[i] is not None]
        if not members:
            seed = next((m for m in good if m.i in unplaced_set and m.j in unplaced_set), None)
            if seed is None:
                for k, idx in enumerate(sorted(unplaced)):
                    poses[idx] = Pose(x=0.0, y=0.0, island=island + k, placed=True)
                    island_of[idx] = island + k
                break
            poses[seed.i] = Pose(x=0.0, y=0.0, theta=0.0, island=island, placed=True)
            poses[seed.j] = Pose(x=seed.tx, y=seed.ty, theta=seed.theta, island=island, placed=True)
            island_of[seed.i] = island_of[seed.j] = island
            accepted_edges.append(seed)
            members = [seed.i, seed.j]

        # Frontier growth, best edge first.
        grew = True
        while grew:
            grew = False
            placed_now = {i for i in range(n_tiles) if poses[i] is not None}
            cands: list[tuple[float, int, int, PairMatch, int]] = []
            for a in placed_now:
                for b, m, sign in adj[a]:
                    if b in placed_now:
                        continue
                    cands.append((-m.quality, a, b, m, sign))
            cands.sort(key=lambda c: c[0])
            for _q, a, b, m, sign in cands:
                if poses[b] is not None:
                    continue
                isl = poses[a].island
                cand = propose(a, m, sign)
                cand.island = isl
                same_island = [k for k in range(n_tiles)
                               if poses[k] is not None and poses[k].island == isl]
                bad, _worst = conflicts(b, cand, same_island)
                if bad:
                    n_rejected += 1
                    continue
                poses[b] = cand
                island_of[b] = isl
                accepted_edges.append(m)
                grew = True

        # Nothing else attaches to the current mosaic: start a fresh island.
        left = [i for i in range(n_tiles) if poses[i] is None]
        if not left:
            break
        island = max(island_of.values(), default=-1) + 1
        left_set = set(left)
        seed = next((m for m in good if m.i in left_set and m.j in left_set), None)
        if seed is None:
            for k, idx in enumerate(sorted(left)):
                poses[idx] = Pose(x=0.0, y=0.0, island=island + k, placed=True)
                island_of[idx] = island + k
            break
        poses[seed.i] = Pose(x=0.0, y=0.0, theta=0.0, island=island, placed=True)
        poses[seed.j] = Pose(x=seed.tx, y=seed.ty, theta=seed.theta, island=island, placed=True)
        island_of[seed.i] = island_of[seed.j] = island
        accepted_edges.append(seed)

    final: list[Pose] = []
    for i in range(n_tiles):
        q = poses[i] or Pose(x=0.0, y=0.0, island=i, placed=True)
        q.island = island_of.get(i, q.island)
        q.placed = True
        final.append(q)

    # ---- polish: robust least squares over every consistent edge -----------
    consistent: list[PairMatch] = []
    for m in good:
        if final[m.i].island != final[m.j].island:
            continue
        ag = agreement(ts, m.i, m.j, final[m.j].x - final[m.i].x, final[m.j].y - final[m.i].y,
                       final[m.i].theta, final[m.j].theta, cache)
        if _contradicts(ag, accept_ncc):
            continue
        # The edge must also predict roughly where the tile already sits.
        if math.hypot(final[m.j].x - final[m.i].x - m.tx,
                      final[m.j].y - final[m.i].y - m.ty) > 0.25 * ts.tiles[m.i].width:
            continue
        consistent.append(m)

    islands: dict[int, list[int]] = {}
    for i, q in enumerate(final):
        islands.setdefault(q.island, []).append(i)

    locked_idx = set(locked or {})
    for isl, members in islands.items():
        if len(members) < 3:
            continue
        mset = set(members)
        isl_edges = [m for m in consistent if m.i in mset and m.j in mset]
        if len(isl_edges) <= len(members) - 1:
            continue
        anchor = next((i for i in members if i in locked_idx), members[0])
        order = [anchor] + [i for i in members if i != anchor]
        for _ in range(3):
            sol, resid = _least_squares_positions(order, isl_edges, final)
            for k, idx in enumerate(order):
                if idx in locked_idx:
                    continue
                final[idx].x, final[idx].y = float(sol[k, 0]), float(sol[k, 1])
            if resid is None or len(isl_edges) <= len(members) - 1:
                break
            cut = max(6.0, 3.0 * float(np.median(resid)))
            keep = resid <= cut
            if keep.all():
                break
            isl_edges = [e for e, k in zip(isl_edges, keep) if k]
            if len(isl_edges) < len(members) - 1:
                break

    ordered = [islands[k] for k in sorted(islands, key=lambda k: -len(islands[k]))]
    for new_id, members in enumerate(ordered):
        for i in members:
            final[i].island = new_id
    if not locked:
        _arrange_islands(final, ordered, ts.tiles)
    else:
        _arrange_islands(final, [m for m in ordered if not (set(m) & locked_idx)], ts.tiles)

    final, merge_stats = merge_islands(ts, final, matches, p, accept_ncc)
    islands = {}
    for i, q in enumerate(final):
        islands.setdefault(q.island, []).append(i)
    ordered = [islands[k] for k in sorted(islands)]

    stats = {
        "n_edges_scored": len(matches),
        "n_edges_used": len(good),
        "n_island_merges": merge_stats["n_merges"],
        "merge_suggestions": merge_stats.get("suggestions", []),
        "n_edges_accepted": len(accepted_edges),
        "n_edges_consistent": len(consistent),
        "n_proposals_rejected": n_rejected,
        "n_islands": len(ordered),
        "island_sizes": [len(m) for m in ordered],
        "assembled": "every pair at once",
    }
    return final, stats


def solve_layout(
    ts: "TileSet",
    matches: Sequence[PairMatch],
    params: StitchParams | None = None,
    locked: dict[int, Pose] | None = None,
    accept_ncc: float = 0.35,
) -> tuple[list[Pose], dict[str, Any]]:
    """
    Assemble the mosaic, trying the acquisition order as a second opinion.

    Two tiles photographed one after another overlap because that is how the
    section was walked.  Two tiles far apart in the sequence overlap only if
    the walk came back past itself -- which happens, so those pairs cannot be
    dropped, but they are not equally likely and one kind of mistake is theirs
    alone: a featureless stretch of wall matching a similar stretch somewhere
    else.  On one 54-tile arch that pulled five tiles out of the branch they
    belonged to and left a hole where they had been.

    So the sequence is also laid down on its own -- consecutive runs first,
    then joined as whole rigid bodies using every pair, which is much harder to
    get wrong, because a bad join misplaces a dozen tiles at once and the
    contradiction test sees all of them.

    Neither route wins everywhere.  Laying the runs down first rescues a
    section whose tissue repeats; it does worse on one photographed as two
    separate loops, where the runs are correct but no single join is clean
    enough to put them back together.  Rather than guess which kind of section
    this is, both are assembled and the better one is kept: more tiles in a
    single island first, then -- when both routes place the same number, which
    is the common case -- fewer contradicted seams, then the higher mean
    agreement.  Tile count alone used to decide it, and on two sections that
    silently preferred a layout with contradictions to an identical-sized one
    without any.  Matching is not repeated: the consecutive pairs are already
    in *matches*, and only the placement runs twice.

    Finally, any group the winner holds by a sliver of wall is settled on the
    tissue rather than on the correlation peak; see ``settle_sliver_joins``.

    ``stats["assembled"]`` says which route won.  ``chain_window = 0`` skips the
    second attempt.
    """
    p = params or ts.params
    solo = p.copy_with(chain_window=0)
    plain = _solve_layout_once(ts, matches, solo, locked, accept_ncc)

    def finish(result):
        if locked:
            return result
        poses, stats = result
        poses, notes = settle_sliver_joins(ts, list(poses), accept_ncc)
        return poses, {**stats, "sliver_joins": notes,
                       "n_sliver_settled": sum(1 for nt in notes if nt["moved"])}

    if not p.chain_window or locked:
        return finish(plain)

    near = [m for m in matches if abs(m.i - m.j) <= p.chain_window]
    if not near or len(near) >= len(matches):
        return finish(plain)

    runs, _ = _solve_layout_once(ts, near, solo, locked, accept_ncc)
    poses, ms = merge_islands(ts, runs, matches, p, accept_ncc)
    chained = (poses, {**plain[1],
                       "n_island_merges": ms["n_merges"],
                       "merge_suggestions": ms.get("suggestions", []),
                       "n_islands": ms["n_islands"],
                       "island_sizes": ms["island_sizes"],
                       "assembled": f"acquisition order first (window {p.chain_window})"})

    def rank(result):
        rposes, rstats = result
        sizes = rstats.get("island_sizes") or [0]
        qc = layout_qc(ts, rposes, accept_ncc)
        score = float(np.nanmean(qc.score)) if len(qc) and qc.score.notna().any() else 0.0
        return (sizes[0], -int(qc.n_conflicts.sum()), score)

    return finish(max((plain, chained), key=rank))


def _evaluate_shift(
    ts: "TileSet",
    poses: Sequence[Pose],
    moving: Sequence[int],
    fixed: Sequence[int],
    dx: float,
    dy: float,
    dth: float,
    cache: dict,
    accept_ncc: float = 0.35,
) -> dict[str, Any]:
    """Score moving *all* of one island onto another by a single rigid shift."""
    conflicts: list[tuple[int, int]] = []
    support = 0.0
    n_agree = 0
    best = -1.0
    for k in moving:
        tp = Pose(x=poses[k].x + dx, y=poses[k].y + dy,
                  theta=poses[k].theta + dth, placed=True)
        for f in fixed:
            if _overlap_area(poses[f], ts.tiles[f], tp, ts.tiles[k]) < 400:
                continue
            ag = agreement(ts, f, k, tp.x - poses[f].x, tp.y - poses[f].y,
                           poses[f].theta, tp.theta, cache)
            if _contradicts(ag, accept_ncc):
                conflicts.append((f, k))
            elif np.isfinite(ag["ncc"]) and ag["n_both"] > 900 and ag["ncc"] > 0.5:
                support += float(ag["n_both"])
                n_agree += 1
                best = max(best, float(ag["ncc"]))
    return {"dx": dx, "dy": dy, "dtheta": dth, "conflicts": conflicts,
            "n_conflicts": len(conflicts), "support": support,
            "n_agree": n_agree, "best_ncc": best}


def merge_islands(
    ts: "TileSet",
    poses: list[Pose],
    matches: Sequence[PairMatch],
    params: StitchParams | None = None,
    accept_ncc: float = 0.35,
    max_rounds: int = 8,
    apply: bool = True,
) -> tuple[list[Pose], dict[str, Any]]:
    """
    Try to slot whole islands into each other, as rigid bodies.

    A tile that could not be attached one at a time may still be placeable as
    part of a group: one cross-island match fixes the position of *every* tile
    in the smaller island at once, and the whole arrangement can then be
    checked.  That is much stronger evidence than the single edge, because a
    wrong merge misplaces a dozen tiles and the contradiction test sees them
    all.

    Candidates are ranked by how much tissue actually agrees after the move, not
    by the score of the edge that proposed it -- on self-similar tissue the
    best-scoring edge is not reliably the best placement, but the amount of
    matching tissue it produces across the whole island is.

    Only merges with *no* contradictions are applied.  A merge with strong
    support that would nevertheless collide is reported in ``suggestions``
    rather than performed, because it means one of the two islands is internally
    misplaced and the honest thing is to show the user where, not to guess.
    """
    p = params or ts.params
    good = sorted(good_pairs(matches, p), key=lambda m: -m.quality)
    cache: dict = {}
    merged = 0
    log: list[dict] = []
    suggestions: list[dict] = []

    for _ in range(max_rounds):
        islands: dict[int, list[int]] = {}
        for i, q in enumerate(poses):
            islands.setdefault(q.island, []).append(i)
        if len(islands) < 2:
            break

        # Collect every distinct rigid placement any cross-island edge proposes.
        cands: dict[tuple[int, int], list[dict]] = {}
        for m in good:
            isl_i, isl_j = poses[m.i].island, poses[m.j].island
            if isl_i == isl_j:
                continue
            if len(islands[isl_i]) >= len(islands[isl_j]):
                fixed, moving, af, am, sign = isl_i, isl_j, m.i, m.j, +1
            else:
                fixed, moving, af, am, sign = isl_j, isl_i, m.j, m.i, -1
            if any(poses[k].locked for k in islands[moving]):
                continue
            dx = poses[af].x + sign * m.tx - poses[am].x
            dy = poses[af].y + sign * m.ty - poses[am].y
            dth = poses[af].theta + sign * m.theta - poses[am].theta
            bucket = cands.setdefault((moving, fixed), [])
            tol = 0.08 * ts.tiles[am].width
            if any(abs(dx - c["dx"]) < tol and abs(dy - c["dy"]) < tol for c in bucket):
                continue
            bucket.append({"dx": dx, "dy": dy, "dth": dth, "edge": m})

        scored: list[dict] = []
        for (moving, fixed), bucket in cands.items():
            for c in bucket:
                r = _evaluate_shift(ts, poses, islands[moving], islands[fixed],
                                    c["dx"], c["dy"], c["dth"], cache, accept_ncc)
                r.update({"moving": moving, "fixed": fixed, "edge": c["edge"]})
                scored.append(r)
        if not scored:
            break

        clean = [r for r in scored if r["n_conflicts"] == 0 and r["n_agree"] >= 1
                 and r["best_ncc"] >= 0.45]
        clean.sort(key=lambda r: -r["support"])
        dirty = [r for r in scored if r["n_conflicts"] > 0 and r["support"] > 0]
        dirty.sort(key=lambda r: -r["support"])
        for r in dirty[:3]:
            suggestions.append({
                "moving_island": r["moving"], "into_island": r["fixed"],
                "dx": r["dx"], "dy": r["dy"], "dtheta": r["dtheta"],
                "support_px": r["support"], "n_agreeing_pairs": r["n_agree"],
                "n_conflicts": r["n_conflicts"],
                "conflicting_tiles": sorted({ts.tiles[f].name for f, _k in r["conflicts"]}),
                "proposed_by": f"{ts.tiles[r['edge'].i].name} - {ts.tiles[r['edge'].j].name}",
            })
        if not clean or not apply:
            break

        best = clean[0]
        for k in islands[best["moving"]]:
            poses[k] = Pose(x=poses[k].x + best["dx"], y=poses[k].y + best["dy"],
                            theta=poses[k].theta + best["dtheta"],
                            island=best["fixed"], placed=True, locked=poses[k].locked)
        log.append({"moved_island": best["moving"], "into": best["fixed"],
                    "n_tiles": len(islands[best["moving"]]),
                    "support_px": best["support"], "best_ncc": best["best_ncc"]})
        merged += 1

    islands = {}
    for i, q in enumerate(poses):
        islands.setdefault(q.island, []).append(i)
    ordered = [islands[k] for k in sorted(islands, key=lambda k: -len(islands[k]))]
    for new_id, members in enumerate(ordered):
        for i in members:
            poses[i].island = new_id
    locked_any = any(q.locked for q in poses)
    _arrange_islands(
        poses,
        ordered if not locked_any else [m for m in ordered
                                        if not any(poses[i].locked for i in m)],
        ts.tiles,
    )
    # Keep only suggestions whose islands still exist separately.
    live = {q.island for q in poses}
    suggestions = [s for s in suggestions if len(live) > 1][:6]
    return poses, {"n_merges": merged, "merges": log, "suggestions": suggestions,
                   "n_islands": len(ordered), "island_sizes": [len(m) for m in ordered]}


def layout_qc(
    ts: "TileSet",
    poses: Sequence[Pose],
    accept_ncc: float = 0.35,
) -> pd.DataFrame:
    """
    Per-tile confidence in a finished layout, however it was produced.

    Every pair of tiles whose footprints overlap is re-scored at the offset the
    layout implies.  A tile's score is the tissue-weighted mean agreement with
    its neighbours; ``n_conflicts`` counts neighbours it flatly contradicts.
    The browser canvas colours tiles by this, so a bad placement is visible
    without hunting for it.
    """
    cache: dict = {}
    n = len(poses)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (poses[i].placed and poses[j].placed):
                continue
            if poses[i].island != poses[j].island:
                continue  # parked islands merely sit near each other
            if _overlap_area(poses[i], ts.tiles[i], poses[j], ts.tiles[j]) < 400:
                continue
            ag = agreement(ts, i, j, poses[j].x - poses[i].x, poses[j].y - poses[i].y,
                           poses[i].theta, poses[j].theta, cache)
            rows.append({
                "i": i, "j": j, "file_i": ts.tiles[i].name, "file_j": ts.tiles[j].name,
                "ncc": ag["ncc"], "n_both": ag["n_both"], "mask_iou": ag["mask_iou"],
                "conflict": bool(_contradicts(ag, accept_ncc)),
            })
    pair_df = pd.DataFrame(rows)

    out = []
    for i in range(n):
        if pair_df.empty:
            out.append({"index": i, "file": ts.tiles[i].name, "score": np.nan,
                        "n_neighbours": 0, "n_conflicts": 0, "island": poses[i].island})
            continue
        sub = pair_df[(pair_df.i == i) | (pair_df.j == i)]
        usable = sub[np.isfinite(sub.ncc) & (sub.n_both > 500)]
        score = float((usable.ncc * usable.n_both).sum() / usable.n_both.sum()) if len(usable) else np.nan
        out.append({
            "index": i, "file": ts.tiles[i].name, "score": score,
            "n_neighbours": int(len(sub)), "n_conflicts": int(sub.conflict.sum()),
            "island": poses[i].island,
        })
    df = pd.DataFrame(out)
    df.attrs["pairs"] = pair_df
    return df


def _components(members: Sequence[int], edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Connected groups of *members* under *edges*, biggest first."""
    parent = {i: i for i in members}

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in members:
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def _media_raster(ts: "TileSet", poses: Sequence[Pose], indices: Sequence[int],
                  scale: float, frame: tuple[float, float, int, int],
                  cut: float | None = None) -> tuple[np.ndarray, np.ndarray, float | None]:
    """The darkest half of the tissue -- the muscle band, without adventitia or blood.

    ``cut`` is the optical density the band starts at; left None it is the
    median of this raster's own tissue.  Two pieces of one mosaic have to be
    given the same cut, or one comes out with a thicker wall than the other for
    no better reason than having been thresholded on its own histogram.

    The second mask says where these tiles actually put pixels.  Wherever the
    band runs up against that edge it stops because the picture stops, and a
    border read off it belongs to the tile, not to the vessel.
    """
    x0, y0, height, width = frame
    out = np.zeros((height + 2, width + 2), bool)
    seen = np.zeros_like(out)
    if not len(indices):
        return out, seen, cut
    img, painted, cov = render_mosaic(ts, poses, scale=scale, indices=sorted(indices),
                                      return_coverage=True)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    od = -np.log10(np.clip(img, 1e-3, 1.0)).sum(2)
    tissue = (od > 0.6) & painted
    if int(tissue.sum()) < 50:
        return out, seen, cut
    if cut is None:
        cut = float(np.median(od[tissue]))
    band = (od > cut) & painted
    oy = int(round((cov["y0"] - y0) * scale))
    ox = int(round((cov["x0"] - x0) * scale))
    out[oy:oy + band.shape[0], ox:ox + band.shape[1]] = band
    seen[oy:oy + painted.shape[0], ox:ox + painted.shape[1]] = painted
    return out, seen, cut


def _band_axis(mask: np.ndarray, cx: float, cy: float, radius: float):
    """Centroid, long axis and flatness of the band within *radius* of (cx, cy)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    keep = (xs - cx) ** 2 + (ys - cy) ** 2 < radius * radius
    if int(keep.sum()) < 60:
        return None
    pts = np.c_[xs[keep], ys[keep]].astype(float)
    centre = pts.mean(0)
    _u, sv, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0] / max(float(np.linalg.norm(vt[0])), 1e-12)
    return centre, axis, float(sv[1] / max(sv[0], 1e-9))


def _wall_track(mask: np.ndarray, seen: np.ndarray, cx: float, cy: float,
                direction: np.ndarray, sign: float, reach: float,
                limit: float | None = None, bins: int = 12, grow: float = 2.0,
                gap_px: float = 12.0, min_n: int = 40, probe: float = 4.0):
    """Follow the wall away from (cx, cy) and report where its borders are heading.

    A straight fit to everything within reach of the contact is not this wall.
    A window that size also holds the branch the wall runs into, and on a small
    lumen the far side of the same vessel, and both drag the line off the
    tissue it claims to describe -- on one aortic arch the fitted line left the
    wall entirely and sat on a branch 300 px away.  So walk outwards in bins
    instead, and in each bin keep only the run of pixels nearest to where the
    wall was in the last one: the far side of a lumen never touches the near
    side, so it is never picked up.  Where the band swells past ``grow`` times
    its own width the walk has wandered into a branch, and the fit stops there.

    Each bin gives three numbers, not one: the two borders of the wall and the
    middle of it.  The middle is the worst of the three to compare across a
    gap, because it moves whenever the wall thickens -- and at a branch root it
    does.  The borders are the vessel itself.  The middle is kept only for the
    case where neither border survives, and is taken as the median of the bin
    rather than the midpoint between its edges, so that whatever hangs off one
    side of the wall moves it as little as possible.

    A border only counts where the picture continues past it.  Where it does
    not, the band stopped because the tile stopped, and the border it draws
    there is the edge of the picture -- straight, convincing, and in exactly
    the place the two pieces are being compared.  Those bins are dropped and
    the border is extrapolated from the stretch that is still inside the
    picture, which may begin some way out: the walk therefore runs as far as
    ``limit``, and each border is fitted over its own ``reach`` of real tissue
    starting where it first appears.  How far back it then has to be carried
    is reported with it, because a border found late is a border the caller
    should believe less than one that was there from the start.

    Returns the middle's offset at the contact, its slope, the number of bins
    the walk held for, the wall's width and the middle's residual, and last a
    dict with ``"minus"`` and ``"plus"`` -- each either None or that border's
    own (offset, slope, residual, bins used, distance back to the contact).
    Offsets and slopes are all measured in the frame whose first axis is
    ``direction``, so the two sides of a gap can be compared directly.
    """
    perp = np.array([-direction[1], direction[0]])
    ys, xs = np.nonzero(mask)
    if len(ys) < min_n:
        return None
    rel = np.c_[xs, ys].astype(float) - np.array([cx, cy])
    along = (rel @ direction) * sign
    across = rel @ perp
    limit = max(reach, limit or reach)
    keep = (along >= 0) & (along < limit)
    along, across = along[keep], across[keep]
    step = reach / bins                       # the bin stays the same size
    bins = int(math.ceil(limit / step))       # however far the walk has to go

    def outside(a: float, b: float) -> bool:
        """Is the picture already over, this far across the wall?"""
        x, y = np.array([cx, cy]) + a * sign * direction + b * perp
        iy, ix = int(round(y)), int(round(x))
        if not (0 <= iy < seen.shape[0] and 0 <= ix < seen.shape[1]):
            return True
        return not bool(seen[iy, ix])

    rows: list[tuple[float, float, float, bool, bool, float]] = []
    last = 0.0
    for b in range(bins):
        here = (along >= b * step) & (along < (b + 1) * step)
        if int(here.sum()) < min_n:
            if rows:
                break
            continue
        vals = np.sort(across[here])
        runs = np.split(vals, np.nonzero(np.diff(vals) > gap_px)[0] + 1)
        run = min(runs, key=lambda r: abs(float(np.median(r)) - last))
        if len(run) < min_n:
            if rows:
                break
            continue
        lo, hi = float(run[0]), float(run[-1])
        centre, wide = float(np.median(run)), hi - lo
        if rows and abs(centre - last) > max(wide, rows[0][2] - rows[0][1]):
            break                                  # the walk jumped off the wall
        a = (b + 0.5) * step
        rows.append((a, lo, hi, not outside(a, lo - probe),
                     not outside(a, hi + probe), centre))
        last = centre
    if len(rows) < 3:
        return None
    widths = np.array([r[2] - r[1] for r in rows])
    typical = float(np.median(widths[:max(3, len(widths) // 2)]))
    swollen = np.nonzero(widths > grow * typical)[0]
    if len(swollen):
        rows = rows[:swollen[0]]
    if len(rows) < 3:
        return None

    A = np.array([r[0] for r in rows])

    def fit(vals: np.ndarray, ok: np.ndarray, slope: float | None = None):
        """A line through the bins that are really there; None if too few are."""
        if int(ok.sum()) < (3 if slope is None else 2):
            return None
        start = float(A[ok].min())
        ok = ok & (A <= start + reach)         # one wall's thickness of it, no more
        if int(ok.sum()) < (3 if slope is None else 2):
            return None
        a, v = A[ok], vals[ok]
        if slope is None:
            slope, offset = np.polyfit(a, v, 1)
        else:
            offset = float(np.mean(v - slope * a))
        rms = float(np.sqrt(np.mean((offset + slope * a - v) ** 2)))
        return float(offset), float(slope * sign), rms, int(ok.sum()), start

    los = np.array([r[1] for r in rows])
    his = np.array([r[2] for r in rows])
    ok_lo = np.array([r[3] for r in rows])
    ok_hi = np.array([r[4] for r in rows])
    borders = {"minus": fit(los, ok_lo), "plus": fit(his, ok_hi)}
    # A border with too little real tissue to give a direction of its own
    # borrows one from the border opposite: the two run parallel, so what the
    # cut takes away is where the border sits, not which way it is going.
    if borders["minus"] is None and borders["plus"] is not None:
        borders["minus"] = fit(los, ok_lo, borders["plus"][1] * sign)
    elif borders["plus"] is None and borders["minus"] is not None:
        borders["plus"] = fit(his, ok_hi, borders["minus"][1] * sign)
    mid = fit(np.array([r[5] for r in rows]), A <= reach)
    if mid is None:
        return None
    return mid[0], mid[1], len(rows), typical, mid[2], borders


def _wall_step(ts: "TileSet", poses: Sequence[Pose], members: Sequence[int],
               rest: Sequence[int], group: Sequence[int], i: int, j: int,
               scale: float, max_band_angle: float, min_leverage: float,
               slide: np.ndarray | None = None):
    """How far this group has to slide for the wall to continue at its far contact.

    Returns ``(why, bits, slide, step)``.  ``why`` is empty when the step is
    usable and otherwise says what stopped it; ``bits`` is what the note should
    report either way; ``slide`` is the unit direction the group is free to
    move along and ``step`` how far along it, in full-resolution pixels.

    The direction is a property of the join as the correlation first saw it, so
    a caller measuring the same group twice passes back the one it was given:
    re-deriving it from a group that has already moved reads a shared strip
    that has slid most of the way out of the wall.
    """
    bits: dict[str, Any] = {}
    width = float(ts.tiles[rest[0]].width)
    frame = mosaic_bounds([poses[k] for k in members],
                          [ts.tiles[k] for k in members], scale)
    _whole, _seen, cut = _media_raster(ts, poses, members, scale, frame)
    mask_rest, seen_rest, _ = _media_raster(ts, poses, rest, scale, frame, cut)
    mask_group, seen_group, _ = _media_raster(ts, poses, group, scale, frame, cut)
    yy, xx = np.mgrid[0:mask_rest.shape[0], 0:mask_rest.shape[1]]

    # The strip the two tiles share is all the correlation ever saw, and
    # the group is free to slide along that strip's own direction.
    ox = (max(poses[i].x, poses[j].x)
          + min(poses[i].x + ts.tiles[i].width, poses[j].x + ts.tiles[j].width)) / 2
    oy = (max(poses[i].y, poses[j].y)
          + min(poses[i].y + ts.tiles[i].height, poses[j].y + ts.tiles[j].height)) / 2
    ox = (ox - frame[0]) * scale
    oy = (oy - frame[1]) * scale
    strip = ((xx >= (max(poses[i].x, poses[j].x) - frame[0]) * scale)
             & (xx <= (min(poses[i].x + ts.tiles[i].width,
                           poses[j].x + ts.tiles[j].width) - frame[0]) * scale)
             & (yy >= (max(poses[i].y, poses[j].y) - frame[1]) * scale)
             & (yy <= (min(poses[i].y + ts.tiles[i].height,
                           poses[j].y + ts.tiles[j].height) - frame[1]) * scale))
    if slide is None:
        sliver = _band_axis(strip & (mask_group | mask_rest), ox, oy, 1e9)
        if sliver is None:
            return "no wall inside the shared strip", bits, None, None
        slide = sliver[1]

    # The second contact: where else the two pieces come close.
    away = np.hypot(xx - ox, yy - oy) > 1.5 * width * scale
    gm, rm = mask_group & away, mask_rest & away
    if int(gm.sum()) < 60 or int(rm.sum()) < 60:
        return "the group touches the mosaic in only one place", bits, None, None
    dist = ndi.distance_transform_edt(~rm)
    ys, xs = np.nonzero(gm)
    k = int(np.argmin(dist[ys, xs]))
    cy, cx = float(ys[k]), float(xs[k])

    longest = 0.75 * width * scale
    seed = _band_axis(gm | rm, cx, cy, longest)
    if seed is None:
        return "too little wall at the second contact", bits, None, None
    # Follow both walls, then turn the frame to lie along them and follow
    # again -- the seed is a chord across a bend, and the offsets it reports
    # are the ones the tilt put there.  The stretch that is followed is set by
    # the wall's own thickness rather than by the size of a tile: a wall stays
    # straight for about as far as it is thick, and past that the fit is
    # reading the next bend rather than extending this one.
    mean_dir, reach, wr, wg = seed[1], longest, None, None
    for it in range(4):
        wr = _wall_track(rm, seen_rest, cx, cy, mean_dir, -1.0, reach, longest)
        wg = _wall_track(gm, seen_group, cx, cy, mean_dir, +1.0, reach, longest)
        if wr is None or wg is None:
            break
        if not it:                          # measured once, at the widest window
            reach = float(np.clip(1.5 * (wr[3] + wg[3]) / 2, 0.1 * width * scale, longest))
            continue
        turn = 0.5 * (wr[1] + wg[1])
        if abs(turn) < 2e-3:
            break
        mean_dir = mean_dir + np.array([-mean_dir[1], mean_dir[0]]) * turn
        mean_dir = mean_dir / max(float(np.linalg.norm(mean_dir)), 1e-12)
    if wr is None or wg is None:
        return "the wall could not be followed into the gap", bits, None, None

    # Both borders of the wall, not the middle of it: the middle moves
    # wherever the wall thickens, and at a branch root it does.  Each border
    # is weighted by how well its two extensions agree in direction, so a
    # border still bending where the other has settled counts for less -- and
    # a border that runs along the edge of its own picture for most of the way
    # is not there at all and has already been dropped.
    steps, weights, angles, used = [], [], [], []
    for side in ("minus", "plus"):
        br, bg = wr[5][side], wg[5][side]
        if br is None or bg is None:
            continue
        used.append(side)
        steps.append(float(bg[0] - br[0]))
        carried = (br[4] + bg[4]) / reach
        weights.append(1.0 / (0.02 + abs(br[1] - bg[1])) / (1.0 + carried))
        angles.append(abs(math.degrees(math.atan(br[1]) - math.atan(bg[1]))))
    if not steps:
        steps = [float(wg[0] - wr[0])]
        weights = [1.0]
        angles = [abs(math.degrees(math.atan(wr[1]) - math.atan(wg[1])))]
    angle = float(np.average(angles, weights=weights))
    wall = 0.5 * (wr[3] + wg[3]) / scale
    bits.update(contact_angle=round(angle, 1), wall_px=round(wall),
                extension_px=round(reach / scale),
                extension_rms_px=[round(wr[4] / scale), round(wg[4] / scale)],
                borders=used or ["the middle of the wall; both borders run off the picture"],
                border_steps_px=[round(v / scale) for v in steps])
    if angle > max_band_angle:
        return (f"the two extensions meet at {angle:.0f} deg, so they are not one wall",
                bits, None, None)
    leverage = float(slide[0] * mean_dir[1] - slide[1] * mean_dir[0])
    bits["leverage"] = round(abs(leverage), 2)
    if abs(leverage) < min_leverage:
        return ("sliding runs along the second wall too, so it cannot settle it",
                bits, None, None)
    gap = float(np.average(steps, weights=weights))
    bits["extension_gap_px"] = round(abs(gap) / scale)
    return "", bits, slide, gap / leverage / scale


def settle_sliver_joins(
    ts: "TileSet",
    poses: list[Pose],
    accept_ncc: float = 0.35,
    scale: float = 0.16,
    sliver_frac: float = 0.3,
    max_band_angle: float = 12.0,
    min_leverage: float = 0.35,
) -> tuple[list[Pose], list[dict[str, Any]]]:
    """
    Re-place a group of tiles that the mosaic is holding by a sliver of wall.

    Two tiles that share only a strip of wall give the correlation almost
    nothing to work with *along* that wall: slide one past the other and the
    same wall still sits on the same wall, so the score barely moves.  What
    decides the placement instead is how much tissue happens to overlap, and
    that pulls the group along the wall until the gap left by a deleted
    snapshot has closed.  On one aortic arch it dropped a twenty-tile lumen
    640 px too low, and every seam still read ncc > 0.7 -- each was measured
    against its own neighbour, and the whole group had moved together.

    The tissue elsewhere knows better.  The group is rigid, so wherever else it
    comes close to the rest of the mosaic the wall has to continue: follow the
    muscle band out of both pieces towards that second contact, extend the
    borders of each into the gap between them, and slide the group along the
    sliver until the two sets of borders meet.  One number, measured where the
    wall crosses the direction of travel instead of running along it.  Both
    pieces are thresholded at the same optical density, both walls are followed
    rather than fitted straight, and the borders are extended no further than
    the wall is thick -- a wall that runs into a branch, a wall read off its
    own histogram, and a wall extended past its next bend will each claim a
    step that is not there.  Moving the group moves the contact too, so the
    measurement is repeated on the new position until little is left to move.

    A group is only moved when it really does hang on joins that share far less
    tissue than this mosaic's typical join, when the second contact really does
    show one wall continuing into another (the two bands agree in direction),
    when sliding can actually fix that contact, and when the move introduces no
    contradiction.  Otherwise the layout is left exactly as it was and the
    reason is returned, because a group free to slide is worth saying out loud.
    """
    poses = list(poses)
    notes: list[dict[str, Any]] = []
    qc = layout_qc(ts, poses, accept_ncc)
    pairs = qc.attrs.get("pairs")
    if pairs is None or pairs.empty:
        return poses, notes
    typical = float(pairs.n_both.median())
    if not np.isfinite(typical) or typical <= 0:
        return poses, notes
    cache: dict = {}

    for isl in sorted({q.island for q in poses if q.placed}):
        members = [i for i, q in enumerate(poses) if q.placed and q.island == isl]
        if len(members) < 6:
            continue
        mset = set(members)
        inside = [r for r in pairs.itertuples()
                  if int(r.i) in mset and int(r.j) in mset and not r.conflict]
        solid = [(int(r.i), int(r.j)) for r in inside if r.n_both >= sliver_frac * typical]
        groups = _components(members, solid)
        if len(groups) < 2:
            continue

        rest = groups[0]
        width = float(ts.tiles[rest[0]].width)

        for group in groups[1:]:
            gset = set(group)
            held_by = [r for r in inside
                       if (int(r.i) in gset) != (int(r.j) in gset)]
            if not held_by:
                continue
            join = max(held_by, key=lambda r: r.n_both)
            i, j = (int(join.i), int(join.j)) if int(join.j) in gset else (int(join.j), int(join.i))
            note = {"island": int(isl), "group": sorted(group), "join": [i, j],
                    "n_both": float(join.n_both), "typical_n_both": typical,
                    "moved": False, "why": ""}

            # A step is measured on the geometry it starts from, and moving the
            # group changes that geometry: the far contact slides along the
            # wall and the wall turns under it.  So measure, move, and measure
            # again, until what is left to move is small against the thickness
            # of the wall being lined up.  On the aortic arch the first pass
            # asked for 727 px and the second for 70 -- taking the first pass
            # as the answer leaves the wall a fifth of its own width out.
            trial, shifted, bits, why, slide = poses, np.zeros(2), {}, "", None
            for k in range(3):
                stop, got, slide, s = _wall_step(ts, trial, members, rest, group, i, j,
                                                 scale, max_band_angle, min_leverage, slide)
                if stop:
                    if not k:
                        why, bits = stop, got   # never got a usable step at all
                    else:
                        bits["stopped"] = stop  # keep the passes that did work
                    break
                bits = got
                shifted = shifted + s * slide
                if float(np.hypot(*shifted)) > width:
                    why = f"the settle wanted {np.hypot(*shifted):.0f} px, further than one tile"
                    break
                trial = [Pose(**q.to_dict()) for q in trial]
                for t in group:
                    trial[t].x += s * slide[0]
                    trial[t].y += s * slide[1]
                if abs(s) < 0.1 * bits["wall_px"]:
                    break
            note.update(bits)
            note["slide_px"] = round(float(np.hypot(*shifted)), 1)
            if why:
                note["why"] = why
                notes.append(note)
                continue

            dx, dy = float(shifted[0]), float(shifted[1])
            ev = _evaluate_shift(ts, poses, group, rest, dx, dy, 0.0, cache, accept_ncc)
            if ev["n_conflicts"]:
                note["why"] = f"the settled position contradicts {ev['n_conflicts']} seam(s)"
                notes.append(note)
                continue
            for t in group:
                poses[t].x += dx
                poses[t].y += dy
            note.update(moved=True, dx=round(dx, 1), dy=round(dy, 1),
                        why=f"held by {join.n_both:.0f} px of shared wall against a typical "
                            f"{typical:.0f}; settled on the wall at the group's other contact")
            notes.append(note)
    return poses, notes


def _least_squares_positions(
    members: Sequence[int],
    edges: Sequence[PairMatch],
    poses: Sequence[Pose],
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Weighted least squares for ``p_j - p_i = t_ij`` with the first tile pinned.

    Solved once for x and y together; the design matrix is the incidence matrix
    of the match graph, so this is a weighted graph-Laplacian solve.
    """
    idx_of = {m: k for k, m in enumerate(members)}
    n = len(members)
    if not edges:
        return np.array([[poses[m].x, poses[m].y] for m in members], dtype=float), None

    rows = len(edges) + 1
    A = np.zeros((rows, n), dtype=np.float64)
    b = np.zeros((rows, 2), dtype=np.float64)
    w = np.ones(rows, dtype=np.float64)
    for r, e in enumerate(edges):
        A[r, idx_of[e.i]] = -1.0
        A[r, idx_of[e.j]] = +1.0
        b[r] = (e.tx, e.ty)
        w[r] = max(e.quality, 1e-3)
    # Pin the first member (gauge fixing); heavy weight keeps it exact.
    A[-1, 0] = 1.0
    b[-1] = (poses[members[0]].x, poses[members[0]].y)
    w[-1] = 1e3

    sol = np.zeros((n, 2))
    for _ in range(6):  # IRLS with a Huber-style weight
        Aw = A * w[:, None]
        bw = b * w[:, None]
        sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        r = np.linalg.norm(A @ sol - b, axis=1)
        scale = max(float(np.median(r[:-1])) if rows > 1 else 1.0, 1.0)
        base = np.array([max(e.quality, 1e-3) for e in edges] + [1e3])
        w = base / np.sqrt(1.0 + (r / (2.0 * scale)) ** 2)
    resid = np.linalg.norm(A[:-1] @ sol - b[:-1], axis=1)
    return sol, resid


def _arrange_islands(
    poses: list[Pose],
    islands: Sequence[Sequence[int]],
    tiles: Sequence["Tile"] | None = None,
    gap_frac: float = 0.15,
) -> None:
    """
    Park the islands side by side so none sits on top of another.

    Unplaceable pieces are laid out to the right of the main mosaic with a gap,
    which is where the user picks them up and drags them into place.  The gap
    matters: overlapping islands would make the quality check compare tiles that
    are merely parked near each other and report phantom conflicts.
    """
    cursor_x = 0.0
    for members in islands:
        if not members:
            continue
        xs = [poses[i].x for i in members]
        ys = [poses[i].y for i in members]
        widths = [tiles[i].width for i in members] if tiles else [0.0] * len(members)
        mnx, mny = min(xs), min(ys)
        span = max(x + w for x, w in zip(xs, widths)) - mnx
        for i in members:
            poses[i].x += cursor_x - mnx
            poses[i].y -= mny
        cursor_x += span * (1.0 + gap_frac) + 1.0


# -- local re-alignment used after a manual drag ------------------------------


def snap_tile(
    ts: TileSet,
    poses: Sequence[Pose],
    index: int,
    params: StitchParams | None = None,
    search_frac: float = 0.22,
    neighbours: Sequence[int] | None = None,
    exclude: Sequence[int] | None = None,
) -> tuple[Pose, dict[str, Any]]:
    """
    Take a hand-placed tile and let the correlation finish the job.

    This is the second half of "drag and drop, then align automatically": the
    human supplies the topology (which piece goes roughly where), which is the
    part software gets wrong; the correlation supplies the last few pixels,
    which is the part humans get wrong.

    Only tiles whose current pose already overlaps *index* are considered, and
    the search is limited to ``search_frac`` of a tile around the manual guess,
    so the human's decision cannot be silently overruled.

    ``exclude`` drops tiles from the vote.  It matters when several tiles are
    being aligned together: a tile must not be settled against neighbours that
    are themselves still in the wrong place, or the group simply agrees with its
    own error.  Moving three tiles 90 px and aligning them as a set recovers only
    a third of the offset without this, and all of it with.
    """
    p = params or ts.params
    me = poses[index]
    ti = ts.tiles[index]
    banned = set(exclude or ()) - {index}

    cands = list(neighbours) if neighbours is not None else [
        k for k in range(len(poses))
        if k != index and poses[k].placed and _overlap_area(poses[k], ts.tiles[k], me, ti) > 0
    ]
    cands = [k for k in cands if k not in banned]
    if not cands:
        reason = ("every overlapping neighbour is also being aligned; "
                  "align them as an island instead") if banned else "no overlapping neighbour"
        return Pose(**{**asdict(me), "placed": True}), {"ok": False, "reason": reason}

    cands.sort(key=lambda k: -_overlap_area(poses[k], ts.tiles[k], me, ti))
    cands = cands[:6]

    scale = min(1.0, p.refine_max_dim / max(ti.height, ti.width))
    win = int(round(search_frac * max(ti.height, ti.width) * scale))

    lvl_me = ts.level_at(index, scale, p)
    if abs(me.theta) > 1e-6:
        lvl_me = _rotate_level(lvl_me, me.theta)

    votes: list[tuple[float, float, float]] = []  # (weight, dx, dy)
    details = []
    for k in cands:
        try:
            lvl_k = ts.level_at(k, scale, p)
            if abs(poses[k].theta) > 1e-6:
                lvl_k = _rotate_level(lvl_k, poses[k].theta)
            pred_tx = (me.x - poses[k].x) * scale
            pred_ty = (me.y - poses[k].y) * scale
            res = _local_align(lvl_k, lvl_me, pred_tx, pred_ty, win, p)
            if res is None:
                continue
            dx = (res["tx"] - pred_tx) / scale
            dy = (res["ty"] - pred_ty) / scale
            votes.append((res["quality"], dx, dy))
            details.append({
                "neighbour": ts.tiles[k].name, "ncc": res["ncc"],
                "peak_ratio": res["peak_ratio"], "dx": dx, "dy": dy,
                "quality": res["quality"],
            })
        except Exception:
            continue

    if not votes:
        return Pose(**{**asdict(me), "placed": True}), {"ok": False, "reason": "no usable overlap"}

    wsum = sum(v[0] for v in votes) or 1.0
    dx = sum(v[0] * v[1] for v in votes) / wsum
    dy = sum(v[0] * v[2] for v in votes) / wsum
    new = Pose(x=me.x + dx, y=me.y + dy, theta=me.theta, island=me.island, placed=True, locked=me.locked)
    return new, {"ok": True, "dx": dx, "dy": dy, "votes": details}


def _local_align(
    fixed: RegLevel, moving: RegLevel, pred_tx: float, pred_ty: float,
    win: int, p: StitchParams,
) -> dict[str, Any] | None:
    fshape = _fast_shape(fixed.shape[0] + moving.shape[0], fixed.shape[1] + moving.shape[1])
    pa = _pack_fft(fixed, fshape, workers=1)
    pb = _pack_fft(moving, fshape, workers=1)
    min_tissue = min(fixed.tissue_px, moving.tissue_px)
    if min_tissue < 200:
        return None
    min_overlap = max(200, int(0.05 * min_tissue))
    ncc, counts = masked_ncc_map(pa, pb, fshape, min_overlap=min_overlap, workers=1)
    fy, fx = fshape
    ty_axis = np.where(np.arange(fy) < fy // 2, np.arange(fy), np.arange(fy) - fy)
    tx_axis = np.where(np.arange(fx) < fx // 2, np.arange(fx), np.arange(fx) - fx)
    allowed = (np.abs(ty_axis - pred_ty)[:, None] <= win) & (np.abs(tx_axis - pred_tx)[None, :] <= win)
    pk = peak_with_confidence(ncc, counts, fshape, allowed, exclusion_px=max(4, win // 5),
                              saturation_overlap=max(3.0 * min_overlap, 0.25 * min_tissue))
    if not pk.get("ok"):
        return None
    overlap_frac = pk["n_overlap"] / max(min_tissue, 1)
    pk["quality"] = _quality(pk["ncc"], pk["peak_ratio"], overlap_frac, p)
    pk["overlap_frac"] = overlap_frac
    return pk


def _overlap_area(pa: Pose, ta: Tile, pb: Pose, tb: Tile) -> float:
    ax0, ay0, ax1, ay1 = pa.x, pa.y, pa.x + ta.width, pa.y + ta.height
    bx0, by0, bx1, by1 = pb.x, pb.y, pb.x + tb.width, pb.y + tb.height
    w = min(ax1, bx1) - max(ax0, bx0)
    h = min(ay1, by1) - max(ay0, by0)
    return float(max(w, 0.0) * max(h, 0.0))


def snap_island(
    ts: "TileSet",
    poses: Sequence[Pose],
    island: int,
    params: StitchParams | None = None,
    search_frac: float = 0.22,
    accept_ncc: float = 0.35,
    merge: bool = True,
) -> tuple[list[Pose], dict[str, Any]]:
    """
    Settle a hand-dropped island against its surroundings as a rigid body.

    Aligning the member tiles one by one would be wrong here: the island's
    internal geometry was already solved and is more trustworthy than any single
    tile's local match, so letting each tile drift independently would take a
    correct group and pull it apart.  Instead every member that touches
    something outside the island votes on one common shift, and the **weighted
    median** of those votes is applied to all of them -- a median, so a single
    member landing on ambiguous tissue cannot drag the group.

    If the settled island then contradicts nothing, its tiles join the island
    they were dropped onto.
    """
    p = params or ts.params
    cur = [Pose(**asdict(q)) for q in poses]
    members = [i for i, q in enumerate(cur) if q.island == island]
    outside = [i for i, q in enumerate(cur) if q.island != island and q.placed]
    if not members:
        return cur, {"ok": False, "reason": "no such island"}
    if not outside:
        return cur, {"ok": False, "reason": "nothing to align against"}

    votes: list[tuple[float, float, float]] = []
    detail = []
    for k in members:
        nb = [o for o in outside
              if _overlap_area(cur[o], ts.tiles[o], cur[k], ts.tiles[k]) > 400]
        if not nb:
            continue
        try:
            _new, info = snap_tile(ts, cur, k, p, search_frac=search_frac, neighbours=nb)
        except Exception:
            continue
        if not info.get("ok"):
            continue
        w = float(sum(v["quality"] for v in info["votes"])) or 1e-3
        votes.append((w, float(info["dx"]), float(info["dy"])))
        detail.append({"tile": ts.tiles[k].name, "dx": info["dx"], "dy": info["dy"],
                       "weight": w, "n_neighbours": len(info["votes"])})
    if not votes:
        return cur, {"ok": False, "reason": "the island does not overlap anything yet"}

    def wmedian(vals: list[tuple[float, float]]) -> float:
        vals = sorted(vals, key=lambda v: v[1])
        total = sum(v[0] for v in vals)
        run = 0.0
        for w, x in vals:
            run += w
            if run >= total / 2:
                return x
        return vals[-1][1]

    dx = wmedian([(w, a) for w, a, _b in votes])
    dy = wmedian([(w, b) for w, _a, b in votes])
    for k in members:
        if cur[k].locked:
            continue
        cur[k].x += dx
        cur[k].y += dy

    info: dict[str, Any] = {"ok": True, "dx": dx, "dy": dy, "n_voters": len(votes),
                            "votes": detail, "merged_into": None}

    if merge:
        cache: dict = {}
        touching: dict[int, float] = {}
        conflicts = 0
        for k in members:
            for o in outside:
                if _overlap_area(cur[o], ts.tiles[o], cur[k], ts.tiles[k]) < 400:
                    continue
                ag = agreement(ts, o, k, cur[k].x - cur[o].x, cur[k].y - cur[o].y,
                               cur[o].theta, cur[k].theta, cache)
                if _contradicts(ag, accept_ncc):
                    conflicts += 1
                elif np.isfinite(ag["ncc"]) and ag["n_both"] > 900 and ag["ncc"] > 0.5:
                    touching[cur[o].island] = touching.get(cur[o].island, 0.0) + ag["n_both"]
        info["n_conflicts"] = conflicts
        if touching and conflicts == 0:
            target = max(touching, key=lambda k2: touching[k2])
            for k in members:
                cur[k].island = int(target)
            info["merged_into"] = int(target)
    return cur, info


def snap_islands(
    ts: TileSet,
    poses: Sequence[Pose],
    params: StitchParams | None = None,
    passes: int = 4,
    search_frac: float = 0.12,
    accept_ncc: float = 0.35,
    tol_px: float = 0.5,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Pose], dict[str, Any]]:
    """
    Settle every island against the others, each one moving as a rigid body.

    This is the pass for after the hand work: pieces dragged roughly into place
    and now wanting a few pixels each rather than a rethink.  Neither existing
    tool does it.  ``snap_all`` nudges tiles one at a time, which takes an
    island whose internal geometry is already solved and pulls it apart -- the
    very thing ``snap_island`` documents as wrong.  ``snap_island`` keeps the
    group rigid but settles one island against everything else, which cannot
    converge when three pieces all want to move at once: each is aligned
    against the others' un-settled positions and the answer depends on which
    was asked first.

    Repeating the rigid alignment is what fixes that.  The largest island holds
    still and the rest move onto it, so the mosaic does not drift bodily and
    there is a fixed frame to converge *to*; an island holding a locked tile
    holds still for the same reason, only more so, since a hand-placed anchor
    is the one thing here not up for revision.  Passes stop as soon as no
    island moves more than ``tol_px``, which is what keeps two pieces from
    trading places with each other forever.

    Island membership is left alone.  Merging is a claim that two pieces are
    one, and it belongs to ``snap_island``, made deliberately, rather than
    happening as a side effect of tidying up.
    """
    p = params or ts.params
    cur = [Pose(**asdict(q)) for q in poses]

    members: dict[int, list[int]] = {}
    for i, q in enumerate(cur):
        if q.placed:
            members.setdefault(int(q.island), []).append(i)
    if len(members) < 2:
        return cur, {"ok": False, "n_islands": len(members),
                     "reason": "only one island -- there is nothing to settle it against"}

    # Something has to stay put, or every island chases the others and the whole
    # mosaic walks off.  A locked tile is the user saying "this one is right".
    anchors = {isl for isl, ms in members.items() if any(cur[k].locked for k in ms)}
    if not anchors:
        anchors = {max(members, key=lambda k: len(members[k]))}
    movable = [isl for isl in members if isl not in anchors]
    if not movable:
        return cur, {"ok": False, "n_islands": len(members),
                     "reason": "every island is locked or anchored"}

    moves: dict[int, list[float]] = {isl: [0.0, 0.0] for isl in movable}
    reasons: dict[int, str] = {}
    n_steps = max(1, passes * len(movable))
    done = 0
    converged = False

    for _pass in range(passes):
        # Settle the best-connected island first: it has the most evidence
        # behind it, so the ones that follow are aligning against a frame that
        # has already improved rather than against a guess.
        order = sorted(movable, key=lambda isl: -sum(
            _overlap_area(cur[k], ts.tiles[k], cur[o], ts.tiles[o])
            for k in members[isl]
            for o in range(len(cur))
            if cur[o].placed and int(cur[o].island) != isl))
        worst = 0.0
        for isl in order:
            done += 1
            if progress:
                progress(done / n_steps, f"settling island {isl}")
            try:
                cur, info = snap_island(ts, cur, isl, p, search_frac=search_frac,
                                        accept_ncc=accept_ncc, merge=False)
            except Exception as exc:            # one bad island must not stop the pass
                reasons[isl] = f"{type(exc).__name__}: {exc}"
                continue
            if not info.get("ok"):
                reasons[isl] = str(info.get("reason", "no votes"))
                continue
            reasons.pop(isl, None)
            dx, dy = float(info["dx"]), float(info["dy"])
            moves[isl][0] += dx
            moves[isl][1] += dy
            worst = max(worst, abs(dx), abs(dy))
        if worst <= tol_px:
            converged = True
            break

    if progress:
        progress(1.0, "islands settled")
    settled = [isl for isl in movable if isl not in reasons]
    reason = ""
    if not settled:
        # Saying "nothing moved" is true and useless.  Much the commonest cause
        # is islands still parked off to the side, where there is nothing to
        # align against and the answer is to drag them in first.
        if any("does not overlap" in v for v in reasons.values()):
            reason = ("no island touches anything yet — drag them roughly into place "
                      "first, then settle")
        else:
            reason = "no island could be aligned: " + "; ".join(sorted(set(reasons.values())))
    return cur, {
        "ok": bool(settled),
        "reason": reason,
        "n_islands": len(members),
        "anchored": sorted(anchors),
        "n_settled": len(settled),
        "converged": converged,
        "passes_used": _pass + 1,
        "moves": {str(isl): {"dx": round(moves[isl][0], 2), "dy": round(moves[isl][1], 2)}
                  for isl in settled},
        "skipped": {str(k): v for k, v in reasons.items()},
    }


def snap_selection(
    ts: TileSet,
    poses: Sequence[Pose],
    indices: Sequence[int],
    params: StitchParams | None = None,
    search_frac: float = 0.22,
    passes: int = 3,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Pose], list[dict[str, Any]]]:
    """
    Align a hand-picked set of tiles against everything that is *not* in it.

    A tile in the set must not be settled against another tile in the set, since
    those are the ones known to be in the wrong place -- otherwise a group
    dragged together simply agrees with its own displacement, and how much of the
    error survives depends on the order of the loop.  Aligning three tiles moved
    90 px this way recovers a third of the offset; excluding the siblings
    recovers all of it.

    Tiles whose only neighbours are siblings therefore cannot move on the first
    pass.  They can on the second: a sibling that has already been settled is a
    good anchor, so the set converges outwards from wherever it touches fixed
    ground.  Anything still without an anchor after the last pass is left where
    the user put it and said so, rather than being nudged by guesswork.
    """
    p = params or ts.params
    cur = [Pose(**asdict(q)) for q in poses]
    pending = {int(i) for i in indices if 0 <= int(i) < len(cur)}
    detail: list[dict[str, Any]] = []
    total = max(len(pending), 1)

    for _round in range(max(1, passes)):
        if not pending:
            break
        progressed = False
        # Tiles with the most fixed ground under them go first; they become the
        # anchors for the rest.
        def anchors(k: int) -> float:
            return sum(_overlap_area(cur[o], ts.tiles[o], cur[k], ts.tiles[k])
                       for o in range(len(cur))
                       if o != k and o not in pending and cur[o].placed)

        for idx in sorted(pending, key=lambda k: -anchors(k)):
            if progress:
                done = total - len(pending)
                progress(done / total, f"aligning {ts.tiles[idx].name}")
            try:
                new, info = snap_tile(ts, cur, idx, p, search_frac=search_frac,
                                      exclude=pending)
            except Exception as exc:
                detail.append({"index": idx, "file": ts.tiles[idx].name,
                               "ok": False, "reason": f"{type(exc).__name__}: {exc}"})
                pending.discard(idx)
                continue
            if info.get("ok"):
                cur[idx] = new
                pending.discard(idx)
                progressed = True
                detail.append({"index": idx, "file": ts.tiles[idx].name, **info})
        if not progressed:
            break

    for idx in sorted(pending):
        detail.append({"index": idx, "file": ts.tiles[idx].name, "ok": False,
                       "reason": "nothing outside the selection overlaps this tile"})
    if progress:
        progress(1.0, "alignment finished")
    return cur, detail


def snap_all(
    ts: TileSet,
    poses: Sequence[Pose],
    params: StitchParams | None = None,
    passes: int = 2,
    search_frac: float = 0.10,
    progress: Callable[[float, str], None] | None = None,
) -> list[Pose]:
    """
    Re-align every non-locked tile against its current neighbours, repeatedly.

    Think of it as relaxation: after the global solve or a batch of manual
    drags, each tile is nudged toward agreement with the tiles it touches.
    Locked (hand-placed and confirmed) tiles never move, so the human's anchors
    hold the whole mosaic in place.
    """
    p = params or ts.params
    cur = [Pose(**asdict(q)) for q in poses]
    total = passes * max(len(cur), 1)
    done = 0
    for _ in range(passes):
        order = sorted(range(len(cur)), key=lambda k: -sum(
            _overlap_area(cur[k], ts.tiles[k], cur[o], ts.tiles[o])
            for o in range(len(cur)) if o != k and cur[o].placed
        ))
        for k in order:
            done += 1
            if progress and done % 3 == 0:
                progress(done / total, f"aligning {ts.tiles[k].name}")
            if cur[k].locked or not cur[k].placed:
                continue
            try:
                new, _info = snap_tile(ts, cur, k, p, search_frac=search_frac)
                cur[k] = new
            except Exception:
                continue
    if progress:
        progress(1.0, "alignment finished")
    return cur


# =============================================================================
# 7.  Mosaic rendering
# =============================================================================


def mosaic_bounds(poses: Sequence[Pose], tiles: Sequence[Tile], scale: float = 1.0
                  ) -> tuple[float, float, int, int]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for p, t in zip(poses, tiles):
        if not p.placed:
            continue
        if abs(p.theta) > 1e-6:
            hw, hh = t.width / 2.0, t.height / 2.0
            c, s = math.cos(math.radians(p.theta)), math.sin(math.radians(p.theta))
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            rot = [(cx * c - cy * s, cx * s + cy * c) for cx, cy in corners]
            w = max(r[0] for r in rot) - min(r[0] for r in rot)
            h = max(r[1] for r in rot) - min(r[1] for r in rot)
        else:
            w, h = float(t.width), float(t.height)
        cx, cy = p.x + t.width / 2.0, p.y + t.height / 2.0
        xs0.append(cx - w / 2.0)
        ys0.append(cy - h / 2.0)
        xs1.append(cx + w / 2.0)
        ys1.append(cy + h / 2.0)
    if not xs0:
        return 0.0, 0.0, 1, 1
    x0, y0 = min(xs0), min(ys0)
    W = int(math.ceil((max(xs1) - x0) * scale)) + 1
    H = int(math.ceil((max(ys1) - y0) * scale)) + 1
    return x0, y0, H, W


def _feather_weight(h: int, w: int, feather: int) -> np.ndarray:
    """Linear ramp from the tile border inwards, for seamless blending."""
    feather = max(1, min(feather, min(h, w) // 2 - 1)) if min(h, w) > 4 else 1
    wy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    wx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    wy = np.clip((wy + 1) / feather, 0.02, 1.0)
    wx = np.clip((wx + 1) / feather, 0.02, 1.0)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def render_mosaic(
    ts: TileSet,
    poses: Sequence[Pose],
    scale: float = 0.25,
    feather_px: int = 40,
    mode: str = "feather",
    indices: Sequence[int] | None = None,
    corrected: bool = True,
    progress: Callable[[float, str], None] | None = None,
    return_coverage: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, dict]:
    """
    Paint the tiles into one image at *scale*.

    ``mode='feather'`` cross-fades overlapping tiles (smooth, standard);
    ``mode='darkest'`` keeps the highest optical density at every pixel, which
    hides focus differences between snapshots at the cost of a slightly harder
    look.
    """
    idxs = list(indices) if indices is not None else [i for i, p in enumerate(poses) if p.placed]
    x0, y0, H, W = mosaic_bounds([poses[i] for i in idxs], [ts.tiles[i] for i in idxs], scale)
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W), dtype=np.float32)
    if mode == "darkest":
        acc[:] = 1.0

    def decoded():
        """
        Tiles at *scale*, a poolful at a time and still in order.

        Re-reading and shrinking the files is nearly all of the render and
        every tile is independent, but the painting below is not: it adds into
        one accumulator, so the reads go wide and the brush stays single.
        """
        k = max(1, min(ts.params.max_workers, 8))
        with ThreadPoolExecutor(max_workers=k) as ex:
            for start in range(0, len(idxs), k):
                chunk = idxs[start:start + k]
                yield from zip(chunk, ex.map(
                    lambda t: ts.full_rgb(t, scale, corrected=corrected), chunk))

    for n, (i, rgb) in enumerate(decoded()):
        if progress:
            progress(n / max(len(idxs), 1), f"rendering {ts.tiles[i].name}")
        p = poses[i]
        h, w = rgb.shape[:2]
        wt = _feather_weight(h, w, max(2, int(feather_px * scale)))
        if abs(p.theta) > 1e-6 and cv2 is not None:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), p.theta, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            rgb = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(1.0, 1.0, 1.0))
            wt = cv2.warpAffine(wt, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0.0)
            h, w = nh, nw
            cx = (p.x + ts.tiles[i].width / 2.0 - x0) * scale
            cy = (p.y + ts.tiles[i].height / 2.0 - y0) * scale
            ox, oy = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
        else:
            ox = int(round((p.x - x0) * scale))
            oy = int(round((p.y - y0) * scale))

        ax0, ay0 = max(0, ox), max(0, oy)
        ax1, ay1 = min(W, ox + w), min(H, oy + h)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        bx0, by0 = ax0 - ox, ay0 - oy
        sub = rgb[by0:by0 + (ay1 - ay0), bx0:bx0 + (ax1 - ax0)]
        sw = wt[by0:by0 + (ay1 - ay0), bx0:bx0 + (ax1 - ax0)]

        if mode == "darkest":
            region = acc[ay0:ay1, ax0:ax1]
            np.minimum(region, sub, out=region)
            wacc[ay0:ay1, ax0:ax1] = np.maximum(wacc[ay0:ay1, ax0:ax1], 1.0)
        else:
            acc[ay0:ay1, ax0:ax1] += sub * sw[..., None]
            wacc[ay0:ay1, ax0:ax1] += sw

    if mode == "darkest":
        out = np.where(wacc[..., None] > 0, acc, 1.0)
    else:
        out = np.where(wacc[..., None] > 1e-6, acc / np.maximum(wacc[..., None], 1e-6), 1.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    if progress:
        progress(1.0, "mosaic rendered")
    if return_coverage:
        info = {"x0": float(x0), "y0": float(y0), "scale": float(scale),
                "height": int(H), "width": int(W), "indices": idxs}
        return out, (wacc > 1e-6), info
    return out


def mosaic_info(ts: TileSet, poses: Sequence[Pose], scale: float,
                indices: Sequence[int] | None = None) -> dict:
    idxs = list(indices) if indices is not None else [i for i, p in enumerate(poses) if p.placed]
    x0, y0, H, W = mosaic_bounds([poses[i] for i in idxs], [ts.tiles[i] for i in idxs], scale)
    return {"x0": float(x0), "y0": float(y0), "scale": float(scale),
            "height": int(H), "width": int(W), "indices": idxs,
            "px_um": (ts.px_um / scale) if ts.px_um else None}


# =============================================================================
# 8.  Persistence
# =============================================================================


def layout_dataframe(ts: TileSet, poses: Sequence[Pose]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "index": t.index, "file": t.name, "path": str(t.path),
            "x_px": p.x, "y_px": p.y, "rotation_deg": p.theta,
            "island": p.island, "placed": bool(p.placed), "locked": bool(p.locked),
            "width": t.width, "height": t.height, "px_um": t.px_um,
        }
        for t, p in zip(ts.tiles, poses)
    ])


def save_layout(path: str | Path, ts: TileSet, poses: Sequence[Pose],
                matches: Sequence[PairMatch] | None = None,
                params: StitchParams | None = None,
                extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "folder": str(ts.folder) if ts.folder else None,
        "params": asdict(params or ts.params),
        "tiles": [t.to_dict() for t in ts.tiles],
        "poses": [p.to_dict() for p in poses],
        "matches": [m.to_dict() for m in (matches or [])],
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, indent=1))
    layout_dataframe(ts, poses).to_csv(path.with_suffix(".csv"), index=False)
    return path


def load_layout(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    data["poses"] = [Pose(**p) for p in data.get("poses", [])]
    data["matches"] = [PairMatch(**m) for m in data.get("matches", [])]
    if data.get("params"):
        known = {f.name for f in StitchParams.__dataclass_fields__.values()}
        data["params"] = StitchParams(**{k: v for k, v in data["params"].items() if k in known})
    return data


# =============================================================================
# 9.  One-call convenience wrapper
# =============================================================================


def auto_stitch(
    folder: str | Path,
    params: StitchParams | None = None,
    files: Sequence[str] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[TileSet, list[Pose], list[PairMatch], dict]:
    """Load, match, refine, solve -- the whole automatic path in one call."""
    p = params or StitchParams()
    ts = TileSet(p)

    def sub(lo: float, hi: float, label: str):
        def f(frac: float, msg: str) -> None:
            if progress:
                progress(lo + (hi - lo) * float(frac), msg or label)
        return f

    ts.load_folder(folder, files=files, progress=lambda f, m: progress(f * 0.25, m) if progress else None)
    matches = match_all_pairs(ts, p, progress=sub(0.25, 0.65, "matching"))
    matches = refine_matches(ts, matches, p, progress=sub(0.65, 0.90, "refining"))
    poses, stats = solve_layout(ts, matches, p)
    if progress:
        progress(1.0, f"{stats['n_islands']} island(s), {stats['n_edges_used']} usable pairs")
    return ts, poses, matches, stats
