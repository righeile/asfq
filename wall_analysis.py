"""
wall_analysis.py -- straighten a vessel wall and quantify what is in it.

Given a stitched (or single-field) trichrome image of a vessel, this module

1.  separates the stains into *concentrations* by colour deconvolution rather
    than guessing from hue,
2.  finds the muscular media and traces its centre line,
3.  unrolls the wall into a straight ribbon: arc length along x, depth through
    the wall along y,
4.  reports how much collagen there is **per unit length of wall**, cumulatively
    along the vessel, and how much blue staining sits *inside the muscle layer*
    as opposed to in the adventitia around it.

Why concentrations rather than colours
--------------------------------------
Transmitted light through a stained section follows Beer-Lambert: optical
density is linear in the amount of dye, RGB intensity is not.  A hue-and-
saturation score, which is what the earlier pipeline used, mixes "how much dye"
with "how dark the field was", so two sections photographed at different lamp
settings are not comparable and, worse, a thick pale region and a thin dark one
score the same.  Deconvolving OD onto stain vectors gives a quantity
proportional to dye per unit area, which can legitimately be integrated over
distance -- and integrating it is exactly what "intensity per length of wall"
requires.

Main entry point: :func:`analyze_wall`.
"""

from __future__ import annotations

import inspect
import math
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi, signal
from scipy.interpolate import splprep, splev

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    ``ndi.gaussian_filter``, by way of OpenCV when it is installed.

    Same kernel, same reflected edges, same answer to a part in a hundred
    thousand -- and a section-sized smooth at the lesion detector's 25 um
    costs 75 ms instead of 1.3 s, which is a quarter of an entire analysis.
    """
    a = np.ascontiguousarray(img, dtype=np.float32)
    if cv2 is None or sigma <= 0:
        return ndi.gaussian_filter(a, sigma)
    # scipy truncates at four sigma; say so rather than take OpenCV's own
    # default, which is a tap wider and moves the answer by more.
    k = 2 * int(4.0 * float(sigma) + 0.5) + 1
    return cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


def square_morph(mask: np.ndarray, size: int, op: str) -> np.ndarray:
    """
    ``ndi.binary_closing`` / ``binary_opening`` / ``binary_dilation`` with a
    square, through OpenCV.

    The same answer, pixel for pixel, at a fraction of the time: closing a
    section-sized mask with the 6 um square the ring test uses takes 16 ms
    here against 2.4 s in scipy, which was an eighth of a whole analysis.

    Two conventions have to be copied for that to be true.  scipy dilates with
    the *reflected* structure, so its origin moves by one when the square has
    an even side; and it treats everything outside the image as background, so
    the pad is zeroed again between the two passes.
    """
    size = max(1, int(size))
    if cv2 is None or size < 2:
        f = {"close": ndi.binary_closing, "open": ndi.binary_opening,
             "dilate": ndi.binary_dilation}[op]
        return f(mask, np.ones((size, size), bool))
    r = size // 2
    k = np.ones((size, size), np.uint8)
    a_ero, a_dil = (r, r), (size - 1 - r, size - 1 - r)
    if op == "dilate":
        # Nothing outside can switch a pixel on, so this one needs no pad.
        return cv2.dilate(np.ascontiguousarray(mask).view(np.uint8), k,
                          anchor=a_dil).astype(bool)
    pad = cv2.copyMakeBorder(np.ascontiguousarray(mask).view(np.uint8),
                             size, size, size, size, cv2.BORDER_CONSTANT, value=0)
    first, second = ((cv2.dilate, a_dil), (cv2.erode, a_ero)) if op == "close" \
        else ((cv2.erode, a_ero), (cv2.dilate, a_dil))
    out = first[0](pad, k, anchor=first[1])
    out[:size] = 0
    out[-size:] = 0
    out[:, :size] = 0
    out[:, -size:] = 0
    return second[0](out, k, anchor=second[1])[size:-size, size:-size].astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    ``ndi.binary_fill_holes``, by way of OpenCV's scanline flood fill.

    scipy grows the background inwards from the border one dilation at a time,
    which on a section-sized mask is a second each; the same question asked as
    "which zeros can the outside reach" is answered in one pass.
    """
    if cv2 is None:
        return ndi.binary_fill_holes(mask)
    m = np.ascontiguousarray(mask).view(np.uint8)
    h, w = m.shape
    # One ring of background around the image, so the flood starts outside it
    # and every border pixel is reachable -- which is what scipy assumes.
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    cv2.floodFill(pad, None, (0, 0), 1, flags=4)
    return np.asarray(mask, bool) | (pad[1:-1, 1:-1] == 0)


def edt(mask: np.ndarray) -> np.ndarray:
    """
    ``ndi.distance_transform_edt``, through OpenCV's exact one.

    Not an approximation: with DIST_MASK_PRECISE the two agree to the last
    bit of a float32 on a real section, and it is twenty times faster.
    """
    if cv2 is None:
        return ndi.distance_transform_edt(mask)
    return cv2.distanceTransform(np.ascontiguousarray(mask).view(np.uint8),
                                 cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


# -- scikit-image compatibility ----------------------------------------------
# 0.26 deprecated `min_size` / `area_threshold` in favour of a keyword-only
# `max_size`, and the two do *not* mean the same thing: `min_size=n` dropped
# components smaller than n, `max_size=n` drops components of n or smaller.
# Passing the old number straight through to the new keyword would therefore
# shift every size cut-off in this file by one pixel -- and the deprecation
# shim inside 0.26 does exactly that, so these numbers already moved silently
# when skimage was upgraded.  One pixel is nothing against thresholds in the
# hundreds, but it is a change nobody asked for, so translate properly and
# keep the original meaning on both old and new skimage.
_SK_MAX_SIZE = "max_size" in inspect.signature(remove_small_objects).parameters


def drop_small_objects(mask: np.ndarray, min_size) -> np.ndarray:
    """
    Remove connected components smaller than *min_size* pixels.

    OpenCV labels and measures in one pass where skimage labels, counts and
    then looks up; four-connected either way, so the components and their
    areas are the same ones.
    """
    n = int(min_size)
    if n <= 1:
        return mask
    if cv2 is None:
        if _SK_MAX_SIZE:
            return remove_small_objects(mask, max_size=n - 1)
        return remove_small_objects(mask, n)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask).view(np.uint8), connectivity=4)
    keep = stats[:, cv2.CC_STAT_AREA] >= n
    keep[0] = False  # label 0 is the background, however big it is
    return keep[lab]


def fill_small_holes(mask: np.ndarray, area_threshold) -> np.ndarray:
    """Fill holes smaller than *area_threshold* pixels."""
    n = int(area_threshold)
    if n <= 1:
        return mask
    if cv2 is None:
        if _SK_MAX_SIZE:
            return remove_small_holes(mask, max_size=n - 1)
        return remove_small_holes(mask, n)
    # What skimage does: a hole is a small component of the complement.
    return ~drop_small_objects(~np.asarray(mask, bool), n)


# =============================================================================
# 1.  Stain separation
# =============================================================================

# Optical-density vectors for Masson's trichrome, in RGB.  These are starting
# values: `estimate_stains` refines them from the section itself, because the
# exact shade of aniline blue depends on differentiation time and how long the
# slide has sat in the drawer.
DEFAULT_STAINS = {
    "muscle": np.array([0.30, 0.75, 0.58]),    # Biebrich scarlet / acid fuchsin
    "collagen": np.array([0.72, 0.55, 0.18]),  # aniline blue
    "nuclei": np.array([0.55, 0.60, 0.58]),    # Weigert's haematoxylin
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def rgb_to_od(rgb01: np.ndarray, white: np.ndarray | float = 1.0) -> np.ndarray:
    """``OD = -log10(I / I_white)``, clipped away from zero transmission."""
    w = np.asarray(white, dtype=np.float32).reshape(1, 1, -1) if np.ndim(white) else float(white)
    ratio = np.clip(rgb01.astype(np.float32) / np.maximum(w, 1e-4), 1e-4, 1.0)
    return -np.log10(ratio)


def od_to_rgb(od: np.ndarray, white: np.ndarray | float = 1.0) -> np.ndarray:
    """The inverse of :func:`rgb_to_od`, clipped to a displayable image."""
    w = np.asarray(white, dtype=np.float32).reshape(1, 1, -1) if np.ndim(white) else float(white)
    return np.clip(np.power(10.0, -np.asarray(od, dtype=np.float32)) * w, 0.0, 1.0)


def restain(rgb01: np.ndarray, stain_matrix: np.ndarray,
            gains: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    The same section with one stain turned down -- **for looking at only**.

    Scoring fibrosis by eye means seeing the blue, and in a section that is
    mostly muscle the red drowns it.  Turning the red down here is not a
    contrast adjustment on the picture: the image is separated into its stains,
    the muscle concentration is scaled, and the section is put back together
    from the result.  Blue is left exactly as it was, so what you are judging
    is the collagen at its true density against less red -- not collagen that
    has been made to look stronger.

    Nothing measured passes through this.  Every number in every table comes
    from the concentrations before any gain is applied; this touches the
    picture on its way to the screen and nothing else.  A display that can be
    adjusted must not be a display that can change an answer.
    """
    g = np.asarray(gains, dtype=np.float32).reshape(1, 1, -1)
    if np.allclose(g, 1.0):
        return np.clip(rgb01, 0, 1)
    M = np.stack([_unit(r) for r in stain_matrix])
    od = rgb_to_od(np.clip(rgb01, 0, 1))
    conc = np.clip(od.reshape(-1, 3) @ np.linalg.pinv(M), 0.0, None)
    conc = conc.reshape(od.shape) * g
    return od_to_rgb(conc.reshape(-1, 3) @ M).reshape(od.shape)


def estimate_stains(
    od: np.ndarray,
    tissue: np.ndarray | None = None,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Read the two dominant stain vectors off the section (Macenko et al., 2009).

    Pixels with appreciable OD are projected onto the plane spanned by the two
    largest eigenvectors of their *covariance*; in that plane every pixel is a
    mixture of the two dyes, so the extremes of the angular distribution are the
    pure-dye directions.  Robust percentiles rather than true extremes keep a
    speck of dirt from defining a stain.

    Using the covariance -- not the raw SVD of the OD matrix -- matters.  Every
    optical density is positive, so the first singular vector of the uncentred
    data is just the mean colour and the angular spread around it collapses to
    nothing; the two "stain vectors" then come back nearly identical and the
    deconvolution is singular.

    Returns a ``(3, 3)`` matrix of unit OD vectors, ``[muscle, collagen,
    residual]``.  The residual row is the cross product, which absorbs nuclei
    and anything else and so keeps them out of the two quantified channels.
    """
    flat = od.reshape(-1, 3).astype(np.float64)
    if tissue is not None:
        flat = flat[tissue.reshape(-1)]
    flat = flat[flat.sum(axis=1) > beta]
    fallback = np.stack([_unit(DEFAULT_STAINS["muscle"]),
                         _unit(DEFAULT_STAINS["collagen"]),
                         _unit(DEFAULT_STAINS["nuclei"])])
    if flat.shape[0] < 500:
        return fallback
    if flat.shape[0] > 300_000:
        sel = np.random.default_rng(0).choice(flat.shape[0], 300_000, replace=False)
        flat = flat[sel]

    try:
        _w, vecs = np.linalg.eigh(np.cov(flat.T))
    except np.linalg.LinAlgError:
        return fallback
    plane = vecs[:, 1:3]  # two largest eigenvectors
    proj = flat @ plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    a_lo, a_hi = np.percentile(angles, [alpha, 100.0 - alpha])
    v1 = _unit(plane @ np.array([math.cos(a_lo), math.sin(a_lo)]))
    v2 = _unit(plane @ np.array([math.cos(a_hi), math.sin(a_hi)]))
    v1, v2 = np.abs(v1), np.abs(v2)
    if float(np.linalg.norm(np.cross(v1, v2))) < 0.08:
        return fallback

    # Which is which: aniline blue absorbs red light, so the collagen vector has
    # the larger red-channel OD.  (A blue-looking dye is one that removes red.)
    muscle, collagen = (v1, v2) if v1[0] < v2[0] else (v2, v1)
    residual = np.abs(_unit(np.cross(muscle, collagen)))
    return np.stack([muscle, collagen, residual])


def separate_stains(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Concentrations of each stain, ``(H, W, 3)``, non-negative."""
    M = np.stack([_unit(r) for r in stain_matrix])
    conc = od.reshape(-1, 3) @ np.linalg.pinv(M)
    return np.clip(conc, 0.0, None).reshape(od.shape).astype(np.float32)


# =============================================================================
# 2.  Geometry: tissue, media, centre line
# =============================================================================


@dataclass
class WallParams:
    """Knobs for the wall analysis; all lengths in microns unless stated."""

    analysis_px_um: float = 0.5       # working resolution
    tissue_od: float | None = None     # None = automatic (triangle threshold)
    media_quantile: float = 0.55       # of the muscle-concentration range
    min_media_area_um2: float = 20_000.0
    min_lumen_um2: float = 20_000.0    # a hole smaller than this is not a vessel
    min_wall_um: float = 4.0           # nor is one ringed by less muscle than this
    keep_all_vessels: bool = False     # keep every wall component, not just the main one
    smooth_centerline_um: float = 100.0
    arc_step_um: float = 1.0           # sampling step along the wall
    recentre_passes: int = 2           # move the sampling curve onto the mid-wall line
    recentre_smooth_um: float = 20.0   # smoothing applied to the re-centred curve
    max_depth_um: float = 260.0        # half-width of the sampled band
    depth_step_um: float = 0.5
    boundary_smooth_um: float = 30.0
    boundary_outlier_um: float = 25.0
    max_thickness_factor: float = 4.0  # beyond this x the median, it is not a wall
    min_thickness_factor: float = 0.35  # below this x the median, it is not a wall either
    recover_reach_um: float = 60.0     # how far to look for muscle when the curve slips off
    fragment_reach_um: float = 25.0    # how far past the media edge to look for muscle
    # Scale that separates a dropout from the lamellar banding, ~2.5 lamellae.
    variation_scale_um: float = 12.0
    # A lamella must be this much of the wall's mean muscle level to count.
    lamella_prominence: float = 0.10
    exclude_non_wall: bool = True      # keep branch junctions out of the headline totals
    branch_min_limb_um: float = 80.0   # a skeleton limb shorter than this is not a branch
    n_depth_relative: int = 260
    relative_lo: float = -0.35      # how far past the lumen the rectangular view reaches
    relative_hi: float = 1.55       # and past the media/adventitia junction
    blue_threshold: float | None = None  # a number forces the absolute rule
    blue_rule: str = "dominance"       # "dominance" (collagen > muscle) or "absolute"
    # Where the media ends: "muscle-end" (the muscle runs out) or
    # "collagen-crossing" (collagen overtakes muscle).  The default is the
    # muscle because the other one ends the media in the same quantity
    # fibrosis is measured in, so a fibrotic stretch shortens its own wall.
    # See `find_wall_edges`.
    media_edge_rule: str = "muscle-end"
    muscle_end_fraction: float = 0.25  # of this section's own muscle level
    exclude_border_um: float = 0.0
    close_ring: bool = True

    def copy_with(self, **kw) -> "WallParams":
        d = asdict(self)
        d.update({k: v for k, v in kw.items() if k in d and v is not None})
        return WallParams(**d)


def auto_tissue_threshold(od_sum: np.ndarray, lo: float = 0.10, hi: float = 0.9) -> float:
    try:
        from skimage.filters import threshold_triangle
        thr = float(threshold_triangle(od_sum))
    except Exception:
        thr = float(np.percentile(od_sum, 60))
    return float(np.clip(thr, lo, hi))


def tissue_mask(od_sum: np.ndarray, threshold: float | None = None,
                min_area_px: int = 400) -> np.ndarray:
    thr = auto_tissue_threshold(od_sum) if threshold is None else float(threshold)
    m = od_sum > thr
    m = square_morph(m, 5, "close")
    m = drop_small_objects(m, min_area_px)
    m = fill_small_holes(m, min_area_px)
    return m.astype(bool)


def media_mask(
    muscle: np.ndarray,
    tissue: np.ndarray,
    px_um: float,
    p: WallParams,
    roi: np.ndarray | None = None,
) -> np.ndarray:
    """
    Isolate the muscular media -- the dark red band -- from everything else.

    Perivascular fat also stains pink, and in an aortic arch there is a lot of
    it, so a threshold on redness alone is not enough.  Two things separate the
    media: its stain concentration is far higher than fat's (dense smooth muscle
    against mostly-empty adipocytes), and it forms one long connected band
    rather than a blob.  Components are therefore scored on area times
    elongation times mean concentration, and the winner plus anything sharing
    its band is kept.
    """
    vals = muscle[tissue] if tissue.any() else muscle.ravel()
    if vals.size < 100:
        raise ValueError("Not enough tissue to find a vessel wall.")
    try:
        base = float(threshold_otsu(vals))
    except Exception:
        base = float(np.percentile(vals, 70))
    hi = float(np.percentile(vals, 99.5))
    thr = base + (hi - base) * (p.media_quantile - 0.5) * 1.2
    thr = float(np.clip(thr, base * 0.6, hi))

    m = (muscle > thr) & tissue
    if roi is not None:
        m &= roi
    m = square_morph(m, 3, "open")
    min_px = max(200, int(p.min_media_area_um2 / (px_um ** 2)))
    m = drop_small_objects(m, min_px)
    if not m.any():
        m = (muscle > base) & tissue
        m = drop_small_objects(m, min_px // 2)
    if not m.any():
        raise ValueError("Could not find a muscular wall band.")

    lab, n = ndi.label(m)
    if n > 1:
        objs = ndi.find_objects(lab)
        scores = []
        for k, sl in enumerate(objs, start=1):
            comp = lab[sl] == k
            area = float(comp.sum())
            if area < min_px:
                scores.append(-np.inf)
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            elong = max(h, w) / max(1.0, math.sqrt(area))
            scores.append(area * math.log1p(elong) * float(muscle[sl][comp].mean()))
        best = int(np.argmax(scores)) + 1
        keep = {best}
        # Keep other components of comparable merit: a ring cut by a fold comes
        # back as two arcs and both are wall.
        top = scores[best - 1]
        cut = 0.0 if p.keep_all_vessels else 0.25 * top
        for k, s in enumerate(scores, start=1):
            if np.isfinite(s) and s > cut:
                keep.add(k)
        m = np.isin(lab, list(keep))
    m = fill_small_holes(m, min_px)
    return m.astype(bool)


def _skeleton_graph(skel: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    coords = np.argwhere(skel)
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    nbrs: list[list[int]] = [[] for _ in range(len(coords))]
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = index.get((int(r) + dr, int(c) + dc))
                if j is not None:
                    nbrs[i].append(j)
    return coords, nbrs


def _dijkstra(nbrs: list[list[int]], coords: np.ndarray, start: int
              ) -> tuple[np.ndarray, np.ndarray]:
    import heapq

    n = len(coords)
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        for j in nbrs[i]:
            w = float(np.hypot(*(coords[j] - coords[i])))
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))
    return dist, prev


def _branch_prune(skel: np.ndarray, min_branch_px: int, max_rounds: int = 40) -> np.ndarray:
    """
    Remove skeleton side branches shorter than *min_branch_px*.

    Stripping every endpoint pixel repeatedly (the obvious approach) shortens
    real branches as well as spurs, and on a ring it never terminates cleanly.
    Here each endpoint is walked inward to the first junction; the branch is
    deleted only if the whole thing is short.  Long branches are left intact.
    """
    for _ in range(max_rounds):
        coords, nbrs = _skeleton_graph(skel)
        if len(coords) == 0:
            return skel
        deg = np.array([len(v) for v in nbrs])
        ends = np.where(deg == 1)[0]
        if len(ends) == 0:
            return skel
        remove: set[int] = set()
        for e in ends:
            path = [int(e)]
            prev, cur = -1, int(e)
            while len(path) <= min_branch_px:
                nxt = [x for x in nbrs[cur] if x != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            if len(path) <= min_branch_px and deg[path[-1]] >= 3:
                remove.update(path[:-1])
        if not remove:
            return skel
        keep = np.array([i for i in range(len(coords)) if i not in remove])
        if len(keep) < 8:
            return skel
        new = np.zeros_like(skel)
        c = coords[keep]
        new[c[:, 0], c[:, 1]] = True
        skel = new
    return skel


def _band_centre_offsets(
    dt: np.ndarray, pts: np.ndarray, nx: np.ndarray, ny: np.ndarray,
    reach_px: float, step_px: float = 0.5,
) -> np.ndarray:
    """
    How far out the middle of the wall lies, separately at every point.

    Walking outwards from the lumen edge, the distance transform of the media
    rises to a maximum halfway through the band and falls away again.  That
    maximum is the medial axis, and it is the middle of the wall *here*.

    The alternative -- one high percentile of the same transform for the whole
    section -- ties every point's offset to the thickest muscle anywhere in the
    mask.  On a section with a branch that is the junction, where two walls have
    merged: on the aortic arch it gave 60 um against a wall whose own
    half-thickness is 28, so the starting curve was put a full wall thickness
    too far out and 93 % of positions began outside their own media.

    Only the band a point actually borders may claim it.  The walk stops at the
    first sample outside the mask, so a second wall further along the normal --
    the far side of a branch, the neighbouring vessel in the same section --
    cannot capture the point.  Where the normal never crosses muscle at all the
    offset is NaN, for the caller to fill from its neighbours.
    """
    n_steps = max(2, int(round(reach_px / step_px)) + 1)
    t = (np.arange(n_steps, dtype=np.float64) * step_px)[None, :]
    rows = pts[:, 1][:, None] + ny[:, None] * t
    cols = pts[:, 0][:, None] + nx[:, None] * t
    prof = ndi.map_coordinates(dt, [rows, cols], order=1, mode="nearest")

    outside = prof < 0.25
    outside[:, 0] = False
    first_out = np.where(outside.any(axis=1), outside.argmax(axis=1), n_steps)
    inside = np.arange(n_steps)[None, :] < first_out[:, None]
    prof_in = np.where(inside, prof, -1.0)
    off = prof_in.argmax(axis=1) * step_px
    ok = (first_out > 1) & (prof_in.max(axis=1) > 0.5)
    return np.where(ok, off, np.nan)


def _outline_and_normals(lumen: np.ndarray, min_points: int = 16):
    """
    The outline of one lumen, smoothed, with outward normals.

    A pixel-stepped outline has a sawtooth normal and a sawtooth normal makes a
    sawtooth offset, so the curve is filtered before it is differentiated.  The
    normals are then checked against the lumen itself and flipped if they point
    inwards, because which way is out depends on the direction the contour came
    back in and not on anything about the vessel.
    """
    from skimage.measure import find_contours

    conts = find_contours(lumen.astype(float), 0.5)
    if not conts:
        raise ValueError("The lumen has no outline.")
    cont = max(conts, key=len)
    xy = np.column_stack([cont[:, 1], cont[:, 0]]).astype(np.float64)
    if len(xy) > 4 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if len(xy) < min_points:
        raise ValueError("The lumen outline is too short to analyse.")

    k = max(3, int(len(xy) / 200) | 1)
    sm = np.column_stack([
        ndi.uniform_filter1d(xy[:, 0], k, mode="wrap"),
        ndi.uniform_filter1d(xy[:, 1], k, mode="wrap"),
    ])
    dx = np.gradient(np.r_[sm[-1, 0], sm[:, 0], sm[0, 0]])[1:-1]
    dy = np.gradient(np.r_[sm[-1, 1], sm[:, 1], sm[0, 1]])[1:-1]
    mag = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / mag, dx / mag

    probe = 2.0
    px_i = np.clip(np.round(sm[:, 1] + ny * probe).astype(int), 0, lumen.shape[0] - 1)
    px_j = np.clip(np.round(sm[:, 0] + nx * probe).astype(int), 0, lumen.shape[1] - 1)
    if lumen[px_i, px_j].mean() > 0.5:
        nx, ny = -nx, -ny
    return sm, nx, ny, k


def wall_around_lumen_um(mask: np.ndarray, lumen: np.ndarray, px_um: float) -> float:
    """
    How thick the muscle is around one hole -- the test of whether it is a vessel.

    Every hole in the media mask is enclosed by media, by construction, so
    "is it surrounded by wall" says nothing.  How *much* wall does: a vessel is
    ringed by a media of some real thickness, while a hole left by a segmentation
    artefact is ringed by a sliver a pixel or two wide.  Measured the same way
    the trace is placed -- outwards along each normal to the ridge of the
    distance transform -- and reported as a median so a fold or a branch at one
    point cannot carry the answer.

    Returns 0.0 for a hole whose outline is too short to measure, which is
    itself a reason not to call it a vessel.
    """
    try:
        sm, nx, ny, _k = _outline_and_normals(lumen)
    except ValueError:
        return 0.0
    dt = edt(mask)
    reach = min(400.0, float(np.percentile(dt[mask], 99.9)) * 1.5 + 4.0) if mask.any() else 4.0
    off = _band_centre_offsets(dt, sm, nx, ny, reach)
    good = off[np.isfinite(off)]
    return 2.0 * float(np.median(good)) * px_um if good.size else 0.0


def _lumen_contour_centerline(
    mask: np.ndarray, lumen: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    A mid-wall curve for a closed vessel, built from the lumen outline.

    The outline of the lumen is a closed curve that is already in order, which
    is the property the skeleton of a real wall band does not have: dense smooth
    muscle skeletonises into a mesh of small loops and junctions (here, 482
    degree-3 nodes around one aorta), and no longest-path rule recovers the
    circumference from that.  The outline is then pushed outwards onto the
    middle of the media, point by point, so sampling lines start in the wall
    rather than at its edge or past it.
    """
    sm, nx, ny, k = _outline_and_normals(lumen)
    dt = edt(mask)

    reach = min(400.0, float(np.percentile(dt[mask], 99.9)) * 1.5 + 4.0) if mask.any() else 4.0
    off = _band_centre_offsets(dt, sm, nx, ny, reach)
    if not np.isfinite(off).any():
        # No normal crossed muscle anywhere: fall back to the old single number
        # rather than fail, and let the re-centring pass sort it out.
        off = np.full(len(sm), float(np.percentile(dt[mask], 97.0)) if mask.any() else 1.0)
    else:
        off = _fill_nan_1d(off)
        med = float(np.median(off))
        # A point claiming several times the vessel's own half-thickness is at a
        # junction, where the medial axis leaves the wall and runs off into the
        # blob two merged walls make.  Following it there ties the curve into a
        # loop.  The wall's own scale is the honest cap; the junction still gets
        # flagged, by the test that is actually about topology.
        off = np.clip(off, 0.0, 2.0 * med + 2.0)
        # The offset should vary as slowly as the wall thickness does, so a
        # stray reading is a mistake rather than a feature.  Same window as the
        # outline itself was smoothed with.
        off = _despike(off, max(3, k), 0.75 * med + 1.0, max(1.0, k / 4.0), True)

    out = np.column_stack([sm[:, 0] + nx * off, sm[:, 1] + ny * off])
    return out, float(np.median(off))


def coverage_from_mosaic(rgb01: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """
    Which pixels the mosaic actually has data for.

    ``render_mosaic`` paints untouched canvas at exactly 1.0 in all three
    channels; illuminated blank slide lands just under it.  The distinction
    matters because "no tile here" and "no tissue here" are completely different
    statements and only one of them is evidence about the specimen.
    """
    return ~np.all(rgb01 >= 1.0 - tol, axis=2)


# NOTE.  There is deliberately no gap-bridging here.  Two attempts were made and
# both were removed:  dilating the wall band into uncovered territory swallowed
# the empty corners of the mosaic and reported a 37 mm vessel with five lumina,
# and pairing up the band's free ends never fired on a real gap -- with eight
# fields removed the media mask has 36 free ends, of which one is anywhere near
# the missing data, because most are mask spurs rather than the two sides of a
# hole.  What is kept instead is the coverage mask, which is enough: stretches
# with no image behind them are marked and excluded rather than measured as
# thin wall, and the per-length readouts are intensive, so a missing field costs
# length and shifts the density by a few per cent rather than corrupting it.


def lumen_candidates(
    mask: np.ndarray, px_um: float, min_lumen_um2: float = 20_000.0,
    min_wall_um: float = 4.0,
) -> list[dict]:
    """
    Every hole the wall band encloses, largest first, with the evidence for and
    against each being a vessel.

    Two tests, and both are needed.  **Size**, because a lumen smaller than a
    cell cluster is not a vessel anyone sectioned on purpose.  And **the wall
    around it**, because every hole here is enclosed by media by construction --
    that is what makes it a hole -- so being surrounded says nothing, while
    being surrounded by 40 µm of muscle rather than a two-pixel sliver says a
    great deal.  Without the second test a fold in the band, or a ring of
    adventitial staining that happens to close, counts as a vessel and gets a
    full analysis run on it.

    Nothing is dropped silently: rejected holes come back with ``vessel=False``
    and the numbers that decided it, so a vessel that should have been measured
    can be seen not to have been.

    Whether a hole exists at all is decided on the mask closed by a few
    microns, not the raw one.  The muscle threshold is a per-pixel decision,
    and a ring that is media all the way round can still lose a pixel-scale
    sliver of it to noise or a slightly pale patch -- on one thoracic aorta
    the true lumen did not register as enclosed at all until the closing
    reached 3 um, even though nothing in the picture looks broken at that
    scale.  A gap that thin is not a real opening in the wall, but it is
    enough for `binary_fill_holes` to see no hole there rather than a thin
    one, which sends the whole vessel down the open-arc path instead of
    tracing it as the ring it is.  The thinness test below still runs on the
    mask as segmented, unclosed, so a gap wide enough to matter is rejected
    on its own merits exactly as before -- closing only lets a hole reach
    that test at all.
    """
    close_px = max(1, int(round(6.0 / max(px_um, 1e-6))))
    closed_for_topology = square_morph(mask, close_px, "close")
    filled = fill_holes(closed_for_topology)
    holes = filled & ~mask
    lab, n = ndi.label(holes)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    out: list[dict] = []
    for k in np.argsort(-sizes):
        if sizes[k] == 0:
            continue
        area = float(sizes[k]) * px_um ** 2
        if area < min_lumen_um2 * 0.25:
            break        # sorted by size: nothing after this is worth weighing
        hole = lab == k
        wall = wall_around_lumen_um(mask, hole, px_um) if area >= min_lumen_um2 else 0.0
        ys, xs = np.nonzero(hole)
        out.append({
            "mask": hole,
            "area_um2": area,
            "wall_um": wall,
            "x_px": float(xs.mean()),
            "y_px": float(ys.mean()),
            "vessel": bool(area >= min_lumen_um2 and wall >= min_wall_um),
            "why": ("" if area >= min_lumen_um2 and wall >= min_wall_um
                    else (f"lumen {area:.0f} um2 < {min_lumen_um2:.0f}"
                          if area < min_lumen_um2
                          else f"wall around it {wall:.1f} um < {min_wall_um:.1f}")),
        })
    for rank, c in enumerate([c for c in out if c["vessel"]]):
        c["rank"] = rank
    return out


def enclosed_lumina(mask: np.ndarray, min_hole_px: int = 400,
                    px_um: float = 1.0, min_wall_um: float = 4.0) -> list[np.ndarray]:
    """The holes that are vessels, largest first -- see :func:`lumen_candidates`."""
    cands = lumen_candidates(mask, px_um, min_hole_px * px_um ** 2, min_wall_um)
    return [c["mask"] for c in cands if c["vessel"]]


def wall_pieces(
    mask: np.ndarray,
    lumina: Sequence[np.ndarray] = (),
    px_um: float = 1.0,
    min_area_um2: float = 20_000.0,
    min_aspect: float = 4.0,
) -> list[dict]:
    """
    The separate pieces of the wall band that enclose no lumen, largest first,
    with the evidence for and against each being a vessel.

    A vessel photographed whole encloses its lumen and is found by that hole
    (:func:`lumen_candidates`).  One that does not -- cut open, incompletely
    stitched, or a ring whose band simply never closed at the muscle threshold
    -- is an open arc, and an arc has no hole to be found by.  So the arcs are
    enumerated by the only thing left: they are the connected pieces of the
    band.  Without this the whole band is skeletonised as one and the longest
    path wins, so a section holding two vessels reports one -- and not
    necessarily the larger, since that walk starts from whichever skeleton
    endpoint the raster reached first.  One real aortic arch section, two rings
    neither of which closed, was measured as a single 2.9 mm wall: the smaller
    vessel, with the larger one silently dropped.

    Two tests, matching the two in :func:`lumen_candidates`.  **Size**, the same
    threshold the band itself was built with.  And **shape**, because a piece of
    wall is long compared with its thickness while the fat and the patches of
    dense stain that survive the band scoring are blobs -- a disc scores 1.8
    here whatever its size, an arc of aorta 40 to 60.

    Length comes from the area and the mean distance to the edge rather than
    from a skeleton: a band of area *A* and width *w* is *A/w* long, and
    ``4 * mean(edt)`` is *w* for any shape much longer than it is thick.  On a
    synthetic annulus of known size that is within 3 %, which is far closer
    than deciding "wall or blob" needs.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    # A lumen sits in the hole of its own piece and touches it, or all but --
    # the holes were found on a copy of the band closed by 6 um, so a piece
    # that far from the hole is still the piece enclosing it.
    reach = max(1, int(round(6.0 / max(px_um, 1e-6)))) + 1
    taken: set[int] = set()
    for lumen in lumina or ():
        touching = lab[square_morph(lumen, 2 * reach + 1, "dilate")]
        taken.update(int(k) for k in np.unique(touching) if k)

    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out: list[dict] = []
    for k in np.argsort(-sizes):
        if k == 0 or sizes[k] == 0 or int(k) in taken:
            continue
        piece = lab == k
        area = float(sizes[k]) * px_um ** 2
        width = 4.0 * float(edt(piece)[piece].mean()) * px_um
        length = area / max(width, 1e-9)
        aspect = length / max(width, 1e-9)
        ys, xs = np.nonzero(piece)
        ok = bool(area >= min_area_um2 and aspect >= min_aspect)
        out.append({
            "mask": piece,
            "area_um2": area,
            "width_um": width,
            "length_um": length,
            "x_px": float(xs.mean()),
            "y_px": float(ys.mean()),
            "vessel": ok,
            "why": ("" if ok else
                    (f"band {area:.0f} um2 < {min_area_um2:.0f}" if area < min_area_um2
                     else f"{length:.0f} um long by {width:.0f} um thick: a blob, not a wall")),
        })
    return out


def centerline_from_mask(
    mask: np.ndarray, prune_px: int = 25, min_hole_px: int = 400, lumen_rank: int = 0,
    px_um: float = 1.0, min_wall_um: float = 4.0,
    lumina: list[np.ndarray] | None = None,
    arcs: list[np.ndarray] | None = None,
    tissue: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Trace the middle of the wall band, as an ordered list of ``(x, y)`` points.

    Two topologies occur and telling them apart is not optional.  A complete
    vessel cross-section encloses its lumen: the reference curve comes from the
    lumen outline, which is closed and already ordered.  An incompletely
    photographed vessel is an open arc with no enclosed lumen, and there the
    skeleton's longest path is the right answer.

    Whether the wall closes is decided from the mask -- a ring encloses a hole,
    an arc does not -- and not from whether the skeleton happens to have
    endpoints, which depends only on how many adventitial strands touched the
    band.
    """
    if lumina is None:
        lumina = enclosed_lumina(mask, min_hole_px, px_um, min_wall_um)
    if lumen_rank < len(lumina):
        xy, _half = _lumen_contour_centerline(mask, lumina[lumen_rank])
        return xy, True

    # Past the rings come the open arcs, one per piece of band that encloses
    # nothing (:func:`wall_pieces`), and each is traced on *its own piece*.
    # Skeletonising the whole band instead lets the two vessels in one section
    # compete for a single longest path, which only one of them can win.
    if arcs is None:
        arcs = [] if lumina else [mask]      # nothing said: the band is one arc
    rank = lumen_rank - len(lumina)
    if rank >= len(arcs):
        raise ValueError(f"this section holds {len(lumina)} enclosed lumen(s) and "
                         f"{len(arcs)} open arc(s); asked for #{lumen_rank}")
    mask = arcs[rank]

    skel = skeletonize(mask)
    if not skel.any():
        raise ValueError("The wall band has no skeleton; check the media threshold.")
    skel = _branch_prune(skel, max(3, int(prune_px)))
    coords, nbrs = _skeleton_graph(skel)
    if len(coords) < 10:
        raise ValueError("The wall centre line is too short to analyse.")
    deg = np.array([len(v) for v in nbrs])
    ends = np.where(deg == 1)[0]
    if len(ends) == 0:
        ends = np.array([0])
    d1, _ = _dijkstra(nbrs, coords, int(ends[0]))
    start = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    d2, _ = _dijkstra(nbrs, coords, start)
    goal = int(np.argmax(np.where(np.isfinite(d2), d2, -1)))

    dist, prev = _dijkstra(nbrs, coords, start)
    if not np.isfinite(dist[goal]):
        goal = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))
    path = []
    cur = goal
    guard = 0
    while cur != -1 and guard < len(coords) + 5:
        path.append(cur)
        if cur == start:
            break
        cur = prev[cur]
        guard += 1
    pts = coords[path[::-1]]
    xy = np.column_stack([pts[:, 1], pts[:, 0]]).astype(np.float64)

    # A path whose two ends are a short step apart across unbroken tissue is a
    # ring, whatever the hole test made of the mask.
    #
    # That test asks a topological question of a per-pixel threshold, so any one
    # thin place answers it for the whole vessel.  On this thoracic aorta a
    # single oblique streak about 9 um across -- a fold, or a pale band in the
    # stain, with the wall plainly continuous through it in the picture -- broke
    # the media mask, `binary_fill_holes` then found no lumen at all, and a
    # 7.2 mm ring was measured as an open arc.  It is also why the same section
    # came out closed from one render of the mosaic and open from another: the
    # streak sits on the threshold, so the answer followed the quantisation
    # rather than the specimen.
    #
    # Distance alone cannot settle it, because a ring with a small tear and an
    # arc that nearly closes are the same shape: on synthetic rings a 60 um tear
    # and a genuinely open 350-degree arc give endpoint gaps of 5 % and 3 % of
    # the perimeter, in the wrong order.  What separates them is not how far
    # apart the ends are but *what lies between them* -- the wall carries on
    # through a staining artefact and stops at a real opening.  So the gap is
    # crossed and the tissue is read along it, and the ring is closed only if
    # the tissue never runs out; `tissue` is the section, not the muscle
    # threshold that just failed, so a streak the muscle mask lost is still
    # solidly tissue.  Without it, distance alone is not trusted.
    seg = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    gap = float(np.hypot(*(xy[0] - xy[-1])))
    if tissue is not None and seg > 0 and len(xy) >= 16 and gap <= 0.02 * seg:
        # ...and no further than a couple of wall thicknesses, so that two ends
        # which merely pass close by are never stitched into one vessel.
        dist = edt(mask)
        thick = 2.0 * float(np.median(dist[mask])) if mask.any() else 0.0
        if gap <= max(2.0 * thick, 4.0 / max(px_um, 1e-9)):
            n = max(2, int(round(gap)) + 1)
            xs = np.linspace(xy[0, 0], xy[-1, 0], n)
            ys = np.linspace(xy[0, 1], xy[-1, 1], n)
            rr = np.clip(np.round(ys).astype(int), 0, tissue.shape[0] - 1)
            cc = np.clip(np.round(xs).astype(int), 0, tissue.shape[1] - 1)
            if float(np.mean(tissue[rr, cc])) >= 0.95:
                return xy, True
    return xy, False


def traced_band_mask(
    shape_hw: tuple[int, int],
    centerline_xy: np.ndarray,
    normals_xy: np.ndarray,
    inner_um: np.ndarray,
    outer_um: np.ndarray,
    px_um: float,
) -> np.ndarray:
    """Rasterise the strip of wall that the straightening actually measured."""
    inner_px = np.nan_to_num(inner_um) / max(px_um, 1e-9)
    outer_px = np.nan_to_num(outer_um) / max(px_um, 1e-9)
    p_in = centerline_xy + normals_xy * inner_px[:, None]
    p_out = centerline_xy + normals_xy * outer_px[:, None]
    mask = np.zeros(shape_hw, dtype=np.uint8)
    quads = [np.round(np.array([p_in[i], p_in[i + 1], p_out[i + 1], p_out[i]])).astype(np.int32)
             for i in range(len(centerline_xy) - 1)]
    if not quads:
        return mask.astype(bool)
    if cv2 is not None:
        cv2.fillPoly(mask, quads, 1)
    else:  # pragma: no cover
        from skimage.draw import polygon
        for q in quads:
            rr, cc = polygon(q[:, 1], q[:, 0], shape_hw)
            mask[rr, cc] = 1
    return mask.astype(bool)


def find_branch_nodes(
    media: np.ndarray,
    centerline_xy: np.ndarray,
    normals_xy: np.ndarray,
    inner_um: np.ndarray,
    outer_um: np.ndarray,
    px_um: float,
    min_limb_um: float = 80.0,
    return_touch: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """
    Where does the wall fork?  Returns the attachment points as ``(x, y)`` pixels.

    With *return_touch* the pixels where each limb meets the traced wall come
    back too.  A junction is not a point: the branch peels away from the parent
    over a few hundred microns, and how far is a property of this junction
    rather than a multiple of the wall thickness.  Those pixels give the span
    directly, which is what the caller needs to know how much of the wall is
    junction rather than wall.

    A branch is a *topological* fact -- the wall continues in a third direction --
    and not a thickness, which is the distinction that matters: a genuinely
    thickened wall is the thing one would be looking for in a diseased vessel and
    must not be mistaken for a junction.

    The test is what is left over.  Rasterise the strip that was actually
    measured, subtract it from the media mask, and see whether any substantial
    piece of wall remains attached: at a fork the branch's own wall is media that
    the traced strip never covered, while a merely thick wall is entirely inside
    it and leaves nothing behind.

    (Skeletonising the media and looking for degree-three nodes is the textbook
    answer and does not work here.  A wall band is tens of microns thick, so its
    skeleton is a mesh of small loops: on a plain thoracic ring with no branch at
    all that test reported 182 forks.)
    """
    if len(centerline_xy) < 4:
        return (np.empty((0, 2)), []) if return_touch else np.empty((0, 2))
    band = traced_band_mask(media.shape, centerline_xy, normals_xy,
                            inner_um, outer_um, px_um)
    reach = max(2, int(round(min_limb_um * 0.12 / max(px_um, 1e-6))))
    leftover = media & ~square_morph(band, 2 * reach + 1, "dilate")
    min_area = (min_limb_um / max(px_um, 1e-6)) ** 2 * 0.25
    leftover = drop_small_objects(leftover, int(min_area))
    if not leftover.any():
        return (np.empty((0, 2)), []) if return_touch else np.empty((0, 2))

    lab, n = ndi.label(leftover)
    band_edge = ndi.binary_dilation(band, np.ones((3, 3)), iterations=reach + 3) & ~band
    out, touches = [], []
    for k in range(1, n + 1):
        comp = lab == k
        touch = comp & band_edge
        if not touch.any():
            continue  # a separate vessel, not a branch of this one
        ys, xs = np.nonzero(touch)
        out.append([float(xs.mean()), float(ys.mean())])
        touches.append(np.column_stack([xs, ys]).astype(float))
    nodes = np.array(out, dtype=float) if out else np.empty((0, 2))
    return (nodes, touches) if return_touch else nodes


def branch_arc_positions(
    nodes_xy: np.ndarray,
    centerline_xy: np.ndarray,
    arc_um: np.ndarray,
    px_um: float,
    max_distance_um: float,
) -> list[dict[str, float]]:
    """Match each fork to the nearest place on the traced wall."""
    out: list[dict[str, float]] = []
    if len(nodes_xy) == 0 or len(centerline_xy) == 0:
        return out
    for x, y in nodes_xy:
        d = np.hypot(centerline_xy[:, 0] - x, centerline_xy[:, 1] - y)
        i = int(np.argmin(d))
        dist_um = float(d[i] * px_um)
        if dist_um <= max_distance_um:
            out.append({"arc_um": float(arc_um[i]), "distance_um": dist_um,
                        "x_px": float(x), "y_px": float(y)})
    out.sort(key=lambda r: r["arc_um"])
    return out


def resample_centerline(
    xy: np.ndarray, step_px: float, smooth_px: float, closed: bool
) -> np.ndarray:
    """
    Smooth the traced path and resample it at a genuinely constant arc step.

    Two passes are needed.  A pixel-traced outline is a staircase, so its raw
    length overstates the true one by up to 40 % and spacing points by that
    length puts them far closer together than asked.  Every per-length quantity
    downstream is a sum times ``arc_step_um``, so the error would go straight
    into the answer.  The curve is therefore smoothed first, re-measured, and
    only then resampled.
    """
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    xy, d = xy[keep], d[keep]
    if len(xy) < 8:
        raise ValueError("Centre line too short after cleaning.")
    per = 1 if closed else 0
    if closed and (abs(xy[0, 0] - xy[-1, 0]) > 1e-6 or abs(xy[0, 1] - xy[-1, 1]) > 1e-6):
        xy = np.vstack([xy, xy[:1]])
        d = np.r_[d, d[-1] + float(np.hypot(*(xy[0] - xy[-2])))]

    # `s` in splprep is a sum-of-squares tolerance, so it must scale with the
    # number of points for the amount of smoothing to be resolution-independent.
    s_par = (smooth_px ** 2) * len(xy) * 0.02
    try:
        tck, _ = splprep([xy[:, 0], xy[:, 1]], u=d / d[-1], s=s_par, k=3, per=per)
    except Exception:
        tck, _ = splprep([xy[:, 0], xy[:, 1]], u=d / d[-1], s=0.0, k=1, per=per)

    dense_u = np.linspace(0, 1, max(2000, 4 * len(xy)))
    dx, dy = splev(dense_u, tck)
    seg = np.r_[0.0, np.cumsum(np.hypot(np.diff(dx), np.diff(dy)))]
    total = float(seg[-1])
    n_out = max(16, int(round(total / max(step_px, 1e-3))))
    targets = np.linspace(0, total, n_out, endpoint=not closed)
    u_even = np.interp(targets, seg, dense_u)
    x, y = splev(u_even, tck)
    return np.column_stack([x, y]).astype(np.float32)


def tangents_normals(xy: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    if closed:
        dx = np.gradient(np.r_[xy[-1, 0], xy[:, 0], xy[0, 0]])[1:-1]
        dy = np.gradient(np.r_[xy[-1, 1], xy[:, 1], xy[0, 1]])[1:-1]
    else:
        dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    mag = np.hypot(dx, dy) + 1e-9
    t = np.column_stack([dx / mag, dy / mag])
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t.astype(np.float32), n.astype(np.float32)


# =============================================================================
# 3.  Unrolling the wall
# =============================================================================


def curvature(xy: np.ndarray, closed: bool) -> np.ndarray:
    """
    Signed curvature of the sampling curve, per pixel of arc length.

    Positive means the curve bends towards the ``(-t_y, t_x)`` normal, which is
    the direction depth is measured in, so a sampling line at depth *d* is
    stretched by ``1 - kappa*d`` relative to the curve.  That factor is the whole
    story of what unrolling does to a curved wall: on the concave side the
    sampling lines crowd together and the tissue is squeezed, on the convex side
    they fan out and it is smeared.
    """
    if closed:
        wrap = lambda v: np.r_[v[-1], v, v[0]]
        dx = np.gradient(wrap(xy[:, 0]))[1:-1]
        dy = np.gradient(wrap(xy[:, 1]))[1:-1]
        ddx = np.gradient(np.r_[dx[-1], dx, dx[0]])[1:-1]
        ddy = np.gradient(np.r_[dy[-1], dy, dy[0]])[1:-1]
    else:
        dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
    speed = np.maximum(np.hypot(dx, dy), 1e-9)
    return ((dx * ddy - dy * ddx) / speed ** 3).astype(np.float64)


def stretch_factor(xy: np.ndarray, depths_um: np.ndarray, px_um: float,
                   closed: bool, smooth_pts: int = 0) -> np.ndarray:
    """
    ``(n_arc, n_depth)`` Jacobian of the unrolling, ``1 - kappa*d``.

    Where this is not positive the sampling lines have crossed: the same tissue
    is being read more than once and the "straightened" image there is a fold,
    not a measurement.  Where it is merely far from 1, the image is locally
    compressed or stretched -- which is unavoidable when a curved band is laid
    flat, but it must be divided out of any integral, or the convex side of
    every bend is counted more than it should be.
    """
    k = curvature(xy, closed) / max(px_um, 1e-9)  # per micron
    if smooth_pts and smooth_pts > 1:
        mode = "wrap" if closed else "nearest"
        k = ndi.uniform_filter1d(k, int(smooth_pts), mode=mode)
    return (1.0 - k[:, None] * depths_um[None, :]).astype(np.float32)


def sample_band(
    img: np.ndarray,
    xy: np.ndarray,
    normals: np.ndarray,
    depths_px: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Sample *img* on the (arc length x depth) grid.

    Returns ``(n_arc, n_depth)`` for a 2-D input, or ``(n_arc, n_depth, C)``.
    """
    xs = xy[:, 0][:, None] + normals[:, 0][:, None] * depths_px[None, :]
    ys = xy[:, 1][:, None] + normals[:, 1][:, None] * depths_px[None, :]
    if order == 1:
        return _bilinear(np.asarray(img, np.float32), ys, xs)
    if img.ndim == 2:
        return ndi.map_coordinates(img.astype(np.float32), [ys, xs], order=order,
                                   mode="constant", cval=np.nan)
    return np.stack([ndi.map_coordinates(img[..., c].astype(np.float32), [ys, xs],
                                         order=order, mode="constant", cval=np.nan)
                     for c in range(img.shape[2])], axis=-1)


def _bilinear(plane: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """
    ``map_coordinates(order=1, mode="constant", cval=nan)``, written out.

    Straightening a section samples four of these bands over a couple of
    million points, and scipy's general spline machinery charges three times
    what the four corner lookups below actually cost.  Same arithmetic, same
    NaN outside the image, agreeing to the last bit of a float32.

    An RGB *plane* is sampled in one pass rather than three.  The corner
    arithmetic is the same for every channel, and the three values of a pixel
    sit next to each other in memory, so one gather of three is most of a
    section's colour sampling for the price of one channel's.
    """
    h, w = plane.shape[:2]
    inside = (ys >= 0) & (ys <= h - 1) & (xs >= 0) & (xs <= w - 1)
    # Clipping rather than masking keeps the gather in one pass; the clipped
    # rows carry a weight of one, so a coordinate sitting exactly on the last
    # row still reads that row rather than falling off it.
    y0 = np.floor(ys); np.clip(y0, 0, h - 2, out=y0)
    x0 = np.floor(xs); np.clip(x0, 0, w - 2, out=x0)
    # Double weights, as scipy uses internally, so the two agree bit for bit
    # and a section measured last month still measures the same today.  Said
    # out loud, because the band coordinates arrive as float32 and letting the
    # subtraction pick the type would carry the whole interpolation in single.
    fy = np.subtract(ys, y0, dtype=np.float64)
    fx = np.subtract(xs, x0, dtype=np.float64)
    if plane.ndim > 2:
        fy, fx, inside = fy[..., None], fx[..., None], inside[..., None]
    flat = plane.reshape(h * w, -1) if plane.ndim > 2 else plane.reshape(-1)
    # Written in place from here down.  An RGB band is twenty million samples,
    # so each named temporary is another 170 MB over the memory bus, and the
    # bus is what this function is waiting on.
    base = y0.astype(np.intp); base *= w
    base += x0.astype(np.intp)
    a = flat[base]
    base += 1
    b = flat[base]
    base += w - 1
    c = flat[base]
    base += 1
    d = flat[base]
    del base
    top = (b - a) * fx
    top += a
    bot = (d - c) * fx
    bot += c
    bot -= top
    bot *= fy
    top += bot
    out = top.astype(np.float32)
    np.copyto(out, np.float32(np.nan), where=~inside)
    return out


def _fill_nan_1d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.zeros_like(v)
    if ok.sum() == 1:
        v[~ok] = v[ok][0]
        return v
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    return v


def _muscle_centre_um(band_muscle: np.ndarray, depths_um: np.ndarray,
                      reach_um: float, level: float) -> np.ndarray:
    """
    Where the muscle is, along each sampling line -- a target for a curve that
    has fallen off the wall.

    The re-centring pass aims at the midpoint between the wall edges, which is
    the right target while the curve is on the wall and no target at all once it
    is not: with no muscle underneath it the edge finder reports a band a few
    microns thick, the correction is refused as implausible, and the curve stays
    where it is.  On the aortic arch that locked a quarter of the traced length
    into the adventitia, 18 um from a media 43 um thick.

    The peak of the muscle profile is a target that survives the curve being
    wrong, because it is a fact about the image rather than about the boundaries
    just measured.  Half-maximum weighting rather than a plain argmax, so the
    answer is the middle of the muscle and not whichever pixel happened to be
    darkest; *level* keeps noise on an empty profile from being mistaken for a
    wall.
    """
    within = np.abs(depths_um) <= reach_um
    prof = np.where(within[None, :], np.nan_to_num(band_muscle, nan=0.0), 0.0)
    peak = prof.max(axis=1)
    w = np.clip(prof - 0.5 * peak[:, None], 0.0, None)
    tot = w.sum(axis=1)
    centre = np.divide((w * depths_um[None, :]).sum(axis=1), tot,
                       out=np.full(len(prof), np.nan), where=tot > 0)
    return np.where(peak > level, centre, np.nan)


def _despike(v: np.ndarray, med_win: int, max_dev: float, smooth: float,
             closed: bool) -> np.ndarray:
    """Median-filter outliers out of a boundary trace, then smooth it."""
    v = _fill_nan_1d(v)
    mode = "wrap" if closed else "nearest"
    med = ndi.median_filter(v, size=max(3, med_win | 1), mode=mode)
    bad = np.abs(v - med) > max_dev
    v = np.where(bad, np.nan, v)
    v = _fill_nan_1d(v)
    if smooth > 0:
        v = ndi.gaussian_filter1d(v, smooth, mode=mode)
    return v


def find_wall_edges(
    muscle_band: np.ndarray,
    collagen_band: np.ndarray,
    tissue_band: np.ndarray,
    depths_um: np.ndarray,
    floor_od: float,
    p: WallParams,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate, for every position along the wall, the three surfaces that matter:
    the luminal surface, the media/adventitia junction, and the outer edge of
    the tissue.

    The luminal surface comes from *total* optical density, which across a
    vessel wall is a clean top-hat: nothing, then wall, then loose adventitia.
    Muscle concentration alone is too spiky for this -- individual elastic
    lamellae and nuclei make it oscillate by a factor of two within the media,
    so a threshold on it finds a short run and reports a wall a few microns
    thick when it is fifty.

    The media/adventitia junction comes from one of two rules, and which is
    right depends on what the section is for.

    ``collagen-crossing`` ends the media where **collagen overtakes muscle**
    and stays ahead.  That is the external elastic lamina on a healthy wall,
    and the crossing is only accepted once it holds for several microns rather
    than at the first dip.  But it defines the edge of the media in the same
    quantity fibrosis is measured in, and no hold length repairs that: a focal
    patch inside the media ends the media at the patch, so the lesion is
    measured as adventitia and the wall is reported thin.  On this thoracic
    aorta, at a patch where collagen/muscle reaches 1.25 against the vessel's
    own 0.64, the edge lands 15 um inside a media whose neighbours are 31 um
    thick.  The more collagen a stretch has, the less of it is counted.

    ``muscle-end`` ends the media where the **muscle runs out**: the first
    sustained fall of muscle below a fraction of this section's own muscle
    level.  It says nothing about collagen, so the boundary cannot be moved by
    the thing being measured, and a fibrotic patch keeps its place inside the
    wall.  It is also the more stable of the two at the edge itself, where
    muscle falls off steeply -- between a quarter and a twentieth of the
    plateau the answer moves by about a micron -- while the crossing rule
    depends on two noisy quantities being compared near equality.
    """
    n_s, n_d = tissue_band.shape
    dd = float(depths_um[1] - depths_um[0])
    k_smooth = max(1, int(round(2.0 / dd)))
    k_hold = max(2, int(round(6.0 / dd)))

    def smooth(band: np.ndarray) -> np.ndarray:
        return ndi.uniform_filter1d(np.nan_to_num(band, nan=0.0), k_smooth, axis=1)

    T, M, C = smooth(tissue_band), smooth(muscle_band), smooth(collagen_band)
    dom = ndi.uniform_filter1d(M - C, max(2, int(round(4.0 / dd))), axis=1)
    centre = int(np.argmin(np.abs(depths_um)))
    near = np.abs(depths_um) <= 8.0

    # Both rules are the same walk: outwards to the first sustained fall of
    # some profile through some level.  Only the profile and the level differ,
    # so the hold, the interpolation and the clamp to the tissue edge below are
    # shared and neither rule gets its own quietly different edge finder.
    if str(getattr(p, "media_edge_rule", "muscle-end")) == "muscle-end":
        edge_prof = M
        # A fraction of the section's own muscle, not an absolute
        # concentration: staining intensity varies between slides, and an
        # absolute level would make the media thicker on a darker slide.
        plateau = float(np.nanmedian(np.nanmax(M, axis=1)))
        edge_level = float(getattr(p, "muscle_end_fraction", 0.25)) * plateau
    else:
        edge_prof = dom
        edge_level = 0.0

    inner = np.full(n_s, np.nan)
    outer = np.full(n_s, np.nan)
    tissue_out = np.full(n_s, np.nan)

    def interp_cross(prof: np.ndarray, i0: int, i1: int, level: float) -> float:
        """Depth where *prof* crosses *level* between samples i0 and i1."""
        a, b = prof[i0], prof[i1]
        if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-12:
            return float(depths_um[i1])
        f = (level - a) / (b - a)
        f = float(np.clip(f, 0.0, 1.0))
        return float(depths_um[i0] + f * (depths_um[i1] - depths_um[i0]))

    for i in range(n_s):
        prof = T[i]
        plateau = float(np.nanmedian(prof[near]))
        level = max(0.5 * plateau, floor_od)
        above = prof > level
        if not above.any():
            continue
        if above[centre]:
            c = centre
        else:
            idx = np.where(above)[0]
            c = int(idx[np.argmin(np.abs(idx - centre))])
        lo = c
        while lo - 1 >= 0 and above[lo - 1]:
            lo -= 1
        hi = c
        while hi + 1 < n_d and above[hi + 1]:
            hi += 1

        inner[i] = interp_cross(prof, max(lo - 1, 0), lo, level) if lo > 0 else depths_um[lo]
        tissue_out[i] = interp_cross(prof, hi, min(hi + 1, n_d - 1), level) if hi < n_d - 1 \
            else depths_um[hi]

        # Walk outward from the reference point to the first *sustained* place
        # where the chosen profile has fallen through its level.
        g = edge_prof[i]
        k = max(c, lo)
        cut = hi
        j = k
        while j <= hi:
            if g[j] <= edge_level and np.all(g[j:min(j + k_hold, hi + 1)] <= edge_level):
                cut = j
                break
            j += 1
        if cut > lo and g[cut] <= edge_level < g[max(cut - 1, lo)]:
            outer[i] = interp_cross(g, cut - 1, cut, edge_level)
        else:
            outer[i] = float(depths_um[cut])
        outer[i] = min(outer[i], tissue_out[i])

    win = max(3, int(round(p.boundary_smooth_um / max(p.arc_step_um, 1e-6))))
    sm = max(1.0, p.boundary_smooth_um / max(p.arc_step_um, 1e-6) / 2.0)
    inner = _despike(inner, win, p.boundary_outlier_um, sm, closed)
    outer = _despike(outer, win, p.boundary_outlier_um, sm, closed)
    tissue_out = _despike(tissue_out, win, p.boundary_outlier_um * 3, sm, closed)
    bad = outer <= inner
    if bad.any():
        outer[bad] = inner[bad] + 1e-3
    return inner, outer, tissue_out


# What counts as a fragmented outer media: muscle reaching this far past the
# edge, with less than this much of the reach being muscle.  `static/app.js`
# repeats both figures in its tooltip; change them together.
# How much of each wall boundary is ramp rather than media, dropped before
# measuring how evenly the muscle inside is laid down.
VARIATION_EDGE_TRIM = 0.15
FRAGMENT_REACH_UM = 12.0
FRAGMENT_FILL = 0.8


def muscle_beyond_edge(
    muscle_band: np.ndarray,
    depths_um: np.ndarray,
    outer_um: np.ndarray,
    p: "WallParams",
) -> tuple[np.ndarray, np.ndarray]:
    """
    How far muscle carries on past the media edge, and how solid it is there.

    Both edge rules stop at a *boundary*, and both are clamped to the tissue
    mask.  Neither can describe a media whose outer layer has broken up: the
    fragments sit in pale gaps, the gaps drag the region below the tissue
    threshold, and the whole layer is then outside the mask that the boundary
    is clamped to.  Measured on a middle aorta, muscle ran to 34 um where the
    tissue edge sat at 22 and the collagen-crossing edge at 17.

    So this looks in a fixed window past the edge -- raw band, no tissue clamp,
    no stopping at the first gap, because the gaps are the point -- and reports
    two numbers rather than moving anything:

    ``reach_um``  how far out the last muscle is found within the window.

    ``fill``  what fraction of that reach is muscle.  The pair separates the
    cases a single number confuses: ``fill`` near 1 is solid muscle the edge
    stopped short of, and a low ``fill`` over a long reach is a genuinely
    fragmented outer media.  A walk that stopped at the first gap could not
    tell them apart -- it would end before the fragments and report ~1 for
    both -- which is why the window is fixed and the gaps are counted.

    Nothing here changes a boundary or a total; it describes what was left
    outside one.  The level, the plateau and the 2 um depth smooth are the
    ``muscle-end`` rule's, so the two agree about what muscle is.

    There is deliberately no smoothing along the wall.  The unsteadiness of
    ``fill`` from one position to the next *is* the fragmentation: on three
    intact sections it sits at 1.00 and does not move at all, and smoothing
    over 10 um pulled the one fragmented section from 0.73 back towards 0.85
    and halved the separation.  It looks like noise and is the signal.
    """
    dd = float(depths_um[1] - depths_um[0])
    M = ndi.uniform_filter1d(np.nan_to_num(muscle_band, nan=0.0),
                             max(1, int(round(2.0 / dd))), axis=1)
    plateau = float(np.nanmedian(np.nanmax(M, axis=1)))
    level = float(getattr(p, "muscle_end_fraction", 0.25)) * plateau
    reach = max(1, int(round(float(getattr(p, "fragment_reach_um", 25.0)) / dd)))
    reach_um = np.zeros(len(outer_um), dtype=np.float32)
    fill = np.full(len(outer_um), np.nan, dtype=np.float32)
    if not np.isfinite(plateau) or plateau <= 0:
        return reach_um, fill
    for i, o in enumerate(outer_um):
        if not np.isfinite(o):
            continue
        j0 = int(np.searchsorted(depths_um, o))
        on = M[i, j0:j0 + reach] >= level
        if not on.size:
            continue
        last = np.flatnonzero(on)
        if not last.size:
            fill[i] = 0.0
            continue
        k = int(last[-1]) + 1
        reach_um[i] = k * dd
        fill[i] = float(on[:k].mean())
    return reach_um, fill


def radial_variation(
    muscle_band: np.ndarray,
    depths_um: np.ndarray,
    inner_um: np.ndarray,
    outer_um: np.ndarray,
    p: "WallParams",
) -> tuple[np.ndarray, np.ndarray]:
    """
    How evenly the muscle is laid down across the wall, lumen to outer edge.

    An intact media is a solid block of muscle: walk from the lumen outwards
    and the stain stays near its own average.  A media that has broken up has
    the same average with holes in it, so the walk rises and falls.  That is
    the coefficient of variation of the radial profile -- spread over its own
    mean, so a darker slide does not read as a more broken one.

    The catch is that an aortic media is *meant* to be layered.  Elastic
    lamellae alternate with muscle on a 4-5 um pitch here, measured the same
    on all four sections to hand, and a section that resolves them crisply
    varies more across the wall than one that smears them together.  Taken
    raw, this measure ranks a well-cut middle aorta (0.29) almost level with a
    visibly fragmented one (0.34): it reads staining and focus, not pathology.

    So the profile is split by scale at ``variation_scale_um``, about two and a
    half lamellae:

    ``coarse``  variation of the profile smoothed at that scale -- dropouts
    bigger than a lamella.  This is the fragmentation readout.  It separates
    the fragmented section from the crisply-lamellar one 1.7x, where the raw
    figure separated them 1.2x.

    Both are taken over the middle of the wall, not all of it.  Muscle ramps
    down at each boundary because that is what a boundary is, so a profile run
    edge to edge partly measures where the edge was drawn -- and a thin wall
    is mostly ramp.  Untrimmed, walls of 16-20 um read 0.44 against 0.20 for
    walls of 30-60 um on one and the same section.  Dropping 15% at each end
    cuts that bias and sharpens the separation between the fragmented section
    and the crisp one from 1.7x to 2.0x; past 20% it starts eating media and
    both get worse.

    ``lamellar``  what the smoothing removed: the banding itself.  Reported
    because it is the reason a coarse figure can surprise -- the crisp section
    scores *higher* here (0.23) than the fragmented one (0.20) -- and because
    a section with no lamellar contrast at all is one whose coarse figure has
    little left to measure.
    """
    dd = float(depths_um[1] - depths_um[0])
    k = max(2, int(round(float(getattr(p, "variation_scale_um", 12.0)) / dd)))
    n = len(inner_um)
    coarse = np.full(n, np.nan, dtype=np.float32)
    lamellar = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo, hi = inner_um[i], outer_um[i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 2.0 * dd * k / 2.0:
            continue
        j0 = int(np.searchsorted(depths_um, lo))
        j1 = int(np.searchsorted(depths_um, hi))
        t = int(round(VARIATION_EDGE_TRIM * (j1 - j0)))
        j0, j1 = j0 + t, j1 - t
        v = np.nan_to_num(muscle_band[i, j0:j1], nan=0.0)
        if v.size < 4:
            continue
        m = float(v.mean())
        if m <= 1e-6:
            continue
        sm = ndi.uniform_filter1d(v, min(k, v.size))
        coarse[i] = float(sm.std()) / m
        lamellar[i] = float((v - sm).std()) / m
    return coarse, lamellar


def lamellar_geometry(
    muscle_band: np.ndarray,
    depths_um: np.ndarray,
    inner_um: np.ndarray,
    outer_um: np.ndarray,
    p: "WallParams",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    How far apart the layers sit, how evenly, and how sharply this section
    resolves them at all.

    ``spacing``  distance between successive lamellae, from the peaks of the
    same residual ``radial_variation`` calls ``lamellar``.  Standard aortic
    morphometry and the one layering number that came out robust here: 5.2,
    5.8 and 6.0 um on the three well-resolved sections.

    ``disorder``  the spread of those gaps over their mean.  Regularly stacked
    lamellae give a low figure whatever the staining; a media whose layers
    have merged, split or gone missing gives a high one.  Read it with its
    coverage and with ``edge_width``: it is measured only where three
    lamellae could be found, and blur inflates it (r = +0.18 to +0.26 against
    edge width) because faint banding is picked out of noise.  It is
    independent of everything else here -- r ~ 0 against ``muscle_variation``
    on all four sections -- but it is the weakest separator of them: 1.8x
    between the fragmented section and the crisply layered one, only 1.3x
    against the intact group as a whole, at any threshold tried.

    ``edge_width``  the 10-90% rise of muscle across the luminal boundary.
    The lumen is empty and the media starts at once, so this is the section's
    own blur, measured without reference to any lamella -- and it is what
    makes the layering numbers readable, because every one of them is
    attenuated by blur.  Ship the two together or neither.

    That pairing is the whole point, and it was not obvious.  ``lamellar``
    looked like pure section quality until this was measured: 1971 is exactly
    as sharp as 3690 (2.0 um both) yet bands at 0.107 against 0.191, so that
    gap is real structure, while 3971's feeble 0.074 comes with a 4.0 um edge
    and is mostly optics.  A layering figure without an edge width beside it
    cannot tell those two cases apart.

    Peak prominence is set against the wall's own mean muscle level, not
    against the residual's spread.  A spread-relative threshold is
    self-scaling: a flat, blurred wall still yields the largest wiggles in its
    own noise and reads as layered.
    """
    dd = float(depths_um[1] - depths_um[0])
    k = max(2, int(round(float(getattr(p, "variation_scale_um", 12.0)) / dd)))
    f = float(getattr(p, "lamella_prominence", 0.10))
    gap = max(1, int(round(2.0 / dd)))
    n = len(inner_um)
    spacing = np.full(n, np.nan, dtype=np.float32)
    disorder = np.full(n, np.nan, dtype=np.float32)
    edge = np.full(n, np.nan, dtype=np.float32)
    win = max(4, int(round(15.0 / dd)))
    for i in range(n):
        lo, hi = inner_um[i], outer_um[i]
        if not np.isfinite(lo):
            continue
        # The luminal rise: inwards of the boundary is lumen, outwards media.
        j = int(np.searchsorted(depths_um, lo))
        seg = muscle_band[i, max(0, j - win):j + win]
        if seg.size >= 2 * win - 1 and not np.isnan(seg).any():
            base = float(np.median(seg[:win // 3]))
            top = float(np.median(seg[-2 * win // 3:]))
            if top - base > 0.05:
                up = np.flatnonzero(seg >= base + 0.9 * (top - base))
                dn = np.flatnonzero(seg <= base + 0.1 * (top - base))
                if up.size and dn.size and dn[0] < up[0]:
                    back = dn[dn < up[0]]
                    if back.size:
                        edge[i] = (up[0] - back[-1]) * dd
        # Lamellar spacing, over the same trimmed middle of the wall.  The
        # requirement here is three peaks, not a thickness -- checked below,
        # once they have been counted.  Demanding two smoothing kernels of
        # wall as `radial_variation` does would drop the thin-walled sections
        # wholesale, and those are not the unlayered ones: it took 60% of the
        # most crisply lamellated section of the four.
        if not np.isfinite(hi) or hi - lo < dd * k:
            continue
        j0, j1 = int(np.searchsorted(depths_um, lo)), int(np.searchsorted(depths_um, hi))
        t = int(round(VARIATION_EDGE_TRIM * (j1 - j0)))
        j0, j1 = j0 + t, j1 - t
        v = np.nan_to_num(muscle_band[i, j0:j1], nan=0.0)
        if v.size < 4:
            continue
        mu = float(v.mean())
        if mu <= 1e-6:
            continue
        pk, _ = signal.find_peaks(v - ndi.uniform_filter1d(v, min(k, v.size)),
                                  prominence=f * mu, distance=gap)
        if len(pk) >= 3:
            gaps = np.diff(pk) * dd
            spacing[i] = float(np.median(gaps))
            disorder[i] = float(gaps.std() / gaps.mean())
    return spacing, disorder, edge


def _keep_long_runs(mask: np.ndarray, min_len: float, closed: bool) -> np.ndarray:
    """
    Keep only the runs of *mask* at least *min_len* positions long.

    On a ring the array is cut open at an arbitrary seam, so a run crossing it
    is one run, not two short ones each thrown away for being short.
    """
    mask = np.asarray(mask, bool)
    n = len(mask)
    need = max(1, int(np.ceil(min_len)))
    if n == 0 or not mask.any() or need <= 1:
        return mask.copy()
    if closed and mask.all():
        return mask.copy()
    shift = 0
    if closed and mask[0] and mask[-1]:
        shift = int(np.argmin(mask))       # roll so the array starts on a gap
    m = np.roll(mask, -shift)
    lab, k = ndi.label(m)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= need
    keep[0] = False
    return np.roll(keep[lab], shift)


def orient_band(
    tissue_band: np.ndarray | None,
    xy: np.ndarray,
    depths_um: np.ndarray,
    p: "WallParams",
    closed: bool,
) -> np.ndarray:
    """
    Which way round the sampling lines are, by whichever means is available.

    A ring is settled by its own geometry and cannot turn over -- it needs no
    band at all, and callers may pass none.  An arc has no enclosed region to
    appeal to, so it is left to the density vote, which abstains where it has
    no evidence rather than flipping the wall on noise.
    """
    if closed:
        return np.full(len(xy), side_from_winding(xy), dtype=bool)
    return lumen_side_per_position(tissue_band, depths_um, p.arc_step_um, closed=closed)


def side_from_winding(xy: np.ndarray) -> bool:
    """
    Which way the tangent's normal points, for a curve that closes.

    On a ring the question the optical-density vote is asking has already been
    answered by the geometry: the lumen is the region the curve encircles, so
    there is nothing to infer and nothing to be misled by.  The normal
    ``(-t_y, t_x)`` is the left hand of travel, and the left hand of a curve
    traversed with positive signed area points into the region it encloses, so
    one shoelace sum orients the whole ring at once -- no threshold, no window,
    and no way for a clean stretch of slide to turn a wall over.

    Returns True when the normal already points *away* from the lumen, which is
    the sense :func:`lumen_side_per_position` reports.
    """
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    twice_area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return bool(twice_area < 0.0)


def lumen_side_per_position(
    tissue_band: np.ndarray,
    depths_um: np.ndarray,
    arc_step_um: float,
    near_um: float = 30.0,
    window_um: float = 250.0,
    closed: bool = False,
    min_margin: float = 0.2,
) -> np.ndarray:
    """
    Which side of each sampling line is the lumen?  One answer per position.

    Past the wall the adventitial side carries connective tissue, vasa vasorum
    and fat while the luminal side is empty, so comparing mean optical density
    well away from the wall settles it -- and it needs no prior knowledge of
    where the wall edges are, which is the point, because the edge finder wants
    the orientation first.

    This has to be decided *per position*, not once for the whole vessel.  Where
    the traced path runs through a branch it comes out on the other wall, and
    from there on the lumen is on the opposite side; deciding globally leaves
    everything past the junction inside out, with adventitia drawn where the
    lumen should be.  The per-position votes are median-filtered over a window
    much wider than a junction, so the answer is stable through the few hundred
    microns where both sides carry tissue and the vote is genuinely ambiguous.

    A comparison always returns an answer, including where there is nothing to
    compare.  On a clean section with no perivascular fat both sides past the
    wall are empty slide: on one thoracic ring the two means came to 0.004 and
    0.001 against the 0.03 a real adventitia gives, and the winner of that was
    rounding noise.  The median filter then held the coin flip steady for the
    232 um it was widest over, and a perfectly good stretch of wall came out
    upside down -- the failure needs a *good* image, not a bad one, which is
    why it looks so unlike a measurement problem.  So a position only votes if
    its two sides differ by an appreciable fraction of the asymmetry this
    section actually shows; the rest abstain and take the answer from the
    nearest position that had one, which is the wall carrying its orientation
    through the quiet stretch rather than turning over inside it.
    """
    lo = depths_um < -near_um
    hi = depths_um > near_um
    a = np.nanmean(np.where(np.isfinite(tissue_band[:, lo]), tissue_band[:, lo], np.nan), axis=1)
    b = np.nanmean(np.where(np.isfinite(tissue_band[:, hi]), tissue_band[:, hi], np.nan), axis=1)
    diff = np.nan_to_num(b) - np.nan_to_num(a)
    vote = (diff > 0).astype(float)
    scale = float(np.percentile(np.abs(diff), 75))
    decided = np.abs(diff) >= min_margin * scale
    if scale > 0 and decided.any() and not decided.all():
        idx = np.arange(len(diff), dtype=float)
        # Binary values, so linear interpolation between the decided positions
        # and a cut at a half is nearest-neighbour fill.
        vote = (np.interp(idx, idx[decided], vote[decided]) > 0.5).astype(float)
    n = max(3, int(round(window_um / max(arc_step_um, 1e-6))) | 1)
    # A closed vessel wraps, so the filter must too -- otherwise the seam where
    # the ring was cut open votes against its own neighbours and a few positions
    # come out inside out on a wall that never turns anywhere.
    return ndi.median_filter(vote, size=n, mode="wrap" if closed else "nearest") > 0.5



# =============================================================================
# 4.  The analysis
# =============================================================================


@dataclass
class WallResult:
    px_um: float
    params: WallParams
    closed: bool
    image_shape: tuple[int, int]
    centerline_xy: np.ndarray
    normals_xy: np.ndarray
    arc_um: np.ndarray
    depth_um: np.ndarray
    straight_rgb: np.ndarray
    straight_muscle: np.ndarray
    straight_collagen: np.ndarray
    straight_tissue: np.ndarray
    inner_um: np.ndarray
    outer_um: np.ndarray
    tissue_outer_um: np.ndarray
    media_band: np.ndarray  # boolean (arc, depth)
    stretch: np.ndarray  # (arc, depth) Jacobian of the unrolling
    profile: pd.DataFrame  # per position along the wall
    depth_profile: pd.DataFrame  # averaged across the wall, absolute depth
    depth_profile_rel: pd.DataFrame  # averaged across the wall, relative depth
    totals: dict[str, float]
    stain_matrix: np.ndarray
    branches: list = field(default_factory=list)
    lumina: list = field(default_factory=list)   # every hole and piece of band
    #                                               weighed as a vessel
    rel_rgb: np.ndarray | None = None
    rel_collagen: np.ndarray | None = None
    rel_muscle: np.ndarray | None = None
    rel_depth: np.ndarray | None = None
    abs_rgb: np.ndarray | None = None      # aligned to the luminal edge
    abs_muscle: np.ndarray | None = None
    abs_collagen: np.ndarray | None = None
    abs_depth_um: np.ndarray | None = None
    # Focal collagen patches inside the media, and any marks placed by hand.
    # Both live in the *original* frame rather than the straightened one, so a
    # lesion's coordinates fall straight onto the map panel and the outline
    # editor without going back through the unrolling.
    lesions: list = field(default_factory=list)
    lesion_labels: np.ndarray | None = None
    points: list = field(default_factory=list)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, or NaN when there is nothing to correlate."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def analyze_wall(
    rgb01: np.ndarray,
    px_um: float,
    params: WallParams | None = None,
    roi: np.ndarray | None = None,
    stain_matrix: np.ndarray | None = None,
    lumen_rank: int = 0,
    coverage: np.ndarray | None = None,
    guide_um: np.ndarray | None = None,
    guide_closed: bool | None = None,
    points_um: np.ndarray | None = None,
    find_lesions_too: bool = True,
    lesion_min_area_um2: float = 2_000.0,
    lesion_k_mad: float = 3.0,
    exclude_lesions_at: np.ndarray | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> WallResult:
    """
    Straighten the wall and measure it.

    *rgb01* is a float image in [0, 1] with blank slide at 1.0 (that is what
    ``stitch_core.render_mosaic`` produces).  *px_um* is its pixel size.

    *coverage* marks the pixels the mosaic actually has data for; pass the mask
    ``render_mosaic(..., return_coverage=True)`` gives back.  Where it is absent
    the wall may be carried across the hole so the vessel still closes, and those
    stretches are then reported as no-data rather than measured.  Left at
    ``None`` it is inferred from the image, since untouched canvas is exactly
    white.

    *guide_um* replaces the automatic reference curve with one drawn by hand,
    in microns from the top-left of the image.  It does not have to be accurate:
    everything downstream is unchanged, so the re-centring pass pulls a rough
    line onto the middle of the media exactly as it does the automatic one.  It
    is for the cases no rule gets right -- which of two touching vessels to
    follow, whether to go up a branch or past it, where a torn wall ought to be
    joined -- and those are judgements about the specimen rather than defects in
    the tracer.
    """
    p = params or WallParams()

    def say(f: float, m: str) -> None:
        if progress:
            progress(f, m)

    # ---- resample to the analysis resolution ---------------------------
    say(0.02, "preparing image")
    scale = px_um / p.analysis_px_um
    if abs(scale - 1.0) > 0.02:
        if cv2 is not None:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(rgb01, None, fx=scale, fy=scale, interpolation=interp)
            if roi is not None:
                roi = cv2.resize(roi.astype(np.uint8), (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0
        else:
            img = ndi.zoom(rgb01, (scale, scale, 1), order=1)
            if roi is not None:
                roi = ndi.zoom(roi.astype(np.uint8), (scale, scale), order=0) > 0
    else:
        img = rgb01
    work_px_um = p.analysis_px_um if abs(scale - 1.0) > 0.02 else px_um

    if coverage is None:
        cover = coverage_from_mosaic(img)
    else:
        cover = coverage
        if cover.shape != img.shape[:2]:
            cover = (cv2.resize(cover.astype(np.uint8), (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0) if cv2 is not None else \
                    (ndi.zoom(cover.astype(np.uint8),
                              (img.shape[0] / cover.shape[0], img.shape[1] / cover.shape[1]),
                              order=0) > 0)

    # ---- stains ---------------------------------------------------------
    say(0.10, "separating stains")
    od = rgb_to_od(img, 1.0)
    od_sum = od.sum(axis=2)
    tis = tissue_mask(od_sum, p.tissue_od,
                      min_area_px=int(400 / max(work_px_um ** 2, 1e-6)))
    if roi is not None:
        tis &= roi
    M = estimate_stains(od, tis) if stain_matrix is None else np.asarray(stain_matrix, float)
    conc = separate_stains(od, M)
    muscle, collagen = conc[..., 0], conc[..., 1]

    # ---- geometry -------------------------------------------------------
    say(0.25, "finding the muscle layer")
    media = media_mask(muscle, tis, work_px_um, p, roi=roi)
    media_traced = media

    # Which holes in the band are vessels, decided once and reported in full.
    # The count alone used to be all that came back, which made "2 lumina" on a
    # section that shows one vessel and a scrap of another impossible to read:
    # you could not tell which two, or where, or whether the second was measured.
    candidates = lumen_candidates(media_traced, work_px_um, p.min_lumen_um2, p.min_wall_um)
    vessels = [c for c in candidates if c["vessel"]]
    n_lumina = len(vessels)
    # And the pieces of band that enclose nothing: each is a vessel in its own
    # right, traced as an open arc.  They are ranked after the rings, so
    # `lumen_rank` counts through both -- rings first, then arcs.
    pieces = wall_pieces(media_traced, [c["mask"] for c in vessels], work_px_um,
                         p.min_media_area_um2)
    arcs = [a for a in pieces if a["vessel"]]
    for rank, a in enumerate(arcs, start=n_lumina):
        a["rank"] = rank
    n_vessels = n_lumina + len(arcs)
    # Rank 0 still has to trace something on a section whose only piece of band
    # was too small or too round to call a vessel -- that is what it did before
    # any of this, and it is how a scrap of wall gets looked at at all.
    traceable = arcs or pieces[:1]
    say(0.35, "tracing the wall")
    if guide_um is not None:
        raw_xy = np.asarray(guide_um, dtype=float) / work_px_um
        if raw_xy.ndim != 2 or raw_xy.shape[1] != 2 or len(raw_xy) < 8:
            raise ValueError("a hand-drawn outline needs at least 8 (x, y) points")
        if guide_closed is None:
            # Closed if the two ends meet, within a fiftieth of the curve's own
            # size -- which is the tolerance a hand or a mouse can hold.
            span = float(max(np.ptp(raw_xy[:, 0]), np.ptp(raw_xy[:, 1]), 1.0))
            guide_closed = bool(np.hypot(*(raw_xy[0] - raw_xy[-1])) < 0.02 * span)
        if guide_closed and np.allclose(raw_xy[0], raw_xy[-1]):
            raw_xy = raw_xy[:-1]
        closed = bool(guide_closed)
    else:
        raw_xy, closed = centerline_from_mask(media_traced,
                                              prune_px=int(20 / max(work_px_um, 1e-3)) or 8,
                                              lumen_rank=lumen_rank,
                                              px_um=work_px_um, min_wall_um=p.min_wall_um,
                                              lumina=[c["mask"] for c in vessels],
                                              arcs=[a["mask"] for a in traceable],
                                              tissue=tis)
    closed = closed and p.close_ring
    xy = resample_centerline(raw_xy, p.arc_step_um / work_px_um,
                             p.smooth_centerline_um / work_px_um, closed)
    _t, base_nrm = tangents_normals(xy, closed)

    depths_um = np.arange(-p.max_depth_um, p.max_depth_um + p.depth_step_um, p.depth_step_um,
                          dtype=np.float32)
    depths_px = depths_um / work_px_um

    say(0.45, "unrolling the wall")
    floor_od = auto_tissue_threshold(od_sum)

    def resample_planes(curve: np.ndarray, normals: np.ndarray):
        """
        The three stain bands the edge finder reads.

        Not the colour band: that is looked at once, at the end, and the
        re-centring passes decide where the wall is without it, so carrying it
        round the loop would sample twenty million points twice for nothing.
        """
        return (sample_band(muscle, curve, normals, depths_px),
                sample_band(collagen, curve, normals, depths_px),
                sample_band(od_sum, curve, normals, depths_px))

    def oriented(curve: np.ndarray, base_normals: np.ndarray):
        """
        Turn the sampling lines the right way round *before* unrolling.

        Negative depth has to mean "towards the lumen" everywhere, including
        past a branch, and it is cheaper to fix the normals than to juggle
        signs downstream.  A ring is settled by its winding and needs nothing
        sampled; only an open arc has to look, and one band answers it.  The
        four bands then get sampled once instead of twice.
        """
        band = None if closed else sample_band(od_sum, curve, base_normals, depths_px)
        s = orient_band(band, curve, depths_um, p, closed)
        return s, base_normals * np.where(s, 1.0, -1.0).astype(np.float32)[:, None]

    side, nrm = oriented(xy, base_nrm)
    band_muscle, band_collagen, band_tissue = resample_planes(xy, nrm)

    inner, outer, tissue_out = find_wall_edges(
        band_muscle, band_collagen, band_tissue, depths_um, floor_od, p, closed)

    # ---- put the sampling curve down the middle of the wall --------------
    #
    # The first curve comes from the lumen outline pushed out by a constant
    # offset, so where the wall thins or thickens it sits off-centre -- on this
    # aorta by up to 20 um, against a media 41 um thick.  Unrolling about a curve
    # that is not the middle of the band makes the distortion lopsided and, at
    # tight bends, pushes one edge past the point where the sampling lines cross.
    # Re-centring on the measured mid-wall line and sampling again fixes both,
    # and costs one extra pass.
    for _ in range(max(0, int(p.recentre_passes))):
        mid = (inner + outer) / 2.0
        if not np.isfinite(mid).any():
            break
        # Re-centre only where the thing measured is plausibly a wall.
        #
        # At a branch the two walls merge and the edge finder reports a "media"
        # hundreds of microns thick; its midpoint is inside the blob, nowhere
        # near a wall.  Following it drags the sampling curve into the junction,
        # which makes the next pass worse still -- on the aortic arch, an
        # unclamped second pass took the folded fraction from 0.3 % to 11 % and
        # inflated the mean media thickness by a quarter.  Clamping the shift to
        # the wall's own scale keeps the correction where it belongs and leaves
        # the junction alone.
        thick_now = outer - inner
        t_med = float(np.nanmedian(thick_now))
        too_thick = ~(np.isfinite(thick_now) & (thick_now < p.max_thickness_factor * t_med))
        too_thin = np.isfinite(thick_now) & (thick_now <= p.min_thickness_factor * t_med)
        sane = np.isfinite(mid) & ~too_thick & ~too_thin

        # Where the curve has come off the wall the midpoint between the edges
        # is meaningless, but the muscle is still there in the sampled profile
        # and can be aimed at directly.  Only the junction is left alone: there
        # the band really is hundreds of microns thick and every target inside
        # it is wrong.
        level = 0.5 * float(np.nanmedian(np.nan_to_num(band_muscle, nan=0.0).max(axis=1)[sane])) \
            if sane.any() else 0.0
        recover = _muscle_centre_um(band_muscle, depths_um, p.recover_reach_um, level)
        shift_um = np.where(sane, mid, np.nan)
        shift_um = np.where(~sane & ~too_thick & np.isfinite(recover), recover, shift_um)
        shift_um = _fill_nan_1d(shift_um)
        shift_um = np.clip(shift_um, -1.5 * t_med, 1.5 * t_med)
        win = max(3, int(round(p.boundary_smooth_um / max(p.arc_step_um, 1e-6))))
        shift_um = _despike(shift_um, win, 1.5 * t_med,
                            max(1.0, win / 4.0), closed)
        shift = shift_um / work_px_um
        moved = xy + nrm * shift[:, None]
        try:
            # Lighter smoothing than the first pass.  That curve came from a
            # pixel-stepped outline and needed heavy filtering; this one is built
            # from boundary traces that have already been despiked, and smoothing
            # it as hard would pull it back off the wall it was just fitted to.
            xy = resample_centerline(moved, p.arc_step_um / work_px_um,
                                     p.recentre_smooth_um / work_px_um, closed)
        except Exception:
            break
        # Re-centring re-derives the curve, so it has a different number of
        # points; the orientation is decided again on the new one rather than
        # carried over.
        _t, base_nrm = tangents_normals(xy, closed)
        side, nrm = oriented(xy, base_nrm)
        band_muscle, band_collagen, band_tissue = resample_planes(xy, nrm)
        inner, outer, tissue_out = find_wall_edges(
            band_muscle, band_collagen, band_tissue, depths_um, floor_od, p, closed)

    band_rgb = sample_band(img, xy, nrm, depths_px)

    # ---- fold check ------------------------------------------------------
    # `curvature` is defined against the unflipped normal, so wherever the
    # sampling line was turned round the depth axis turns with it.
    smooth_pts = max(3, int(round(8.0 / p.arc_step_um)))
    jac = stretch_factor(xy, depths_um, work_px_um, closed, smooth_pts=smooth_pts)
    if not side.all():
        jac_neg = stretch_factor(xy, -depths_um, work_px_um, closed, smooth_pts=smooth_pts)
        jac = np.where(side[:, None], jac, jac_neg)
    folded = jac <= 0.15
    for band in (band_rgb,):
        band[folded] = np.nan
    band_muscle = np.where(folded, np.nan, band_muscle)
    band_collagen = np.where(folded, np.nan, band_collagen)
    band_tissue = np.where(folded, np.nan, band_tissue)

    media_raw = (depths_um[None, :] >= inner[:, None]) & (depths_um[None, :] <= outer[:, None])
    media_band = media_raw & ~folded
    thickness = outer - inner
    arc_um = np.arange(len(xy), dtype=np.float64) * p.arc_step_um

    # ---- measurements ---------------------------------------------------
    say(0.65, "measuring")
    dd = float(p.depth_step_um)
    ds = float(p.arc_step_um)

    jac_pos = np.clip(jac, 0.0, None)

    def line_integral(band: np.ndarray, where: np.ndarray) -> np.ndarray:
        """
        Integrate a concentration through the wall: OD x microns.

        Weighted by the unrolling Jacobian, because a strip of the straightened
        image does not correspond to a constant area of the section: on the
        convex side of a bend one row of samples covers more tissue than on the
        concave side.  Summing the raw samples over-counts the outside of every
        curve, which for a vessel means over-counting the adventitial half.
        """
        v = np.where(where & np.isfinite(band), band, 0.0)
        return (v * jac_pos).sum(axis=1) * dd

    def area_per_length(where: np.ndarray) -> np.ndarray:
        """True cross-sectional area per unit length of the sampling curve."""
        return (np.where(where, jac_pos, 0.0)).sum(axis=1) * dd

    # ---- what counts as "blue" inside the muscle layer -------------------
    #
    # The default asks a question with no free parameter: is this pixel *more
    # collagen than muscle*?  An absolute threshold cannot be used to compare
    # sections when it is derived from each section separately -- and it has to
    # be, because staining intensity varies between slides.  Worse, collagen
    # inside the media is unimodal, so Otsu has nothing to lock onto: on one
    # thoracic section, re-centring the sampling curve moved the collagen
    # percentiles by under 0.02 but moved Otsu from 0.778 to 0.632, and with it
    # the reported fibrosis from 9 % to 34 %.  A per-pixel comparison of the two
    # stains has none of that: it is invariant to overall staining intensity and
    # means the same thing on every slide.
    #
    # Set `blue_threshold` to a number for the classical absolute-threshold
    # index; `blue_rule="absolute"` with None reproduces the old Otsu behaviour
    # and its instability.
    blue_thr = p.blue_threshold
    rule = p.blue_rule
    if blue_thr is not None:
        rule = "absolute"
    if rule == "absolute" and blue_thr is None:
        inside = band_collagen[media_band & np.isfinite(band_collagen)]
        if inside.size > 500:
            try:
                blue_thr = float(threshold_otsu(inside))
            except Exception:
                blue_thr = float(np.percentile(inside, 80))
            blue_thr = max(blue_thr, float(np.percentile(inside, 60)), 0.02)
        else:
            blue_thr = 0.05

    adventitia = (depths_um[None, :] > outer[:, None]) & \
                 (depths_um[None, :] <= tissue_out[:, None])

    red_line = line_integral(band_muscle, media_band)
    blue_line = line_integral(band_collagen, media_band)
    blue_adv_line = line_integral(band_collagen, adventitia)

    # Collagen normalised to muscle.  Dividing by a muscle integral that is
    # essentially zero does not give a large ratio, it gives an undefined one:
    # there is no media at that position to integrate through.  The floor is a
    # fraction of the section's own muscle level, so it carries no absolute
    # units and travels between stains and exposures.
    _red_ok = red_line[np.isfinite(red_line) & (red_line > 0)]
    red_floor = 0.05 * float(np.median(_red_ok)) if _red_ok.size else 0.0
    _stain = blue_line + red_line
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_line = np.where(red_line > max(red_floor, 1e-9),
                              blue_line / red_line, np.nan)
        frac_line = np.where(_stain > max(red_floor, 1e-9),
                             blue_line / _stain, np.nan)
    if rule == "absolute":
        blue_pos = (band_collagen > blue_thr) & media_band & np.isfinite(band_collagen)
    else:
        blue_pos = (band_collagen > band_muscle) & media_band \
            & np.isfinite(band_collagen) & np.isfinite(band_muscle)
    media_area = area_per_length(media_band)          # um^2 per um of wall
    blue_area = (np.where(blue_pos, jac_pos, 0.0)).sum(axis=1) * dd
    blue_area_frac = np.divide(blue_area, media_area,
                               out=np.zeros_like(media_area),
                               where=media_area > 1e-9)

    # Where the measured band is far thicker than the rest of the vessel it is
    # not a wall: at a branch the two walls merge and the edge finder follows the
    # junction.  Those positions are flagged rather than dropped -- a genuinely
    # thickened wall would look the same to this test, and which of the two it is
    # is a question about the specimen.  On the aortic arch the flagged span
    # carries a fifteen-fold spike in collagen per unit length.
    t_med_final = float(np.nanmedian(thickness))
    too_thick = ~(np.isfinite(thickness) & (thickness < p.max_thickness_factor * t_med_final))

    # And the same judgement the other way round.  A stretch measuring a few
    # microns of media is not a thin wall, it is no wall: either the curve is
    # beside it or there is a real gap in the band.  Counting it kept the
    # length in the denominator of every per-length readout while contributing
    # no numerator -- on the aortic arch a quarter of the traced length, which
    # was reading collagen per mm a third low.  Flagged rather than silently
    # dropped, and reported next to the branch fraction.
    too_thin = ~(np.isfinite(thickness) & (thickness > p.min_thickness_factor * t_med_final))

    # Where the mosaic has no data, the wall was carried across a hole so the
    # vessel would close.  Those stretches are geometry, not measurement.
    # Asked over the band actually measured, not over a window scaled by the
    # median thickness.  A symmetric +/- t_med window reaches a whole wall's
    # width past each edge into space no measurement depends on, so on a
    # sparsely covered mosaic it reports missing image for a wall that is
    # completely covered -- and it moves when the thickness does, which made
    # the same section lose 10 % of its length to "no image" purely for being
    # measured with a rule that reads the media thicker.  Whether the picture
    # is there is a fact about the section, and it should not depend on that.
    cov_band = sample_band(cover.astype(np.float32), xy, nrm, depths_px, order=0)
    near = media_raw | (np.abs(depths_um) <= 10.0)[None, :]
    covered_frac = np.nanmean(np.where(near, cov_band, np.nan), axis=1)
    no_data = ~(covered_frac > 0.5)
    if no_data.any():
        span = max(1, int(round(10.0 / max(p.arc_step_um, 1e-6))))
        no_data = ndi.binary_dilation(no_data, np.ones(span, bool))

    # Which of the thick stretches are forks, and which are just thick?
    # Build the reference strip from the *plausible* wall: at a junction the
    # measured edges run hundreds of microns out, and a strip drawn from them
    # swallows the branch whole, leaving nothing to find.  Interpolating the
    # edges across the suspect stretches carries the normal wall through the
    # junction, so whatever the branch adds sticks out of it.
    inner_b = _fill_nan_1d(np.where(~too_thick, inner, np.nan))
    outer_b = _fill_nan_1d(np.where(~too_thick, outer, np.nan))
    fork_xy, fork_touch = find_branch_nodes(media, xy, nrm, inner_b, outer_b, work_px_um,
                                            p.branch_min_limb_um, return_touch=True)
    branches = branch_arc_positions(fork_xy, xy, arc_um, work_px_um,
                                    max_distance_um=3.0 * t_med_final)

    # How far along the wall a junction reaches, taken from where the branch's
    # own wall actually touches this one rather than assumed to be a multiple of
    # the thickness.  A branch leaves at a shallow angle and runs alongside its
    # parent for a few hundred microns; a fixed window around a single point
    # either falls short of that or overshoots it, and which of the two is a
    # fact about the specimen rather than something to be guessed.
    near_branch = np.zeros_like(too_thick)
    span = max(3, int(round(t_med_final / max(p.arc_step_um, 1e-6))) | 1)
    for pts in fork_touch:
        idx = np.argmin(
            (pts[:, 0][:, None] - xy[None, :, 0]) ** 2
            + (pts[:, 1][:, None] - xy[None, :, 1]) ** 2, axis=1)
        near = np.zeros_like(near_branch)
        near[np.unique(idx)] = True
        # Close the gaps between touching pixels, then allow the junction the
        # width of one wall on either side of what was actually seen.
        near = ndi.binary_closing(near, np.ones(3 * span, bool))
        near_branch |= ndi.binary_dilation(near, np.ones(span, bool))
    # A fork whose limb never touched the traced band still marks its own place.
    reach = max(4.0 * t_med_final, 60.0)
    for b in branches:
        near_branch |= np.abs(arc_um - b["arc_um"]) <= reach
    # A fork opens gradually: the wall swells into the junction over a few
    # hundred microns and only its middle passes the 4x test.  Once the fork
    # itself has been found, thickness near it is evidence about the junction
    # rather than about the vessel, and a gentler threshold is the right one --
    # on the aortic arch this is the difference between excluding 37 um of a
    # 400 um junction and excluding the junction.  Away from any fork the strict
    # test still stands, because there a thick wall is a finding, not an
    # artefact.
    junction = near_branch & np.isfinite(thickness) & (thickness > 1.5 * t_med_final)
    at_branch = (too_thick | junction) & near_branch & ~no_data
    thickened = too_thick & ~near_branch & ~no_data

    # A branch does not only show as a fork.  Where a side vessel leaves the
    # aorta through the wall rather than beside it, the section cuts its lumen
    # *inside* the media: the wall parts around a tunnel, and what the sampling
    # line measures as media there is the thin roof between the two lumina.
    # The fork search above finds limbs of media and the junction test finds
    # swelling, and an ostium is neither -- it is an opening, so the wall gets
    # thinner, not thicker.  On a middle aorta a branch did exactly this over
    # 80 um: the wall halved (19 um against 38), collagen/muscle rose to four
    # times the vessel's own, no fork was found, and the roof was measured as
    # wall.  It had only ever been excluded by accident, by a "no muscle" test
    # too loose to mean what it said, and tightening that test let it through.
    #
    # So an opening is looked for directly: a stretch where the wall has thinned
    # to under 60 % of its own median for at least one wall thickness, and which
    # somewhere carries collagen at more than twice the vessel's own level.  The
    # thinning gives the extent -- it is continuous across an ostium where the
    # collagen spikes are patchy either side of the muscular flow divider -- and
    # the collagen gives the evidence, since a wall that is merely thin is a
    # finding and not a junction.  On that aorta only 33 positions in 2.3 mm
    # were both that thin and that collagen-rich, and all 33 were the ostium.
    #
    # The thinning is measured to where the *tissue* ends, not to the media
    # edge, and that is the whole difference between finding ostia and deleting
    # lesions.  Under the collagen-crossing rule the media edge stops early at
    # every fibrotic patch -- that is its known fault -- so by media thickness a
    # lesion and an ostium look the same: on a thoracic aorta a focal patch read
    # 0.44 of normal, *thinner* than the branch's 0.50, and was excluded as an
    # opening.  But a lesion is still solid wall all the way through and an
    # ostium is a hole, and the tissue extent says which: 0.81 of normal at the
    # patch, 0.54 at the ostium.  It also means the answer does not depend on
    # which edge rule the section was measured by.
    tis_thick = tissue_out - inner
    _tref = np.isfinite(tis_thick) & ~no_data & ~too_thick
    t_tis = float(np.nanmedian(tis_thick[_tref])) if _tref.any() else np.nan
    thin = np.isfinite(tis_thick) & (tis_thick < 0.6 * t_tis) & ~no_data \
        if np.isfinite(t_tis) else np.zeros_like(no_data)
    # Bridge the odd position where the roof briefly reads thicker, then keep
    # only thinnings as long as the wall is thick.
    bridge = max(1, int(round(0.25 * t_med_final / max(p.arc_step_um, 1e-6))))
    thin = ndi.binary_closing(thin, np.ones(2 * bridge + 1, bool)) & np.isfinite(tis_thick)
    thin = _keep_long_runs(thin, t_med_final / max(p.arc_step_um, 1e-6), closed)
    r_own = float(np.nanmedian(ratio_line[np.isfinite(ratio_line) & ~too_thick & ~no_data])) \
        if np.isfinite(ratio_line).any() else np.nan
    opening = np.zeros_like(thin)
    if thin.any() and np.isfinite(r_own) and r_own > 0:
        # A ring's seam is arbitrary: roll so an opening across it is one run.
        shift = int(np.argmin(thin)) if (closed and thin[0] and thin[-1]
                                         and not thin.all()) else 0
        lab, k = ndi.label(np.roll(thin, -shift))
        rr = np.roll(ratio_line, -shift)
        found = np.zeros_like(thin)
        for j in range(1, k + 1):
            run = lab == j
            if np.nanmax(np.where(run, rr, np.nan)) > 2.0 * r_own:
                found |= run
        opening = np.roll(found, shift)

    # The opening is only the middle of the junction.  Either side of the
    # tunnel the branch's own walls merge into the aorta's, and there the wall
    # reads *thick* -- on that middle aorta +26 % on one flank and up to +63 %
    # on the other, both inflating the thickness just outside what had been
    # excluded.  So the junction is grown out from the opening for as long as
    # the wall stays abnormal (tissue more than 25 % off its own median either
    # way), across the brief normal-looking crossing where it swings from thin
    # to thick, and never more than two wall thicknesses each side -- far
    # enough to take the flanks, not so far that a junction can swallow a
    # stretch of genuinely thickened wall next to it, which is a finding.
    if opening.any() and np.isfinite(t_tis) and t_tis > 0:
        wall_n = max(1, int(round(t_med_final / max(p.arc_step_um, 1e-6))))
        abnormal = np.isfinite(tis_thick) & (np.abs(tis_thick / t_tis - 1.0) > 0.25)
        zone = ndi.binary_closing(abnormal | opening, np.ones(2 * wall_n + 1, bool)) | opening
        reach = ndi.binary_dilation(opening, np.ones(4 * wall_n + 1, bool))
        lab, k = ndi.label(zone & reach)
        grown = np.zeros_like(opening)
        for j in range(1, k + 1):
            run = lab == j
            if (run & opening).any():
                grown |= run
        opening = grown
    at_branch = at_branch | (opening & ~no_data)

    # `side` is a vote, not a measurement, and it is free to vote wrong without
    # a branch anywhere nearby: on a stretch where both directions carry some
    # tissue -- a fold, a vessel pressed against its neighbour, a smear of
    # stain -- the near/far optical density comparison can settle on the wrong
    # answer for a couple hundred microns and the 250 um median filter is wide
    # enough to hold that wrong answer steady rather than average it out. A
    # flip a real branch caused has `near_branch` next to it, because that is
    # what the branch search was built from; a flip with no branch nearby has
    # nothing else vouching for it, and unlike a branch the wall does not
    # actually change identity there, so inner and outer are swapped on the
    # vessel's own media rather than correctly relabelled on a different one.
    # Trusting it anyway is what an upside-down stretch of unrolled wall is.
    majority_side = bool(np.mean(side.astype(np.float64)) >= 0.5)
    flip_unsupported = (side != majority_side) & ~near_branch
    # The media/adventitia boundary is defined here by collagen overtaking
    # muscle.  A stretch whose whole measured "media" is already
    # collagen-dominant therefore contains no media by that same definition:
    # it is adventitia, or the collagen wedge at the mouth of a branch, where
    # the trace runs along the interface and the sampling line catches almost
    # nothing else.  On the arch sec2 ostium that reported a local ratio of 13
    # against a vessel whose own is 0.34.
    #
    # This is not a threshold on fibrosis.  Collagen laid down between muscle
    # cells still leaves muscle dominant through the band; a stretch that fails
    # this test has no muscle in it at all, which is a different statement.
    #
    # That last sentence was the intent and the code never enforced it.  The
    # test was collagen/muscle > 1, which is not "no muscle": on a middle aorta
    # whose own wall runs at 0.67 with a 90th percentile of 0.90, ordinary
    # fibrotic media crossed 1.0 in 26 places, every one shorter than the wall
    # is thick and eleven of them one or two microns long, and each was cut out
    # of the wall.  Those stretches still carried 72 % of the section's normal
    # muscle concentration.  The claim is about muscle, so it is now tested on
    # muscle: the band has to be collagen-dominant *and* have run out of
    # muscle -- below the same fraction of this section's own level that the
    # muscle-end rule uses for "ran out" -- and stay that way for at least one
    # wall thickness, since a media does not vanish and come back on a scale
    # smaller than its own depth.  The ostium it was written for, at 13 against
    # 0.34 over hundreds of microns, still qualifies on all three counts.
    musc_conc = np.divide(red_line, np.maximum(thickness, 1e-6),
                          out=np.full_like(red_line, np.nan), where=np.isfinite(thickness))
    _ref = np.isfinite(musc_conc) & ~too_thick & ~too_thin & ~no_data
    musc_ref = float(np.nanmedian(musc_conc[_ref])) if _ref.any() else 0.0
    no_muscle = np.isfinite(musc_conc) & (
        musc_conc < float(getattr(p, "muscle_end_fraction", 0.25)) * musc_ref)
    collagen_dominated = (np.isfinite(ratio_line) & (ratio_line > 1.0) & no_muscle
                          & ~no_data & ~too_thick)
    collagen_dominated = _keep_long_runs(
        collagen_dominated, t_med_final / max(p.arc_step_um, 1e-6), closed)
    # ...but only under the rule that reasoning belongs to.  With the edge set
    # by where muscle ends, the band was drawn without consulting collagen at
    # all, so a band whose collagen outweighs its muscle is a fibrotic stretch
    # of media rather than a stretch containing none -- dropping it would throw
    # away the strongest lesions on the section, which is the whole reason the
    # other rule needed replacing.  It stays flagged, and stops excluding.  The
    # case it guarded against, a trace running along the adventitial interface,
    # still has no muscle under it, so it comes out of the muscle-end walk as a
    # band a micron or two thick and is excluded by `too_thin` as before.
    edge_rule = str(getattr(p, "media_edge_rule", "muscle-end"))
    disqualifying = collagen_dominated if edge_rule != "muscle-end" \
        else np.zeros_like(collagen_dominated)
    off_wall = too_thin & ~no_data & ~too_thick & ~disqualifying
    # What the totals may be taken over: wall, and not a junction, a gap in the
    # band, or a stretch with no image behind it.  `at_branch` has to appear
    # here explicitly -- flagging a junction and then counting it anyway was
    # the state of things before, and it made the reported branch fraction a
    # label rather than a decision.  `flip_unsupported` for the same reason:
    # an unsupported flip is measured wall, not a gap, so it would otherwise
    # sail straight into the totals upside down.
    wall_like = ~too_thick & ~too_thin & ~disqualifying & ~no_data & ~at_branch \
        & ~flip_unsupported

    # Cut the wall where it stops being one continuous piece.  A branch is not a
    # place where the wall carries on: the traced path goes into the junction and
    # comes out on the other wall, so distance measured straight through it joins
    # two stretches that are not neighbours, and the depth axis turns over with
    # the path.  Cutting there makes the discontinuity explicit instead of
    # drawing a curve across it.
    turned = np.zeros_like(at_branch)
    if not side.all():
        turned[1:] = side[1:] != side[:-1]
        turned = ndi.binary_dilation(turned, np.ones(max(3, int(round(
            20.0 / max(p.arc_step_um, 1e-6)))), bool))
    # An unsupported flip is cut over its whole span, not just the 20 um at its
    # edges: `turned` on its own only marks where the vote *changes*, which
    # leaves the upside-down interior between one flip and the next undrawn as
    # a discontinuity but not excluded from it.
    turned = turned | flip_unsupported
    cut = at_branch | no_data | turned | off_wall | disqualifying
    seg_lab, n_seg = ndi.label(~cut)
    segment = np.where(cut, -1, seg_lab - 1)

    beyond_um, beyond_fill = muscle_beyond_edge(band_muscle, depths_um, outer, p)
    var_coarse, var_lamellar = radial_variation(band_muscle, depths_um, inner, outer, p)
    lam_spacing, lam_disorder, edge_width = lamellar_geometry(
        band_muscle, depths_um, inner, outer, p)

    # What the totals are taken over.  The cumulative curves follow the same
    # decision, so the curve ends at the reported total instead of climbing
    # through a stretch the totals leave out.
    keep = wall_like if p.exclude_non_wall else ~no_data
    if not keep.any():
        keep = np.ones_like(wall_like, dtype=bool)
    counted = lambda v: np.where(keep & np.isfinite(v), v, 0.0)

    profile = pd.DataFrame({
        "arc_um": arc_um,
        "wall_like": wall_like,
        "at_branch": at_branch,
        "thickened": thickened,
        "off_wall": off_wall,
        "collagen_dominated": collagen_dominated,
        "no_data": no_data,
        "turned_over": turned,
        "flip_unsupported": flip_unsupported,
        "segment": segment,
        "x_px": xy[:, 0], "y_px": xy[:, 1],
        "media_thickness_um": thickness,
        "media_area_per_um": media_area,
        "stretch_min": np.nanmin(np.where(media_band, jac, np.nan), axis=1),
        "stretch_max": np.nanmax(np.where(media_band, jac, np.nan), axis=1),
        "inner_offset_um": inner, "outer_offset_um": outer,
        "tissue_outer_offset_um": tissue_out,
        "adventitia_thickness_um": tissue_out - outer,
        # Muscle found past the media edge, and how solid it is out there.
        # Descriptive: neither figure moves a boundary or enters a total.
        "muscle_beyond_edge_um": beyond_um,
        "muscle_beyond_edge_fill": beyond_fill,
        # How evenly the muscle is laid down from lumen to outer edge, split
        # from the lamellar banding it would otherwise be swamped by.
        "muscle_variation": var_coarse,
        "lamellar_contrast": var_lamellar,
        # Where the layers sit, and how sharply this section resolves them.
        "lamellar_spacing_um": lam_spacing,
        "lamellar_disorder": lam_disorder,
        "edge_width_um": edge_width,
        "muscle_od_um": red_line,          # integrated muscle stain per unit length
        "collagen_od_um": blue_line,       # integrated collagen stain per unit length
        "collagen_adventitia_od_um": blue_adv_line,
        "collagen_fraction_of_stain": frac_line,
        # Collagen normalised to muscle.  Both are integrated through the same
        # thickness, so the thickness divides out -- which is the whole point.
        "collagen_to_muscle": ratio_line,
        "collagen_area_fraction": blue_area_frac,
        "collagen_mean_conc": blue_line / np.maximum(thickness, 1e-6),
        "muscle_mean_conc": red_line / np.maximum(thickness, 1e-6),
    })
    profile["arc_fraction"] = profile.arc_um / max(profile.arc_um.iloc[-1], 1e-9)
    profile["counted"] = keep
    profile["cum_collagen_od_um2"] = np.cumsum(counted(blue_line)) * ds
    profile["cum_muscle_od_um2"] = np.cumsum(counted(red_line)) * ds
    profile["cum_media_area_um2"] = np.cumsum(counted(media_area)) * ds
    profile["cum_collagen_area_um2"] = np.cumsum(counted(blue_area)) * ds
    profile["cum_collagen_od_um2_all"] = np.cumsum(np.nan_to_num(blue_line)) * ds
    profile["cum_length_um"] = np.cumsum(keep) * ds

    # ---- depth profiles -------------------------------------------------
    # Depth measured from the luminal edge of the media, so position 0 means
    # "inner surface" for every sampling line whatever its thickness.
    shifted = depths_um[None, :] - inner[:, None]
    max_um = float(np.nanpercentile(thickness, 98)) + 80.0
    grid = np.arange(-40.0, max_um, p.depth_step_um)
    def regrid(band: np.ndarray) -> np.ndarray:
        out = np.full((band.shape[0], len(grid)), np.nan, dtype=np.float32)
        for i in range(band.shape[0]):
            v = band[i]
            ok = np.isfinite(v)
            if ok.sum() < 4:
                continue
            out[i] = np.interp(grid, shifted[i][ok], v[ok], left=np.nan, right=np.nan)
        return out
    m_abs, c_abs = regrid(band_muscle), regrid(band_collagen)
    rgb_abs = np.stack([regrid(band_rgb[..., c]) for c in range(3)], axis=-1)
    def col_mean(a: np.ndarray) -> np.ndarray:
        """Column means that tolerate a column with no data at all."""
        n = np.isfinite(a).sum(axis=0)
        tot = np.nansum(np.where(np.isfinite(a), a, 0.0), axis=0)
        return np.divide(tot, n, out=np.full(a.shape[1], np.nan, dtype=float), where=n > 0)

    keep_rows = keep
    depth_from_lumen = pd.DataFrame({
        "depth_from_lumen_um": grid,
        "muscle_sd": np.nanstd(m_abs[keep_rows], axis=0),
        "collagen_sd": np.nanstd(c_abs[keep_rows], axis=0),
        "muscle_mean": col_mean(m_abs),
        "collagen_mean": col_mean(c_abs),
        "n_rows": np.isfinite(m_abs).sum(axis=0),
    })

    # ---- relative depth: the wall as a rectangle -------------------------
    #
    # Rescaling each sampling line by its own wall thickness turns the media into
    # a true rectangle: 0 is the luminal surface and 1 the media/adventitia
    # junction everywhere, whatever the wall does locally.  This is the view to
    # compare positions across a vessel or between animals, because "a third of
    # the way through the media" means the same thing at every arc position,
    # which "20 µm from the lumen" does not.  The axis runs a little past both
    # edges so the lumen and the adventitia stay visible as context.
    rel = np.linspace(p.relative_lo, p.relative_hi, p.n_depth_relative, dtype=np.float32)
    in_media = (rel >= 0.0) & (rel <= 1.0)

    def to_relative(band: np.ndarray) -> np.ndarray:
        out = np.full((band.shape[0], len(rel)), np.nan, dtype=np.float32)
        for i in range(band.shape[0]):
            t = thickness[i]
            if not np.isfinite(t) or t <= 1e-6:
                continue
            target = inner[i] + rel * t
            v = band[i]
            ok = np.isfinite(v)
            if ok.sum() < 4:
                continue
            out[i] = np.interp(target, depths_um[ok], v[ok], left=np.nan, right=np.nan)
        return out

    rel_muscle = to_relative(band_muscle)
    rel_collagen = to_relative(band_collagen)
    rel_rgb = np.stack([to_relative(band_rgb[..., c]) for c in range(3)], axis=-1)
    rel_jac = np.clip(to_relative(jac), 0.0, None)

    # `to_relative` rescales every position by its own `inner`/`thickness`
    # whether or not that position is trustworthy -- it has no way to know.
    # A position `cut` above (a junction, an unsupported flip, a gap) is
    # rescaled from edges that are wrong or meaningless, and rescaling does
    # not make a wrong edge more correct, only harder to recognise as wrong:
    # the absolute-depth view still shows the true image behind a bad edge,
    # but the rectangle view shows only where that edge said the wall was.
    # Blanking it here keeps the two views, and the mean profile below that
    # is built from this one, agreeing about which stretches were measured.
    rel_muscle = np.where(cut[:, None], np.nan, rel_muscle)
    rel_collagen = np.where(cut[:, None], np.nan, rel_collagen)
    rel_rgb = np.where(cut[:, None, None], np.nan, rel_rgb)
    rel_jac = np.where(cut[:, None], np.nan, rel_jac)

    # Averaging across the wall is an average over *area*, so each sample counts
    # for as much tissue as the unrolling stretched it to cover.
    def wmean(a: np.ndarray) -> np.ndarray:
        w = np.where(np.isfinite(a) & np.isfinite(rel_jac), rel_jac, 0.0)
        num = np.nansum(np.where(np.isfinite(a), a, 0.0) * w, axis=0)
        den = w.sum(axis=0)
        return np.divide(num, den, out=np.full(len(rel), np.nan), where=den > 1e-9)

    depth_profile_rel = pd.DataFrame({
        "depth_relative": rel,
        "in_media": in_media,
        "muscle_mean": wmean(rel_muscle),
        "collagen_mean": wmean(rel_collagen),
        "muscle_sd": np.nanstd(rel_muscle, axis=0),
        "collagen_sd": np.nanstd(rel_collagen, axis=0),
        "n_rows": np.isfinite(rel_muscle).sum(axis=0),
    })

    # ---- totals ---------------------------------------------------------
    # Totals are taken over the stretches that are actually wall.  On the aortic
    # arch the branch junction is 4.5 % of the traced length and carries enough
    # collagen to lift the per-millimetre figure by 40 %; including it silently
    # would make an arch and a thoracic section incomparable, since only the arch
    # has a branch in the frame.  Set `exclude_non_wall=False` to include it.
    L = float(keep.sum()) * ds
    L_traced = float(arc_um[-1] + ds) if len(arc_um) else 0.0
    tot_blue = float(np.nansum(counted(blue_line)) * ds)
    tot_red = float(np.nansum(counted(red_line)) * ds)
    tot_area = float(np.nansum(counted(media_area)) * ds)
    tot_blue_area = float(np.nansum(counted(blue_area)) * ds)
    totals = {
        "wall_length_um": L,
        "wall_length_mm": L / 1000.0,
        "traced_length_mm": L_traced / 1000.0,
        "excluded_non_wall": bool(p.exclude_non_wall),
        "closed_ring": bool(closed),
        "media_area_um2": tot_area,
        "mean_media_thickness_um": float(np.nanmean(np.where(keep, thickness, np.nan))),
        "sd_media_thickness_um": float(np.nanstd(np.where(keep, thickness, np.nan))),
        "collagen_od_um2_total": tot_blue,
        "muscle_od_um2_total": tot_red,
        # The headline per-length numbers.
        "collagen_od_um_per_um_length": tot_blue / max(L, 1e-9),
        "collagen_od_um2_per_mm_length": tot_blue / max(L / 1000.0, 1e-9),
        "muscle_od_um_per_um_length": tot_red / max(L, 1e-9),
        "collagen_fraction_of_stain": tot_blue / max(tot_blue + tot_red, 1e-9),
        "collagen_to_muscle_ratio": tot_blue / max(tot_red, 1e-9),
        "collagen_area_fraction_media": tot_blue_area / max(tot_area, 1e-9),
        "collagen_area_um2_per_mm_length": tot_blue_area / max(L / 1000.0, 1e-9),
        "collagen_mean_conc_media": tot_blue / max(tot_area, 1e-9),
        "collagen_adventitia_od_um2_total": float(np.nansum(blue_adv_line) * ds),
        "collagen_rule": rule,
        "collagen_threshold": float(blue_thr) if blue_thr is not None else None,
        # Which definition of "the media ends here" produced every thickness and
        # every per-area number above.  The two rules do not measure the same
        # band, so a number from one is not comparable with a number from the
        # other and the run has to say which it was.
        "media_edge_rule": edge_rule,
        "analysis_px_um": float(work_px_um),
        "arc_step_um": float(p.arc_step_um),
        "median_media_thickness_um": t_med_final,
        # Fraction of the traced length where the band is too thick to be a wall.
        # The totals above include it; subtract it deliberately if you mean to.
        "non_wall_length_fraction": float(1.0 - wall_like.mean()),
        "no_data_length_fraction": float(no_data.mean()),
        # Outer media that broke up: muscle carrying on past the edge, in
        # pieces rather than as a solid layer the edge merely stopped short of.
        # A wall with an intact outer media reads ~0 here.
        # The fragmentation readout: muscle unevenness across the wall, at a
        # scale coarser than a lamella.  It describes the media itself, so
        # unlike the two figures below it does not rest on the outer edge
        # having been put in the right place.
        "muscle_variation": float(np.nanmedian(np.where(keep, var_coarse, np.nan))),
        # Its companion, not a second opinion: the banding the split removed.
        # A section that resolves lamellae crisply scores high here and it
        # means nothing is wrong -- it is what makes a raw radial variance
        # unreadable, and it is printed so a coarse figure can be read against
        # the section's own contrast.
        "lamellar_contrast": float(np.nanmedian(np.where(keep, var_lamellar, np.nan))),
        # Distance between successive lamellae.
        "lamellar_spacing_um": float(np.nanmedian(np.where(keep, lam_spacing, np.nan))),
        # How evenly those layers are stacked, and over how much of the wall
        # three of them could be found at all.  The coverage belongs with the
        # figure: on a soft section most of the wall cannot be judged, and a
        # disorder read off a third of the wall is not the same claim.
        "lamellar_disorder": float(np.nanmedian(np.where(keep, lam_disorder, np.nan))),
        "lamellar_coverage": float(np.nanmean(np.where(keep, np.isfinite(lam_disorder), np.nan))),
        # The section's own blur, from the luminal rise -- no lamella involved.
        # Both figures above are attenuated by it, so a layering number from a
        # 4 um section and one from a 2 um section are not comparable.  Read
        # them together or not at all.
        "section_edge_width_um": float(np.nanmedian(np.where(keep, edge_width, np.nan))),
        "median_muscle_beyond_edge_um": float(np.nanmedian(np.where(keep, beyond_um, np.nan))),
        # The discriminating one.  1.0 means whatever lies past the edge is
        # solid, so the edge merely stopped short; below it the outer media is
        # in pieces.  An arch with 15 um of solid muscle past its edge and a
        # middle aorta with 12 um of fragments read alike without this.
        "median_muscle_beyond_edge_fill": float(np.nanmedian(np.where(keep, beyond_fill, np.nan))),
        # A position with no fill measured is not evidence of an intact outer
        # media, so it is left out rather than counted as one.
        "fragmented_length_fraction": float(np.nanmean(np.where(
            keep & np.isfinite(beyond_fill),
            (beyond_um > FRAGMENT_REACH_UM) & (beyond_fill < FRAGMENT_FILL), np.nan))),
        "n_segments": int(n_seg),
        "turned_over_fraction": float(turned.mean()),
        "image_coverage_fraction": float(cover.mean()),
        "r_thickness_vs_collagen_per_length": _safe_pearson(
            thickness[keep], blue_line[keep]),
        "r_thickness_vs_collagen_fraction": _safe_pearson(
            thickness[keep], frac_line[keep]),
        "r_thickness_vs_collagen_to_muscle": _safe_pearson(
            thickness[keep], ratio_line[keep]),
        "branch_length_fraction": float(at_branch.mean()),
        "thickened_length_fraction": float(thickened.mean()),
        "off_wall_length_fraction": float(off_wall.mean()),
        "collagen_dominated_length_fraction": float(collagen_dominated.mean()),
        "n_branches": len(branches),
        "lumen_rank": int(lumen_rank),
        "n_lumina": int(n_lumina),
        "n_vessels": int(n_vessels),
        "guided": bool(guide_um is not None),
        "n_lumina_rejected": int(len(candidates) - n_lumina),
        "collagen_od_um2_per_mm_all_traced": float(
            np.nansum(np.where(np.isfinite(blue_line), blue_line, 0.0)) * ds
            / max(L_traced / 1000.0, 1e-9)),
        # How much the unrolling had to distort the picture.  A value far from 1
        # means the wall bends tightly compared with its thickness there.
        "stretch_p1": float(np.nanpercentile(np.where(media_band, jac, np.nan), 1)),
        "stretch_p99": float(np.nanpercentile(np.where(media_band, jac, np.nan), 99)),
        # Only the wall itself is interesting here: deep samples far outside it
        # fold routinely and mean nothing.
        "folded_fraction": float(folded[media_raw].mean()) if media_raw.any() else 0.0,
    }

    # -- focal lesions, and any hand-placed marks ---------------------------
    # Imported here rather than at the top: `fibrosis` imports this module, and
    # a lesion is the same thing in a wall as in a ventricle, so it is defined
    # once over there and borrowed rather than written twice.
    say(0.92, "looking for lesions")
    from fibrosis import find_lesions as _find_lesions, measure_points as _measure_points

    def _arc_of(x_um: float, y_um: float) -> float:
        """Where along the wall a point sits, in microns of arc."""
        d = np.hypot(xy[:, 0] * work_px_um - x_um, xy[:, 1] * work_px_um - y_um)
        return float(arc_um[int(np.argmin(d))])

    lesion_labels, lesions, lesion_info = (
        (np.zeros(media_traced.shape, np.int32), [], {}) if not find_lesions_too else
        _find_lesions(collagen, muscle, media_traced, work_px_um,
                      min_area_um2=lesion_min_area_um2, k_mad=lesion_k_mad,
                      exclude_at=exclude_lesions_at))
    for lesion in lesions:
        lesion["arc_um"] = _arc_of(lesion["x_um"], lesion["y_um"])
        lesion["arc_mm"] = lesion["arc_um"] / 1000.0
    marks = _measure_points(points_um, collagen, muscle, media_traced, work_px_um,
                            lesion_labels=lesion_labels)
    for mark in marks:
        mark["arc_um"] = _arc_of(mark["x_um"], mark["y_um"])
        mark["arc_mm"] = mark["arc_um"] / 1000.0
        # A mark placed on another vessel, or in the adventitia, has no media
        # under it.  Flagged rather than dropped: a mark that vanishes looks
        # like a click that did not register.
        mark["in_media"] = bool(mark["n_tissue_px"] > 0)

    wall_mm = float(totals.get("wall_length_um", float("nan"))) / 1000.0
    media_um2 = float(totals.get("media_area_um2", float("nan")))
    lesion_area = float(sum(l["area_um2"] for l in lesions))
    totals.update({
        "n_lesions": len(lesions),
        "lesion_area_mm2": lesion_area / 1e6,
        "lesion_area_fraction_media": (lesion_area / media_um2) if media_um2 > 0 else float("nan"),
        "lesions_per_mm_wall": (len(lesions) / wall_mm) if wall_mm > 0 else float("nan"),
        **lesion_info,
    })

    say(0.95, "done")
    return WallResult(
        px_um=float(work_px_um), params=p, closed=bool(closed),
        image_shape=(int(img.shape[0]), int(img.shape[1])),
        centerline_xy=xy, normals_xy=nrm, arc_um=arc_um, depth_um=depths_um,
        straight_rgb=band_rgb, straight_muscle=band_muscle,
        straight_collagen=band_collagen, straight_tissue=band_tissue,
        inner_um=inner, outer_um=outer, tissue_outer_um=tissue_out,
        media_band=media_band, stretch=jac,
        profile=profile, depth_profile=depth_from_lumen,
        depth_profile_rel=depth_profile_rel, totals=totals, stain_matrix=M,
        branches=branches,
        lumina=[{k: v for k, v in c.items() if k != "mask"}
                for c in candidates + pieces],
        abs_rgb=rgb_abs, abs_muscle=m_abs, abs_collagen=c_abs, abs_depth_um=grid,
        rel_rgb=rel_rgb, rel_collagen=rel_collagen, rel_muscle=rel_muscle,
        rel_depth=rel,
        lesions=lesions, lesion_labels=lesion_labels, points=marks,
    )


def analyze_vessels(
    rgb01: np.ndarray,
    px_um: float,
    params: WallParams | None = None,
    roi: np.ndarray | None = None,
    max_vessels: int = 6,
    points_um: np.ndarray | None = None,
    lesion_min_area_um2: float = 2_000.0,
    lesion_k_mad: float = 3.0,
    exclude_lesions_at: np.ndarray | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[WallResult]:
    """
    Measure every vessel in the frame, not just the biggest one.

    A vessel is either an enclosed lumen (:func:`lumen_candidates`) or, where
    the wall does not close, a separate piece of the band (:func:`wall_pieces`).
    Each is straightened and measured on its own terms; the rings come first,
    largest first, then the arcs, and ``totals["lumen_rank"]`` names which row
    is which.

    Both kinds are needed.  Rings alone miss the section whose wall is open --
    incompletely stitched, cut open, or simply not closed at the muscle
    threshold -- and an aortic arch caught at a branch can show two such walls
    at once.  Pieces alone would split one ring that a fold has broken into two
    vessels, which is why the ring test runs first and its pieces are taken out
    of the running.

    Branch vessels cut near their origin often share the parent's lumen rather
    than enclosing their own, and stay part of the parent's band; those are
    reported by :func:`find_branch_nodes` as attachment points on the parent
    instead.
    """
    p = (params or WallParams())
    if not p.keep_all_vessels:
        p = p.copy_with(keep_all_vessels=True)

    # Marks go to every vessel, and each keeps only the ones that land in its
    # own wall -- a mark on vessel two would otherwise be silently attributed
    # to vessel one, which is the first vessel measured and not the one meant.
    kw = dict(points_um=points_um, lesion_min_area_um2=lesion_min_area_um2,
              lesion_k_mad=lesion_k_mad, exclude_lesions_at=exclude_lesions_at)
    first = analyze_wall(rgb01, px_um, p, roi=roi, lumen_rank=0, progress=progress, **kw)
    n = int(first.totals.get("n_vessels", first.totals.get("n_lumina", 1)))
    out = [first]
    for rank in range(1, min(n, max_vessels)):
        if progress:
            progress(rank / max(n, 1), f"vessel {rank + 1} of {n}")
        try:
            out.append(analyze_wall(rgb01, px_um, p, roi=roi, lumen_rank=rank, **kw))
        except Exception:
            # A small lumen may have too little wall around it to straighten;
            # that is not a reason to lose the vessels that did work.
            continue
    return out


# =============================================================================
# 5.  Figures
# =============================================================================


def straight_view(res: WallResult, pad_um: float = 60.0) -> np.ndarray:
    """The unrolled wall as an RGB image in real microns, cropped to the wall."""
    lo = float(np.nanmin(res.inner_um)) - pad_um
    hi = float(np.nanmax(res.outer_um)) + pad_um
    sel = (res.depth_um >= lo) & (res.depth_um <= hi)
    img = res.straight_rgb[:, sel]
    img = np.where(np.isfinite(img), img, 1.0)
    return np.clip(np.transpose(img, (1, 0, 2)), 0, 1)


def rectangular_view(res: WallResult, what: str = "rgb") -> np.ndarray:
    """
    The wall with its thickness normalised away: a true rectangle.

    Rows are depth through the wall (0 = luminal surface, 1 = media/adventitia
    junction) and columns are distance along the vessel.  This is the view for
    comparing *where in the wall* something sits, since a given row means the
    same fraction of the media everywhere -- which a row of the micron-depth
    view does not, because the wall is thicker in some places than others.

    ``what`` is ``'rgb'``, ``'muscle'`` or ``'collagen'``.
    """
    src = {"rgb": res.rel_rgb, "muscle": res.rel_muscle, "collagen": res.rel_collagen}[what]
    if src is None:
        raise ValueError(f"no {what} map on this result")
    arr = np.transpose(src, (1, 0, 2) if src.ndim == 3 else (1, 0))
    if what == "rgb":
        return np.clip(np.where(np.isfinite(arr), arr, 1.0), 0, 1)
    return arr


def _rect_extent(res: WallResult) -> list[float]:
    r = res.rel_depth
    return [0.0, float(res.arc_um[-1]) / 1000.0, float(r[-1]), float(r[0])]


# One type scale for the whole figure.  Panels that disagree about what a title
# or a tick looks like read as separate figures pasted together.
# The traced wall, drawn to be looked past rather than at.  A quarter-point
# white dash sits on the section without covering it, which is what scoring
# fibrosis by eye needs; the arc-length rainbow it replaced was prettier and
# was painted straight over the tissue being judged.
TRACE_COLOUR = "white"
TRACE_LW = 0.25
TRACE_DASH = (0, (4, 3))

FS_TITLE = 10.5
FS_LABEL = 9.5
FS_NOTE = 8.5


def _rotate_frame(img: np.ndarray, points: list[np.ndarray], angle_deg: float
                  ) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Turn the map to whatever angle reads best, and carry every set of pixel
    points through the same rotation so what is drawn from them still lands on
    the anatomy.

    The canvas grows to fit the whole rotated image rather than cropping the
    corners -- the crop that matters happens afterwards, in `_map_layout`,
    once the vessel's own extent is known in the rotated frame.
    """
    if abs(angle_deg) < 1e-6:
        return img, points
    from scipy import ndimage

    h0, w0 = img.shape[:2]
    cval = 1.0 if np.issubdtype(img.dtype, np.floating) else 255
    rotated = ndimage.rotate(img, angle_deg, reshape=True, order=1,
                             mode="constant", cval=cval)
    h1, w1 = rotated.shape[:2]
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    cx0, cy0 = (w0 - 1) / 2.0, (h0 - 1) / 2.0
    cx1, cy1 = (w1 - 1) / 2.0, (h1 - 1) / 2.0
    out = []
    for pts in points:
        pts = np.asarray(pts, float)
        if pts.size == 0:
            out.append(pts.reshape(-1, 2))
            continue
        dx, dy = pts[:, 0] - cx0, pts[:, 1] - cy0
        rx = dx * c + dy * s
        ry = -dx * s + dy * c
        out.append(np.stack([rx + cx1, ry + cy1], axis=1))
    return rotated, out


def _map_layout(res: "WallResult", overview_rgb: np.ndarray, rotation_deg: float = 0.0):
    """
    Lay the map down and crop it to the vessel.

    A section standing on end wastes the width of the page and shrinks the
    picture to a sliver; the map has no privileged orientation, so if the vessel
    is taller than it is wide the whole thing is turned a quarter turn first.
    `rotation_deg` turns it further still, on top of that automatic quarter
    turn rather than instead of it, so the number in the box always means "how
    far from what you're already looking at."  Returns the image, the centre
    line, the branch marks and the crop box, all in the same (possibly
    rotated) frame.
    """
    sc = overview_rgb.shape[1] / float(res.image_shape[1])
    xy = np.asarray(res.centerline_xy, float) * sc
    branches = getattr(res, "branches", None) or []
    bxy = (np.array([[b["x_px"], b["y_px"]] for b in branches], float) * sc
           if branches else np.zeros((0, 2)))
    lum = getattr(res, "lumina", None) or []
    lxy = (np.array([[c["x_px"], c["y_px"]] for c in lum], float) * sc
           if lum else np.zeros((0, 2)))
    # Lesion centroids ride along with everything else, so they get the same
    # quarter turn, the same rotation and the same crop.  Mapping them by hand
    # afterwards is how a mark ends up somewhere the lesion is not -- which is
    # exactly what happened the first time.
    les = getattr(res, "lesions", None) or []
    exy = (np.array([[q["x_um"] / max(res.px_um, 1e-9),
                      q["y_um"] / max(res.px_um, 1e-9)] for q in les], float) * sc
           if les else np.zeros((0, 2)))

    if float(np.ptp(xy[:, 1])) > float(np.ptp(xy[:, 0])):
        w = overview_rgb.shape[1]
        overview_rgb = np.ascontiguousarray(np.rot90(overview_rgb, 1))
        turn = lambda q: np.stack([q[:, 1], (w - 1.0) - q[:, 0]], axis=1)
        xy = turn(xy)
        bxy = turn(bxy) if len(bxy) else bxy
        lxy = turn(lxy) if len(lxy) else lxy
        exy = turn(exy) if len(exy) else exy

    if abs(rotation_deg) > 1e-6:
        overview_rgb, (xy, bxy, lxy, exy) = _rotate_frame(
            overview_rgb, [xy, bxy, lxy, exy], rotation_deg)

    pad = max(float(np.nanmax(res.tissue_outer_um)) / max(res.px_um, 1e-9) * sc * 1.6,
              0.06 * max(overview_rgb.shape[:2]))
    x0 = max(0.0, xy[:, 0].min() - pad)
    x1 = min(overview_rgb.shape[1] - 1.0, xy[:, 0].max() + pad)
    y0 = max(0.0, xy[:, 1].min() - pad)
    y1 = min(overview_rgb.shape[0] - 1.0, xy[:, 1].max() + pad)
    return overview_rgb, xy, bxy, lxy, exy, (x0, x1, y0, y1)


def draw_traced_wall(
    ax,
    res: WallResult,
    overview_rgb: np.ndarray,
    tick_mm: float = 1.0,
    font_scale: float = 1.0,
    rotation_deg: float = 0.0,
    svg: bool = False,
    show_trace: bool = True,
    stain_gain: Sequence[float] = (1.0, 1.0, 1.0),
) -> None:
    """
    Show the section with the traced wall drawn on it, marked where the
    straightened strip starts and stops.

    The line is drawn thin, white and dashed, and dotted at whole millimetres
    with the same numbers as the panels below.  Thin and dashed because the
    picture underneath is the thing being judged when fibrosis is scored by
    eye, and a fat opaque curve lying along the wall hides the very tissue the
    score is about; ``show_trace=False`` takes it away entirely.  The numbered
    marks stay either way, so a feature at 1.7 mm can still be found.  Without that, the unrolled wall is a picture with no way back: a
    feature at 1.7 mm is somewhere on the vessel and you cannot say where.
    """

    # Ticks read as the smallest thing on the page, but only just: a size
    # that's a full step down from the type everything else uses looks like a
    # different figure's leftovers. One point smaller is enough to tell them
    # apart without breaking the resemblance.
    fs_tick = FS_TITLE * font_scale - 1
    fs_note = FS_NOTE * font_scale

    # Crop to the vessel.  A section is mostly empty slide, and letterboxing the
    # map to the full frame leaves the part anyone needs to read too small.
    img, xy, _bxy, _lxy, exy, (x0, x1, y0, y1) = _map_layout(res, overview_rgb, rotation_deg)
    ax.imshow(restain(img, res.stain_matrix, stain_gain))
    ax.set_axis_off()
    arc_mm = res.arc_um / 1000.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)   # imshow puts the origin at the top

    # One stroke colour, so the raster and the vector are now the same drawing:
    # the SVG used to trade the arc-length rainbow for a single selectable
    # <path>, and with a plain dash there is nothing left to trade.
    if show_trace:
        ax.plot(xy[:, 0], xy[:, 1], color=TRACE_COLOUR, linewidth=TRACE_LW,
                linestyle=TRACE_DASH, solid_joinstyle="round",
                solid_capstyle="round")

    # Whole-millimetre marks, so a position in the strip can be found again here.
    if tick_mm > 0:
        marks = np.arange(0.0, arc_mm[-1] + 1e-9, tick_mm)
        idx = np.searchsorted(arc_mm, marks)
        idx = np.clip(idx, 0, len(xy) - 1)
        for m, i in zip(marks, idx):
            ax.plot(xy[i, 0], xy[i, 1], "o", ms=4.5, mfc="white", mec="k", mew=0.8, zorder=5)
            ax.annotate(f"{m:g}", (xy[i, 0], xy[i, 1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=fs_tick, color="k", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # Focal lesions, pointed at rather than drawn over: an outline traces the
    # very tissue the score is about, and this panel exists to be looked
    # through.  The shapes are on the editor canvas, where there is room.
    # Sized off the type scale rather than fixed in points, and sized to
    # survive being shown small: the app can display this panel under 200 px
    # wide, where a mark that looked right in a full-page export shrinks to a
    # couple of screen pixels and stops being findable at all.
    arrow_ms = 7.0 * font_scale
    for lx, ly in exy:
        ax.plot(lx, ly - 3.4 * arrow_ms, marker="v", ms=arrow_ms, mfc="white",
                mec="#1a1a1a", mew=1.1, zorder=8, linestyle="none")

    # Start and end in the same white-and-black as the millimetre marks.  A
    # green dot and a red square told you which end was which by hue, and hue
    # on top of a trichrome is the one thing that cannot be spared: red and
    # green markers sit on a red-and-blue section and read as tissue.  Shape
    # carries it instead -- circle for the start, square for the end -- and
    # the labels say so in words.
    ax.plot(*xy[0], marker="o", ms=7, mfc="white", mec="k", mew=1.0, zorder=7)
    ax.plot(*xy[-1], marker="s", ms=6.5, mfc="white", mec="k", mew=1.0, zorder=7)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="#555", alpha=0.9)
    if res.closed:
        ax.annotate("start / end (the ring is cut here)", xy[0],
                    textcoords="offset points", xytext=(10, -16),
                    fontsize=fs_note, color="k", bbox=box)
    else:
        ax.annotate("start (0 mm)", xy[0], textcoords="offset points", xytext=(10, -16),
                    fontsize=fs_note, color="k", bbox=box)
        ax.annotate(f"end ({arc_mm[-1]:.2f} mm)", xy[-1], textcoords="offset points",
                    xytext=(10, 10), fontsize=fs_note, color="k", bbox=box)


# `plt_norm` and `arc_colormap` lived here.  They existed to run a rainbow
# along the traced wall by colouring every segment separately; the trace is
# now one thin white dash so the tissue under it stays readable while
# fibrosis is scored by eye, and nothing else ever used them.


def _profile_axis(ax, depth, mean, sd, colour, label, horizontal=True):
    """One depth profile with a +/-1 SD band, drawn sideways."""
    mean = np.asarray(mean, float)
    sd = np.nan_to_num(np.asarray(sd, float))
    ok = np.isfinite(mean)
    if horizontal:
        ax.fill_betweenx(depth[ok], (mean - sd)[ok], (mean + sd)[ok],
                         color=colour, alpha=0.18, lw=0)
        ax.plot(mean[ok], depth[ok], color=colour, lw=1.4, label=label)
    else:
        ax.fill_between(depth[ok], (mean - sd)[ok], (mean + sd)[ok],
                        color=colour, alpha=0.18, lw=0)
        ax.plot(depth[ok], mean[ok], color=colour, lw=1.4, label=label)


def collagen_running_ratio(profile) -> np.ndarray:
    """
    Collagen so far over muscle so far: the running curve the local/running
    panel plots, computed once so the figure and the saved CSV agree exactly
    rather than two copies of this threshold drifting apart.
    """
    cum_red = profile.cum_muscle_od_um2.values
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cum_red > 0.01 * max(float(cum_red[-1]), 1e-9),
                        profile.cum_collagen_od_um2.values / cum_red, np.nan)


def plot_rectangular(
    res: WallResult,
    overview_rgb: np.ndarray | None = None,
    tick_mm: float = 1.0,
    rotation_deg: float = 0.0,
    font_scale: float = 1.0,
    show_grid: bool = True,
    show_spines: bool = True,
    show_trace: bool = True,
    stain_gain: Sequence[float] = (1.0, 1.0, 1.0),
    svg: bool = False,
    row_height: float = 1.7,
    left_width: float = 9.0,
    right_width: float = 2.5,
):
    """
    One page: where the wall is, what it looks like unrolled two ways, what is in
    it across the wall, and how much collagen there is along it.

    The layout is built around two shared axes.  Down the left, every panel
    shares *distance along the wall*, so a step in the running curve lines up
    with the patch of wall that caused it and the numbered marks on the map say
    where on the vessel that is.  Across, each unrolled image is paired with the
    depth profile taken from it, sharing the vertical axis and turned on its
    side, so a band in the image and a bump in the profile sit at the same
    height.  The two pairings are the two ways of measuring depth -- real
    microns from the luminal surface, and fraction of the wall thickness --
    which answer different questions and disagree wherever the wall changes
    thickness.

    The bottom panel is collagen *normalised to muscle*.  Both stains are
    integrated through the same wall, so thickness divides out and a rise means
    a genuinely more collagenous wall rather than a thicker one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fs_title = FS_TITLE * font_scale
    fs_label = FS_LABEL * font_scale
    fs_note = FS_NOTE * font_scale
    # One point smaller than the title -- see draw_traced_wall for why.
    fs_tick = fs_title - 1
    # Bold and one point over the title -- the same kind of step down that
    # ticks take below it, so the three sit in a fixed, evenly-spaced order
    # (panel letter, title, tick) at any font size rather than title and
    # letter closing up or drifting apart as the size changes.
    fs_panel = fs_title + 1

    has_map = overview_rgb is not None

    # The seven panels below the map get one shared height and two column
    # widths -- left for the three unrolled images and the local/running
    # plot, right for the three depth profiles -- rather than a size each,
    # so "make the bottom row taller" is one number instead of four.
    row_height = max(0.5, float(row_height))
    left_width = max(1.0, float(left_width))
    right_width = max(0.5, float(right_width))
    L, R, TOP, BOT = 0.065, 0.955, 0.965, 0.05
    col_w = left_width + right_width
    rows_in = [row_height] * 4
    if has_map:
        _i, _x, _b, _l, _e, (bx0, bx1, by0, by1) = _map_layout(res, overview_rgb, rotation_deg)
        aspect = (bx1 - bx0) / max(by1 - by0, 1e-6)
        map_w = col_w          # the map spans both columns
        heights = [float(np.clip(map_w / max(aspect, 1e-6), 2.4, 7.0))] + rows_in
    else:
        heights = list(rows_in)
    n_rows = len(heights)
    figsize = (col_w / (R - L), sum(heights) + 0.9)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, 2, figure=fig, height_ratios=heights,
                  width_ratios=[left_width, right_width], hspace=0.30, wspace=0.11,
                  left=L, right=R, top=TOP, bottom=BOT)
    row = 0

    if has_map:
        ax_map = fig.add_subplot(gs[0, :])
        draw_traced_wall(ax_map, res, overview_rgb, tick_mm=tick_mm,
                         font_scale=font_scale, rotation_deg=rotation_deg,
                         svg=svg, show_trace=show_trace, stain_gain=stain_gain)
        ax_map.set_title("Traced wall" if show_trace else "Section",
                         fontsize=fs_title)
        row = 1

    prof = res.profile
    arc_mm = prof.arc_um.values / 1000.0
    RED, BLUE = "#d1382f", "#2c7fb8"
    _counted = prof.counted.values.astype(bool) if "counted" in prof else np.ones(len(prof), bool)

    # ---- real microns from the luminal surface --------------------------
    ax_um = fig.add_subplot(gs[row, 0])
    grid = res.abs_depth_um
    # Scale to the wall that is actually being measured: an excluded branch
    # junction is many times thicker and would otherwise set the axis for
    # everything, squashing the wall into a line.
    th_counted = prof.media_thickness_um.values[_counted]
    lo = float(grid[0])
    hi = float(np.nanpercentile(th_counted, 98)) + 70.0
    img_abs = np.where(np.isfinite(res.abs_rgb), res.abs_rgb, 1.0)
    ax_um.imshow(restain(np.transpose(img_abs, (1, 0, 2)), res.stain_matrix, stain_gain),
                 aspect="auto",
                 extent=[0, arc_mm[-1], float(grid[-1]), float(grid[0])])
    ax_um.set_ylim(hi, lo)
    ax_um.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_um.plot(arc_mm, prof.media_thickness_um.values, color="k", lw=0.7, ls="--")
    ax_um.set_ylabel("Depth from the\nluminal surface (µm)", fontsize=fs_label)
    ax_um.set_title("Wall unrolled, real microns", fontsize=fs_title)
    ax_um.grid(False)
    ax_um.tick_params(labelbottom=False, labelsize=fs_tick)

    ax_pum = fig.add_subplot(gs[row, 1], sharey=ax_um)
    d = res.depth_profile
    _profile_axis(ax_pum, d.depth_from_lumen_um.values, d.muscle_mean.values,
                  d.muscle_sd.values, RED, "muscle")
    _profile_axis(ax_pum, d.depth_from_lumen_um.values, d.collagen_mean.values,
                  d.collagen_sd.values, BLUE, "collagen")
    ax_pum.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_pum.axhline(res.totals["mean_media_thickness_um"], color="k", lw=0.8, ls="--")
    ax_pum.set_title("Mean ± SD", fontsize=fs_title)
    ax_pum.legend(fontsize=fs_note, loc="lower right")
    ax_pum.tick_params(labelleft=False, labelsize=fs_tick)
    if show_grid:
        ax_pum.grid(alpha=0.25)

    # ---- thickness-normalised -------------------------------------------
    ax_rel = fig.add_subplot(gs[row + 1, 0], sharex=ax_um)
    ext = _rect_extent(res)
    ax_rel.imshow(restain(rectangular_view(res, "rgb"), res.stain_matrix, stain_gain),
                  aspect="auto", extent=ext)
    ax_rel.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_rel.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_rel.set_ylabel("Depth through the media\n(0 = lumen, 1 = adventitia)", fontsize=fs_label)
    ax_rel.set_title("Wall unrolled, thickness-normalised", fontsize=fs_title)
    ax_rel.grid(False)
    ax_rel.tick_params(labelbottom=False, labelsize=fs_tick)

    ax_prel = fig.add_subplot(gs[row + 1, 1], sharey=ax_rel)
    dr = res.depth_profile_rel
    _profile_axis(ax_prel, dr.depth_relative.values, dr.muscle_mean.values,
                  dr.muscle_sd.values, RED, "muscle")
    _profile_axis(ax_prel, dr.depth_relative.values, dr.collagen_mean.values,
                  dr.collagen_sd.values, BLUE, "collagen")
    ax_prel.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_prel.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_prel.set_title("Mean ± SD", fontsize=fs_title)
    ax_prel.tick_params(labelleft=False, labelsize=fs_tick)
    ax_prel.set_xlabel("Concentration", fontsize=fs_label)
    if show_grid:
        ax_prel.grid(alpha=0.25)

    # ---- collagen, and how thickness relates to it -----------------------
    ax_col = fig.add_subplot(gs[row + 2, 0], sharex=ax_um)
    col = rectangular_view(res, "collagen")
    ax_col.imshow(col, aspect="auto", extent=ext, cmap="Blues",
                  vmin=0, vmax=np.nanpercentile(col, 99))
    ax_col.axhline(0.0, color="k", lw=0.7, ls="--")
    ax_col.axhline(1.0, color="k", lw=0.7, ls="--")
    ax_col.set_ylabel("Depth through the media", fontsize=fs_label)
    ax_col.set_title("Collagen concentration", fontsize=fs_title)
    ax_col.grid(False)
    ax_col.tick_params(labelbottom=False, labelsize=fs_tick)

    # thickness against the normalised readout, on the counted stretches
    ax_reg = fig.add_subplot(gs[row + 2, 1])
    x = prof.media_thickness_um.values[_counted]
    y = prof.collagen_od_um.values[_counted]
    ratio = prof.collagen_to_muscle.values[_counted]
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(ratio)
    x, y, ratio = x[good], y[good], ratio[good]
    if len(x) > 20:
        ax_reg.hexbin(x, ratio, gridsize=28, cmap="Blues", mincnt=1, linewidths=0)
        # Fit on everything counted, but frame the bulk: a handful of junction
        # positions reach several times the wall thickness and would otherwise
        # squeeze the cloud into the corner.
        ax_reg.set_xlim(*np.percentile(x, [1, 97]))
        ax_reg.set_ylim(*np.percentile(ratio, [1, 99]))
        try:
            from scipy import stats as _st
            lr = _st.linregress(x, ratio)
            xs = np.linspace(x.min(), x.max(), 50)
            ax_reg.plot(xs, lr.intercept + lr.slope * xs, "k-", lw=1.2)
            r_len = _st.pearsonr(x, y)[0]
            ax_reg.set_title(f"Collagen ÷ muscle:  r = {lr.rvalue:+.2f}", fontsize=fs_title)
            ax_reg.annotate(
                f"per unit length instead:\nr = {r_len:+.2f}",
                (0.96, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=fs_note, color="#8c1d18")
        except Exception:
            ax_reg.set_title("Thickness vs collagen ÷ muscle", fontsize=fs_title)
    ax_reg.set_ylabel("Collagen ÷ muscle", fontsize=fs_label)
    ax_reg.set_xlabel("Media thickness (µm)", fontsize=fs_label)
    ax_reg.yaxis.set_label_position("right")   # keep it off the image beside it
    ax_reg.yaxis.tick_right()
    ax_reg.tick_params(labelsize=fs_tick)
    if show_grid:
        ax_reg.grid(alpha=0.25)

    # ---- collagen per unit length, normalised to muscle ------------------
    # Raw collagen per unit length tracks thickness almost perfectly (r ~ 0.97),
    # which is arithmetic, not biology: a thicker wall has more of everything.
    # Dividing by muscle integrated through the same wall cancels it.
    # Local and running are now the same quantity, so they share one axis and
    # can be compared by eye; the running curve is the local one integrated,
    # collagen so far over muscle so far, and it settles on the whole-wall
    # value.  A drift away from the dashed line is a real gradient along the
    # vessel rather than a thick patch.
    ax_loc = fig.add_subplot(gs[row + 3, 0], sharex=ax_um)
    # Drawn only where it is counted, so an excluded junction leaves a gap
    # rather than a spike that has to be explained away.
    local = np.where(_counted, prof.collagen_to_muscle.values, np.nan)
    run = collagen_running_ratio(prof)
    ax_loc.plot(arc_mm, local, color=BLUE, lw=0.6, alpha=0.85, label="local")
    ax_loc.plot(arc_mm, run, color="#123c5e", lw=1.8, label="running")
    ax_loc.axhline(res.totals["collagen_to_muscle_ratio"], color="k", lw=0.8, ls="--",
                   label="whole wall")
    if np.isfinite(local).any():
        ax_loc.set_ylim(0, float(np.nanpercentile(local, 98)) * 1.2)
    ax_loc.set_ylabel("Collagen ÷ muscle\nper unit length", fontsize=fs_label)
    ax_loc.set_xlabel("Distance along the wall (mm)", fontsize=fs_label)
    ax_loc.tick_params(labelsize=fs_tick)
    if show_grid:
        ax_loc.grid(alpha=0.25)

    # ---- flagged stretches, on every panel that runs along the wall -----
    # Shading with no key is a colour code only the person who wrote it can
    # read; the legend is built from whichever of these actually appear in
    # this trace; rather than every possible flag every time, so it says
    # exactly what's shaded here rather than what could be.
    along = [ax_um, ax_rel, ax_col, ax_loc]
    flag_handles = []
    for colname, colour, label in (("no_data", "#8a8f98", "no image"),
                                   ("turned_over", "#9b6bd6", "path turns over"),
                                   ("at_branch", "#e08a2e", "branch"),
                                   ("thickened", "#e2574c", "thickened"),
                                   ("off_wall", "#8a8f98", "no wall under the trace"),
                                   ("collagen_dominated", "#3f7fb5", "no muscle in the band")):
        if colname not in prof:
            continue
        flag = prof[colname].values.astype(bool)
        if not flag.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
        for a_i, b_i in zip(edges[::2], edges[1::2]):
            for ax in along:
                ax.axvspan(arc_mm[a_i], arc_mm[min(b_i, len(arc_mm) - 1)],
                           color=colour, alpha=0.16, zorder=0, lw=0)
        flag_handles.append(Patch(color=colour, alpha=0.4, label=label))
    # Where the wall is cut, say so with a hard line: the two sides are not
    # neighbours on the vessel and nothing should be read across the join.
    if "segment" in prof:
        seg = prof.segment.values
        edges = np.flatnonzero(np.diff(seg)) if len(seg) > 1 else []
        if len(edges):
            for e in edges:
                for ax in along:
                    ax.axvline(arc_mm[e], color="#3a3f4a", lw=1.1, alpha=0.85)
            flag_handles.append(Line2D([0], [0], color="#3a3f4a", lw=1.1, label="wall cut here"))
    if tick_mm > 0:
        for m_ in np.arange(0.0, arc_mm[-1] + 1e-9, tick_mm):
            for ax in along:
                ax.axvline(m_, color="k", lw=0.5, alpha=0.25, ls=":")
    ax_um.set_xlim(0, arc_mm[-1])

    # One legend for the whole stack: local/running/whole-wall (from ax_loc's
    # own labelled lines) plus whatever's shaded above, so what a colour or a
    # line means is answered once rather than guessed at on every panel.
    handles, _labels = ax_loc.get_legend_handles_labels()
    ax_loc.legend(handles=handles + flag_handles, fontsize=fs_note,
                 loc="lower right", ncol=4, framealpha=0.85)

    panel_axes = ([ax_map] if has_map else []) + [ax_um, ax_pum, ax_rel, ax_prel,
                                                    ax_col, ax_reg, ax_loc]
    for letter, ax in zip("ABCDEFGH", panel_axes):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=fs_panel,
               fontweight="bold", va="bottom", ha="right")

    if not show_spines:
        # "Off" opens the box rather than removing it: the two sides a
        # reading is actually taken against carry the frame, so a panel with
        # no frame at all reads as unfinished rather than as clean. ax_reg's
        # y-axis was moved to the right above, to keep it off the image
        # beside it, so its open side is the mirror of every other panel's.
        for ax in (ax_um, ax_pum, ax_rel, ax_prel, ax_col, ax_loc):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_reg.spines["top"].set_visible(False)
        ax_reg.spines["left"].set_visible(False)
    return fig


def _savez_fast(path: str | Path, **arrays: np.ndarray) -> None:
    """
    ``np.savez_compressed``, at the cheap end of deflate.

    The maps are a hundred megabytes and numpy gives no way to ask for less
    zlib; at its default level the archive costs 2.0 s to write against 0.8 s
    at level 1, for 39.6 MB against 41.9.  Six percent of a file nobody reads
    twice is not worth a second of every analysis.  Still an ordinary .npz.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for key, arr in arrays.items():
            with z.open(key + ".npy", "w") as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)


def save_results(res: WallResult, out_dir: str | Path, name: str = "wall",
                 overview_rgb: np.ndarray | None = None) -> dict[str, str]:
    """Write the tables and the straightened image."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    # The local/running collagen-to-muscle panel plots a curve derived from
    # columns already here (cum_collagen_od_um2 over cum_muscle_od_um2) but
    # not itself saved anywhere -- added on a copy so the in-memory profile
    # this WallResult carries elsewhere stays exactly what analyze_wall built.
    prof_out = res.profile.copy()
    prof_out["collagen_to_muscle_running"] = collagen_running_ratio(res.profile)
    prof_out.to_csv(out_dir / f"{name}_along_wall.csv", index=False)
    paths["profile"] = str(out_dir / f"{name}_along_wall.csv")
    res.depth_profile.to_csv(out_dir / f"{name}_depth_absolute_um.csv", index=False)
    paths["depth_absolute"] = str(out_dir / f"{name}_depth_absolute_um.csv")
    res.depth_profile_rel.to_csv(out_dir / f"{name}_depth_relative.csv", index=False)
    paths["depth_relative"] = str(out_dir / f"{name}_depth_relative.csv")
    pd.DataFrame([res.totals]).to_csv(out_dir / f"{name}_totals.csv", index=False)
    paths["totals"] = str(out_dir / f"{name}_totals.csv")
    if res.lesions:
        pd.DataFrame(res.lesions).to_csv(out_dir / f"{name}_lesions.csv", index=False)
        paths["lesions"] = str(out_dir / f"{name}_lesions.csv")
    if res.points:
        pd.DataFrame(res.points).to_csv(out_dir / f"{name}_points.csv", index=False)
        paths["points"] = str(out_dir / f"{name}_points.csv")
    plt.imsave(out_dir / f"{name}_straightened.png", straight_view(res))
    paths["straightened"] = str(out_dir / f"{name}_straightened.png")
    plt.imsave(out_dir / f"{name}_rectangular.png", rectangular_view(res, "rgb"))
    paths["rectangular"] = str(out_dir / f"{name}_rectangular.png")
    # The figure goes out as a raster to look at and as vector to edit. The
    # SVG pass is drawn separately -- svg=True trades the map trace's
    # per-segment rainbow for one selectable curve, since a vector backend
    # can only give a gradient like that one <path> per segment -- and saved
    # with real <text> elements rather than matplotlib's default of glyphs
    # traced out as curves, so labels can be restyled for a figure panel
    # without going back through the analysis.
    fig = plot_rectangular(res, overview_rgb=overview_rgb)
    fig.savefig(out_dir / f"{name}_rectangular_channels.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig_svg = plot_rectangular(res, overview_rgb=overview_rgb, svg=True)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig_svg.savefig(out_dir / f"{name}_rectangular_channels.svg", bbox_inches="tight")
    plt.close(fig_svg)
    paths["rectangular_channels"] = str(out_dir / f"{name}_rectangular_channels.png")
    paths["rectangular_channels_svg"] = str(out_dir / f"{name}_rectangular_channels.svg")
    _savez_fast(
        out_dir / f"{name}_maps.npz",
        straight_rgb=res.straight_rgb.astype(np.float16),
        straight_muscle=res.straight_muscle.astype(np.float16),
        straight_collagen=res.straight_collagen.astype(np.float16),
        media_band=res.media_band, depth_um=res.depth_um, arc_um=res.arc_um,
        inner_um=res.inner_um, outer_um=res.outer_um, stretch=res.stretch.astype(np.float16),
        rel_depth=res.rel_depth, rel_muscle=res.rel_muscle.astype(np.float16),
        rel_collagen=res.rel_collagen.astype(np.float16),
        centerline_xy=res.centerline_xy, stain_matrix=res.stain_matrix,
    )
    paths["maps"] = str(out_dir / f"{name}_maps.npz")
    return paths
