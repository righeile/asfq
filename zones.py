#!/usr/bin/env python3
"""
zones.py -- proposing the anatomical zones of a kidney section, unsupervised.

A kidney is three tissues stacked in one section.  Cortex, outer and inner
medulla and papilla have visibly different baseline collagen, so a fibrosis
number over a section that spans several of them mostly reports which ones the
knife caught.  Drawing the boundaries by hand works and is what the region
editor is for; this module proposes them so that the hand-drawing starts from
something rather than from nothing.

**It proposes, it does not decide -- and on this data it proposes badly.**
Checked against the region lists already written into the folder names, the
number of zones it finds matched what the folder said in **5 of 17** sections.
It finds three zones in almost everything, including crops cut entirely from
cortex, and it invented a papilla in six sections that have none.  The reason
is not a bad parameter: uniform cortex has plenty of internal texture variation
(glomeruli against tubules), so a clustering asked "are there three
distinguishable groups of pixels here" answers yes whether or not there is an
anatomical boundary anywhere in the frame.  No amount of tuning fixes a method
that is answering a different question from the one being asked.

What *is* solid is the gradient underneath it.  Coherence measured at 1 um/px,
clear of the section edge, falls monotonically across a whole kidney section --
0.41 at the papilla, 0.34, 0.33, 0.27, 0.26, 0.22 at the cortex.  That is a
real corticomedullary signal, and :func:`texture_guide` returns it as a picture
to draw against.  Drawing the boundary yourself while looking at that map is
the honest version of this feature; the polygons are a starting point and are
labelled as one.
  Everything here comes back as polygons
into the same editor, to be corrected or thrown away.  That is not modesty
about the method, it is the only defensible arrangement: there is no public
dataset of mouse trichrome kidney with cortex/medulla/papilla labels to check a
detector against (the ones that annotate regions -- PrPSeg, NePathTK -- are
human PAS biopsies, and the ones that are mouse -- KPIs 2024 -- annotate
glomeruli, not zones), so a boundary drawn here has never been checked against
a pathologist and must not be presented as though it has.

What it can be checked against is the region list already written into the
folder names ("Pap, Med, Ctx"), and ``evaluate_against_names`` in the tests
does exactly that.  It is a coarse label -- which zones are present, not where
they are -- but it is independent of this code, which is what makes it worth
anything.

The features are structural rather than chromatic, because the thing that
separates these zones is architecture:

*Orientation coherence.*  Medulla and papilla are parallel straight tubules,
vasa recta and collecting ducts all running the same way; cortex is glomeruli
and convoluted tubules, which point everywhere.  The structure tensor measures
exactly this, and it is a property of the geometry rather than of how hard the
section was stained.

*Collagen share of the stain.*  Interstitial matrix increases from cortex
through medulla to papilla in a normal kidney, which is the same gradient that
makes a whole-section number misleading in the first place.

*Nuclear density.*  Cortex is nuclear-dense -- glomerular tufts, tubular
epithelium packed close; papillary interstitium is not.

*Distance from the outside.*  Cortex is the rim of a kidney and papilla the
tip.  This is the one feature that is about where a pixel sits rather than
what it looks like, and it is deliberately weak: a wedge or an off-axis cut
breaks it, and a detector that leaned on it would confidently mislabel exactly
the sections that are hardest to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

import stitch_core as SC
from wall_analysis import rgb_to_od, separate_stains, tissue_mask, auto_tissue_threshold
from fibrosis import FIXED_STAINS

#: The zones this knows about, innermost first.  Order matters: it is the
#: anatomical axis the clusters are ranked along.
KIDNEY_ZONES = ("papilla", "medulla", "cortex")


@dataclass
class ZoneResult:
    px_um: float                      # of the label image
    labels: np.ndarray                # int, 0 = not tissue, 1..k = zone
    names: list[str]                  # names[i-1] is the name of label i
    features: dict[str, np.ndarray]   # the per-pixel feature maps, for looking at
    scores: dict[str, float]          # per-zone mean of the ranking score
    n_zones: int
    confident: bool                   # False when the clusters are not distinct
    note: str = ""


def _assign(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest centre for every row, in chunks so a big section still fits."""
    out = np.empty(len(X), dtype=np.int16)
    for i in range(0, len(X), 200_000):
        block = X[i:i + 200_000]
        d = ((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        out[i:i + 200_000] = d.argmin(axis=1)
    return out


def _fit(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means, restarted a few times and keeping the tightest fit."""
    from scipy.cluster.vq import kmeans2

    if k <= 1:
        return X.mean(axis=0, keepdims=True)
    best, best_rss = None, np.inf
    for s in range(seed, seed + 4):
        try:
            centres, lab = kmeans2(X, k, minit="++", seed=s)
        except Exception:
            continue
        if len(np.unique(lab)) < k:
            continue
        rss = float(((X - centres[lab]) ** 2).sum())
        if rss < best_rss:
            best, best_rss = centres, rss
    return best if best is not None else X.mean(axis=0, keepdims=True)


def _bic(X: np.ndarray, centres: np.ndarray) -> float:
    """
    BIC for a spherical, equal-variance Gaussian mixture -- which is the model
    k-means is fitting whether or not anyone says so.

    Used only to choose *how many* zones, and that choice matters more than it
    sounds: a section cut entirely from cortex is one zone, and a method that
    always returns three will invent two and draw a confident boundary through
    uniform tissue.
    """
    n, d = X.shape
    k = len(centres)
    lab = _assign(X, centres)
    rss = float(((X - centres[lab]) ** 2).sum())
    var = max(rss / max(n * d, 1), 1e-12)
    log_l = -0.5 * n * d * (np.log(2 * np.pi * var) + 1.0)
    n_params = k * d + 1
    return float(-2.0 * log_l + n_params * np.log(max(n, 2)))


def _coherence(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    How consistently the local structure points one way, 0..1.

    The structure tensor's two eigenvalues are the strength of variation along
    and across the dominant direction; their normalised difference is high for
    parallel tubules and near zero where the texture has no direction.  This is
    the feature that separates medulla from cortex, and it costs nothing in
    stain calibration because it is a statement about geometry.
    """
    gy, gx = np.gradient(gray.astype(np.float32))
    jxx = ndi.gaussian_filter(gx * gx, sigma)
    jyy = ndi.gaussian_filter(gy * gy, sigma)
    jxy = ndi.gaussian_filter(gx * gy, sigma)
    tr = jxx + jyy
    root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0))
    return np.where(tr > 1e-12, root / np.maximum(tr, 1e-12), 0.0).astype(np.float32)


def zone_features(
    rgb01: np.ndarray,
    px_um: float,
    work_px_um: float = 4.0,
    window_um: float = 160.0,
    texture_px_um: float = 1.0,
    edge_margin_um: float = 80.0,
    stain_matrix: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """
    The feature maps, plus the tissue mask they are defined on.

    Two resolutions on purpose.  The *zones* are hundreds of microns across and
    are clustered at ``work_px_um``, where that is cheap.  The *texture* that
    distinguishes them is tubules 20-50 um wide, and asking about those at 4 um
    per pixel does not measure them -- it measures the section's outline.
    Measured on this data: at 4 um/px the coherence map lit up along the capsule
    and the central cleft and was flat everywhere else, and the clusters built
    on it carved uniform cortex into three confident blobs.  Computed at
    ``texture_px_um`` instead, and kept ``edge_margin_um`` clear of the tissue
    boundary so the outline's own gradient cannot dominate, the same feature
    falls monotonically from 0.41 at the papilla to 0.22 at the cortex.

    The edge margin is why this needs a whole section rather than a crop: a
    field small enough that the margin eats it has no interior to measure.
    """
    # -- texture, at a resolution where tubules exist ------------------------
    t_scale = px_um / max(texture_px_um, 1e-6)
    t_img = np.clip(SC.resize(np.clip(rgb01, 0, 1), t_scale, "area"), 0, 1) \
        if abs(t_scale - 1.0) > 1e-3 else np.clip(rgb01, 0, 1)
    tpx = px_um / t_scale
    t_od = rgb_to_od(t_img).sum(axis=2)
    t_tis = tissue_mask(t_od, auto_tissue_threshold(t_od),
                        min_area_px=max(1, int(2e5 / tpx ** 2)))
    inner = ndi.binary_erosion(t_tis, np.ones((3, 3)),
                               iterations=max(1, int(edge_margin_um / tpx)))
    coh = _coherence(t_od, sigma=max(1.0, 15.0 / tpx))
    # Averaged over the window using only interior pixels, then divided by how
    # many there were -- otherwise the margin bleeds a spurious zero inwards.
    sig_t = max(1.0, window_um / (2.0 * tpx))
    num = ndi.gaussian_filter(np.where(inner, coh, 0.0).astype(np.float32), sig_t)
    den = ndi.gaussian_filter(inner.astype(np.float32), sig_t)
    coh_t = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)

    # -- everything else, and the clustering grid, at the coarse scale -------
    scale = px_um / max(work_px_um, 1e-6)
    rgb = np.clip(SC.resize(np.clip(rgb01, 0, 1), scale, "area"), 0, 1) \
        if abs(scale - 1.0) > 1e-3 else np.clip(rgb01, 0, 1)
    wpx = px_um / scale

    od = rgb_to_od(rgb)
    od_sum = od.sum(axis=2)
    thr = auto_tissue_threshold(od_sum)
    tis = tissue_mask(od_sum, thr, min_area_px=max(1, int(2e5 / wpx ** 2)))
    conc = separate_stains(od, FIXED_STAINS if stain_matrix is None else stain_matrix)
    muscle, collagen, nuclei = conc[..., 0], conc[..., 1], conc[..., 2]

    sigma = max(1.0, window_um / (2.0 * wpx))
    smooth = lambda a: ndi.gaussian_filter(a.astype(np.float32), sigma)
    total = smooth(collagen) + smooth(muscle)
    col_frac = np.where(total > 1e-6, smooth(collagen) / np.maximum(total, 1e-6), 0.0)

    coh_w = SC.resize(coh_t, wpx and (tpx / wpx), "area")
    coh_w = _match(coh_w, tis.shape)

    feats = {
        "coherence": coh_w.astype(np.float32),
        "collagen_share": col_frac.astype(np.float32),
        "nuclei": smooth(nuclei),
    }
    # `depth` -- distance from the outside of the section -- was tried as a
    # fifth feature and removed.  It is pure geometry: a distance transform of
    # any blob is a ridge down its middle, so it partitions a uniform crop into
    # confident bands that have nothing to do with anatomy, and it was doing
    # much of the work in the version that failed.  Where a pixel sits is a
    # weak prior at best and a wedge or an off-axis cut breaks it entirely.
    return feats, tis, wpx


def _match(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop or pad an array to *shape* -- resize rounding is off by a pixel."""
    out = np.zeros(shape, dtype=a.dtype)
    h, w = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def detect_kidney_zones(
    rgb01: np.ndarray,
    px_um: float,
    n_zones: int | None = None,
    work_px_um: float = 4.0,
    window_um: float = 160.0,
    stain_matrix: np.ndarray | None = None,
    min_zone_frac: float = 0.04,
) -> ZoneResult:
    """
    Propose zones, and say how much to believe them.

    *n_zones* left as None lets the data choose between one, two and three by
    BIC.  That matters more than it sounds: a section cut entirely from cortex
    is one zone, and a method that always returns three will invent two of
    them and put a confident boundary through uniform tissue.

    The clusters are named by ranking them along one anatomical axis -- more
    parallel, more collagen, fewer nuclei, deeper in means closer to the
    papilla -- and the naming is the weakest step here by some way.  When two
    clusters rank too closely to separate, ``confident`` comes back False and
    the names fall back to "zone 1..k", because a wrong anatomical name is
    worse than no name: a boundary you are told is the corticomedullary
    junction gets believed.
    """
    feats, tis, wpx = zone_features(rgb01, px_um, work_px_um, window_um,
                                    stain_matrix=stain_matrix)
    keys = ["coherence", "collagen_share", "nuclei"]
    X = np.stack([feats[k][tis] for k in keys], axis=1).astype(np.float64)
    labels = np.zeros(tis.shape, dtype=np.int16)
    if len(X) < 200:
        return ZoneResult(wpx, labels, [], feats, {}, 0, False,
                          "too little tissue to look for zones")

    # Standardised, or `depth` (0..1) would be shouted down by whichever
    # feature happens to have the widest raw range.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / np.where(sd > 1e-9, sd, 1.0)

    # How many zones there are is decided on *spatially independent* samples:
    # one per window, on a grid at the window spacing.  Every pixel is not a
    # sample -- the features are smoothed over that window, so neighbouring
    # pixels carry the same measurement -- and BIC fed a million correlated
    # rows will pick the largest k on offer every time, whatever the tissue
    # looks like.  Measured here: it chose three zones in sixteen of
    # seventeen sections, including ones cut entirely from cortex.  The honest
    # n is the number of windows the section holds, which is hundreds rather
    # than millions.
    grid = np.zeros(tis.shape, bool)
    step = max(1, int(round(window_um / wpx)))
    grid[::step, ::step] = True
    picks = grid[tis]
    sub = Z[picks] if picks.sum() >= 8 * len(keys) else Z[::max(1, len(Z) // 400)]

    if n_zones:
        best_k, centres = int(n_zones), _fit(Z, int(n_zones))
    else:
        best_k, best_bic = 1, np.inf
        for k in (1, 2, 3):
            c = _fit(sub, k)
            bic = _bic(sub, c)
            if bic < best_bic:
                best_k, best_bic = k, bic
        # Chosen on the grid, fitted on everything: the choice needs
        # independent samples, the centres are better for having all of them.
        centres = _fit(Z, best_k)

    labels[tis] = _assign(Z, centres) + 1

    # Zones are contiguous pieces of anatomy, not speckle.  A majority filter
    # at the window scale removes the salt-and-pepper a per-pixel classifier
    # produces without moving a real boundary, which is far cheaper than the
    # graph cut it approximates.
    r = max(1, int(round(window_um / (2.0 * wpx))))
    if best_k > 1:
        smoothed = np.zeros_like(labels)
        for k in range(1, best_k + 1):
            smoothed = np.where(
                ndi.uniform_filter((labels == k).astype(np.float32), 2 * r + 1)
                > ndi.uniform_filter((smoothed > 0).astype(np.float32), 2 * r + 1),
                k, smoothed)
        votes = np.stack([ndi.uniform_filter((labels == k).astype(np.float32), 2 * r + 1)
                          for k in range(1, best_k + 1)])
        labels = np.where(tis, votes.argmax(axis=0) + 1, 0).astype(np.int16)

    # Drop a zone too small to be anatomy and give its pixels to the nearest
    # surviving one, rather than reporting a 2% sliver as a compartment.
    sizes = {k: int((labels == k).sum()) for k in range(1, best_k + 1)}
    total = max(int(tis.sum()), 1)
    keep = [k for k, n in sizes.items() if n / total >= min_zone_frac]
    if not keep:
        keep = [max(sizes, key=sizes.get)]
    if len(keep) < best_k:
        remap = np.zeros(best_k + 1, dtype=np.int16)
        for new, k in enumerate(keep, start=1):
            remap[k] = new
        drop = labels > 0
        for k in range(1, best_k + 1):
            if k not in keep:
                drop &= (labels != k)
        idx = ndi.distance_transform_edt(~np.isin(labels, keep) & (labels > 0),
                                         return_distances=False, return_indices=True)
        moved = labels[tuple(idx)]
        labels = np.where(np.isin(labels, keep), labels, moved)
        labels = remap[np.clip(labels, 0, best_k)]
        best_k = len(keep)

    # -- naming, the weak step ----------------------------------------------
    # One axis: parallel, collagen-rich, nucleus-poor and deep is papillary;
    # the opposite is cortical.  Ranked rather than thresholded, so it does not
    # depend on absolute values that vary with the stain and the scanner.
    score = {}
    for k in range(1, best_k + 1):
        m = labels == k
        if not m.any():
            score[k] = -np.inf
            continue
        score[k] = float(
            np.mean(feats["coherence"][m]) / max(np.std(feats["coherence"][tis]), 1e-6)
            + np.mean(feats["collagen_share"][m]) / max(np.std(feats["collagen_share"][tis]), 1e-6)
            - np.mean(feats["nuclei"][m]) / max(np.std(feats["nuclei"][tis]), 1e-6))
    order = sorted(range(1, best_k + 1), key=lambda k: -score[k])   # papilla first

    # Distinct enough to name?  Clusters whose scores are within a fraction of
    # the spread are two halves of one tissue, and naming them "papilla" and
    # "medulla" would be inventing a junction.
    vals = np.array([score[k] for k in order], dtype=float)
    spread = float(vals.max() - vals.min()) if len(vals) > 1 else 0.0
    gaps = np.diff(-vals) if len(vals) > 1 else np.array([np.inf])
    confident = bool(len(vals) == 1 or (spread > 0 and gaps.min() > 0.25 * spread))

    if confident and best_k <= len(KIDNEY_ZONES):
        # k zones out of three: with two, the outermost is cortex and the
        # other is medulla -- a two-zone section is far more often cortex and
        # medulla than medulla and papilla.
        chosen = (list(KIDNEY_ZONES) if best_k == 3
                  else ["medulla", "cortex"] if best_k == 2
                  else ["cortex"])
        names_by_label = {k: chosen[i] for i, k in enumerate(order)}
        note = ""
    else:
        names_by_label = {k: f"zone {i + 1}" for i, k in enumerate(order)}
        note = ("the zones are not distinct enough to name with any confidence — "
                "they are numbered, innermost-looking first")

    names = [names_by_label[k] for k in range(1, best_k + 1)]
    return ZoneResult(px_um=wpx, labels=labels, names=names, features=feats,
                      scores={names_by_label[k]: score[k] for k in range(1, best_k + 1)},
                      n_zones=best_k, confident=confident, note=note)


def zone_polygons(res: ZoneResult, simplify_um: float = 40.0,
                  min_area_um2: float = 5e4) -> list[dict[str, Any]]:
    """
    The zones as polygons in microns, ready for the region editor.

    Polygons rather than a mask because that is what the editor speaks, and
    because a boundary you can grab a corner of is a boundary you can correct.
    """
    from skimage.measure import find_contours, approximate_polygon

    out: list[dict[str, Any]] = []
    for k, name in enumerate(res.names, start=1):
        m = res.labels == k
        if not m.any():
            continue
        # Close small gaps first: a contour that threads between two tubules
        # is a hundred vertices of noise, not a boundary.
        m = ndi.binary_closing(m, np.ones((5, 5)))
        m = ndi.binary_fill_holes(m)
        lab, n = ndi.label(m)
        for j in range(1, n + 1):
            piece = lab == j
            if piece.sum() * res.px_um ** 2 < min_area_um2:
                continue
            padded = np.pad(piece.astype(float), 1)
            for contour in find_contours(padded, 0.5):
                poly = approximate_polygon(contour, tolerance=simplify_um / res.px_um)
                if len(poly) < 4:
                    continue
                xy = (poly[:, ::-1] - 1.0) * res.px_um      # (row, col) -> (x, y) microns
                out.append({"name": name, "polygon": xy.tolist()})
    return out


def texture_guide(
    rgb01: np.ndarray,
    px_um: float,
    work_px_um: float = 4.0,
    window_um: float = 160.0,
    stain_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """
    The corticomedullary gradient as a picture to draw against.

    Returns an RGB image at *work_px_um*, blue where the texture looks
    cortical (isotropic, nucleus-dense, little collagen) and red where it looks
    papillary (parallel tubules, more matrix), with the tissue mask applied.

    This is the part of this module that survived checking.  It asserts no
    boundary -- there is no threshold in it and nothing is segmented -- so
    there is nothing here to be wrong about beyond the gradient itself, and
    the gradient is monotonic across a whole kidney section.  Where to put the
    junction is left to whoever is looking at it, which given the state of
    automatic zone detection on this tissue is where it belongs.
    """
    import matplotlib as mpl

    feats, tis, wpx = zone_features(rgb01, px_um, work_px_um, window_um,
                                    stain_matrix=stain_matrix)
    z = lambda a: (a - np.mean(a[tis])) / max(float(np.std(a[tis])), 1e-9)
    score = (z(feats["coherence"]) + z(feats["collagen_share"]) - z(feats["nuclei"])) / 3.0
    lo, hi = np.percentile(score[tis], [2, 98])
    norm = np.clip((score - lo) / max(hi - lo, 1e-9), 0, 1)
    # matplotlib 3.9 removed cm.get_cmap; this spelling works on both.
    rgb = mpl.colormaps["coolwarm"](norm)[..., :3]
    rgb[~tis] = 1.0
    stats = {
        "score_p02": float(lo), "score_p98": float(hi),
        "coherence_mean": float(np.mean(feats["coherence"][tis])),
        "px_um": float(wpx),
    }
    return rgb.astype(np.float32), wpx, stats
