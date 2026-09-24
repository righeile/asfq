#!/usr/bin/env python3
"""
Fibrosis by area, for tissue that has no wall to straighten.

The aorta module measures a *structure*: it finds the media, lays it flat and
reports collagen through the thickness of it.  Heart and kidney have no such
structure to follow -- interstitial fibrosis is diffuse, scattered between
myocytes or around tubules -- so there is nothing to unroll and the question
becomes simply how much of this tissue is collagen.

Two things carry over from the wall work unchanged, because they are about the
stain rather than about the aorta:

*Optical density, not hue.*  Concentrations come from colour deconvolution, so
a pale section and a dark one give the same answer.  A hue-and-saturation
score cannot, because it confounds how much dye is present with how dark the
field happened to be.

One thing does *not* carry over: where the stain vectors come from.  The wall
analysis reads them off each section (Macenko), which works there because a
piece of aorta always contains both stains.  A field of myocardium does not --
it is nearly all muscle -- and Macenko then fits its second direction to noise
and reports that noise as collagen.  On this data that called healthy
myocardium 23 % fibrotic; the same tissue with vectors that did not come from
it reads 0.4 %.  So the default here is a fixed trichrome matrix, with
:func:`shared_stain_matrix` to estimate one across a whole run when the
staining needs following.

*No threshold on fibrosis.*  A pixel counts as fibrotic when collagen
outweighs muscle in it, which is a comparison between two measured quantities.
The alternative -- an intensity cut-off chosen per section from the collagen
itself -- was tried on the aorta and moved the reported fibrosis from 9 % to
34 % depending on where it landed, and it is wrong in principle as well: a
more fibrotic section earns a higher threshold, which cancels part of the
difference being measured.

*Background is subtracted, not thresholded.*  Concentrations are clipped at
zero, so in a pale pixel the counterstain lands exactly on zero and any trace
of collagen wins the comparison.  On a stitched ventricle 87 % of everything
called fibrotic was such a pixel -- median collagen 0.02 OD against 1.05 in
real muscle -- and heart came out at 13 % fibrotic, overlapping the kidneys.

What those pixels are carrying is background, so background is what comes off
them.  Both stain channels are measured on blank slide -- the part of the
frame the microscope looked at and found nothing -- and that level is
subtracted before anything is compared.  Nothing is thresholded and nothing is
chosen: a pixel counts as collagen when, after background, collagen outweighs
the counterstain.

Measured well clear of the section, because the halo around one is part tissue
and a couple of those pixels ruin the estimate: taken raw at the section's
edge, one heart section reported 0.83 OD of "counterstain" on its own blank
slide.  Median plus MAD, thirty microns out, for the same reason.

**What this costs, said plainly.**  Blank slide understates the uncertainty
inside tissue: the deconvolution's error grows with a pixel's optical density,
and blank slide has almost none.  So subtracting the background removes the
offset but not the whole of the pale-pixel problem, and the area fraction
lands between the two extremes -- heart around 7 % where an explicit 0.10 OD
cut-off gave 2.6 % and no correction at all gave 13.5 %.  The *ranking* holds
throughout (heart below kidney by two to three fold at every setting) and the
collagen-to-counterstain **ratio is untouched by any of it**, being an
integral in which an empty pixel contributes nothing.

That is why the ratio is the headline and the area fraction is read beside
it, and why a focal lesion is better counted as a lesion (see
:func:`find_lesions`) than inferred from a pixel count.

The one thing that is genuinely new here is the denominator.  These sections
are half empty slide, and how much varies with where the field was taken, so
everything is reported **per unit tissue area** rather than per unit image
area.  Per image area would mostly measure how much blank slide the operator
included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import wall_analysis as WA
from wall_analysis import (
    DEFAULT_STAINS,
    _unit,
    auto_tissue_threshold,
    coverage_from_mosaic,
    estimate_stains,
    rgb_to_od,
    separate_stains,
    tissue_mask,
)

FIXED_STAINS = np.stack([_unit(DEFAULT_STAINS["muscle"]),
                         _unit(DEFAULT_STAINS["collagen"]),
                         _unit(DEFAULT_STAINS["nuclei"])])

# What the red channel of a trichrome actually is, organ by organ.
#
# Biebrich scarlet-acid fuchsin stains *cytoplasm*.  In an aorta or a
# ventricle the cytoplasm in question is smooth or cardiac muscle, so calling
# the channel "muscle" is fair.  In a kidney it is tubular epithelium, and
# there is almost no muscle in the field at all -- a ratio reported as
# "collagen / muscle" there is not wrong arithmetic but it is a wrong name,
# and a wrong name on a figure is how a reader is misled by a correct number.
# The column names stay put, because they are what the pooling and the CSVs
# are keyed on; what changes is every label anybody reads.
COUNTERSTAIN = {"aorta": "muscle", "heart": "muscle", "kidney": "parenchyma"}


def counterstain_name(organ: str | None) -> str:
    return COUNTERSTAIN.get(str(organ or "").lower(), "muscle")


def shared_stain_matrix(
    images: Sequence[np.ndarray],
    max_px_per_image: int = 200_000,
) -> np.ndarray:
    """
    One stain matrix read off a whole set of images rather than each alone.

    Macenko finds the plane two stains span, which requires both to be *there*
    to find.  A field of nearly pure myocardium contains almost no collagen,
    so the second direction it returns is fitted to noise -- and the
    deconvolution then reports that noise as collagen.  Measured on this data
    it called healthy myocardium 23 % fibrotic; with a stain matrix that did
    not come from that one field, the same tissue reads 0.4 %.

    Pooling the tissue pixels of many fields fixes it, because the set as a
    whole does contain both stains even where individual fields do not.  Use
    this once per staining run and pass the result to every section in it;
    the numbers are then comparable across the batch for the same reason a
    shared threshold would be, without a threshold's arbitrariness.
    """
    chunks = []
    rng = np.random.default_rng(0)
    for rgb in images:
        od = rgb_to_od(np.clip(rgb, 0, 1))
        tis = tissue_mask(od.sum(axis=2))
        flat = od.reshape(-1, 3)[tis.reshape(-1)]
        if len(flat) > max_px_per_image:
            flat = flat[rng.choice(len(flat), max_px_per_image, replace=False)]
        if len(flat):
            chunks.append(flat)
    if not chunks:
        return FIXED_STAINS
    pooled = np.concatenate(chunks, axis=0)
    return estimate_stains(pooled.reshape(-1, 1, 3))


def polygon_mask(shape: tuple[int, int], polygons_um, px_um: float) -> np.ndarray:
    """
    A boolean mask from closed polygons given in microns from the top-left.

    Microns rather than pixels so a region drawn on the overview survives a
    change of render scale or of analysis resolution between the run that
    produced the picture and the run that uses the drawing -- the editor works
    on a shrunken overview, and pixels there mean nothing anywhere else.
    """
    from skimage.draw import polygon as _sk_polygon

    mask = np.zeros(shape, bool)
    for poly in polygons_um or []:
        xy = np.asarray(poly, dtype=float)
        if xy.ndim != 2 or len(xy) < 3:
            continue
        xy = xy / max(px_um, 1e-9)
        rr, cc = _sk_polygon(xy[:, 1], xy[:, 0], shape=shape)
        mask[rr, cc] = True
    return mask


@dataclass
class FibrosisParams:
    """Knobs for the area analysis; all lengths in microns unless stated."""

    analysis_px_um: float = 0.5        # working resolution, as for the wall
    tissue_od: float | None = None     # None = automatic (triangle threshold)
    min_tissue_area_um2: float = 2_000.0   # ignore specks of debris
    blue_rule: str = "dominance"       # "dominance" (collagen > muscle) or "absolute"
    blue_threshold: float | None = None    # only used by the absolute rule
    # Background is measured off the blank slide and subtracted from both
    # stain channels before anything is compared.  None means measure it; a
    # number forces it, for a section with no blank slide in the frame.
    subtract_background: bool = True
    background_od: float | None = None
    # "fixed" uses the trichrome vectors; "shared" means one matrix estimated
    # across the run and passed in; "per_image" is Macenko on each image alone
    # and is unsafe on tissue that is nearly all one stain.
    stain_estimation: str = "fixed"

    def copy_with(self, **kw) -> "FibrosisParams":
        d = asdict(self)
        d.update({k: v for k, v in kw.items() if k in d and v is not None})
        return FibrosisParams(**d)


@dataclass
class FibrosisResult:
    px_um: float
    rgb: np.ndarray            # the image the numbers were measured on
    tissue: np.ndarray         # bool, what counted as tissue
    fibrotic: np.ndarray       # bool, tissue where collagen outweighs muscle
    muscle: np.ndarray         # float32 concentration
    collagen: np.ndarray       # float32 concentration
    stain_matrix: np.ndarray
    totals: dict[str, Any] = field(default_factory=dict)
    regions: list[dict[str, Any]] = field(default_factory=list)     # one per drawn region
    lesions: list[dict[str, Any]] = field(default_factory=list)     # focal collagen excess
    lesion_labels: np.ndarray | None = None
    points: list[dict[str, Any]] = field(default_factory=list)      # marks, measured
    region_masks: dict[str, np.ndarray] = field(default_factory=dict)
    excluded: np.ndarray | None = None      # what was taken out of the mask


def analyze_fibrosis(
    rgb01: np.ndarray,
    px_um: float,
    params: FibrosisParams | None = None,
    coverage: np.ndarray | None = None,
    stain_matrix: np.ndarray | None = None,
    organ: str = "",
    exclude_um: Sequence[Sequence[Sequence[float]]] | None = None,
    regions_um: Sequence[dict[str, Any]] | None = None,
    points_um: Sequence[Sequence[float]] | None = None,
    find_lesions_too: bool = True,
    lesion_min_area_um2: float = 2_000.0,
    lesion_k_mad: float = 3.0,
    exclude_lesions_at: Sequence[Sequence[float]] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> FibrosisResult:
    """
    How much of this tissue is collagen.

    *rgb01* is a float image in [0, 1] with blank slide near 1.0, *px_um* its
    pixel size.  Pass *coverage* (from ``render_mosaic(..., return_coverage=
    True)``) when the image is a mosaic: canvas that no tile ever painted is
    not blank slide, and counting it as such would inflate the denominator
    with area the microscope never looked at.

    *stain_matrix* defaults to the fixed trichrome vectors rather than to
    Macenko on this one image, which is deliberate -- see
    :func:`shared_stain_matrix` for what per-image estimation does to tissue
    that is nearly all one stain.  ``params.stain_estimation="per_image"``
    restores the old behaviour for anything that needs it.

    *exclude_um* is a list of closed polygons, in microns from the top-left,
    taken **out** of the tissue before anything is measured -- a fold, a tear,
    a knife score, the mounting medium at the edge of a section.  Nothing that
    falls inside them counts towards either the numerator or the denominator,
    which is the point: masking a fold by hand is honest, and leaving it in to
    be counted as tissue is not.

    *regions_um* is a list of ``{"name": ..., "polygon": [[x, y], ...]}``, each
    measured separately as well as being part of the whole.  A kidney is what
    this is for: cortex, medulla and papilla have visibly different baseline
    collagen, so one number over a section that spans all three mostly reports
    which of them the knife caught.
    """
    p = params or FibrosisParams()
    say = progress or (lambda f, m: None)

    say(0.05, "preparing image")
    import stitch_core as SC
    scale = px_um / max(p.analysis_px_um, 1e-6)
    if abs(scale - 1.0) > 1e-3:
        rgb = np.clip(SC.resize(rgb01, scale, "area"), 0, 1)
        cov = (SC.resize(coverage.astype(np.float32), scale, "area") > 0.5
               if coverage is not None else None)
    else:
        rgb, cov = np.clip(rgb01, 0, 1), coverage
    work_px_um = px_um / scale
    if cov is None:
        cov = coverage_from_mosaic(rgb)

    say(0.25, "separating stains")
    od = rgb_to_od(rgb)
    od_sum = od.sum(axis=2)
    thr = auto_tissue_threshold(od_sum) if p.tissue_od is None else float(p.tissue_od)
    min_px = max(1, int(round(p.min_tissue_area_um2 / (work_px_um ** 2))))
    tis = tissue_mask(od_sum, thr, min_area_px=min_px) & cov

    # Hand-drawn regions, in the working frame.  Exclusions come off the
    # tissue before anything is measured, so they leave both the numerator and
    # the denominator -- an excluded fold must not quietly become "tissue with
    # no collagen in it", which would dilute the fraction instead of removing
    # the fold from the question.
    excluded = polygon_mask(tis.shape, exclude_um, work_px_um)
    if excluded.any():
        tis = tis & ~excluded
    region_masks: dict[str, np.ndarray] = {}
    for r in (regions_um or []):
        name = str(r.get("name") or "region").strip() or "region"
        poly = r.get("polygon") or r.get("um")
        if poly is None:
            continue
        m = polygon_mask(tis.shape, [poly], work_px_um)
        # Two strokes of the same name are one region: a cortex that the
        # section catches twice is still the cortex.
        region_masks[name] = (region_masks[name] | m) if name in region_masks else m

    if stain_matrix is not None:
        M = np.asarray(stain_matrix, float)
    elif p.stain_estimation == "per_image":
        M = estimate_stains(od, tis)
    else:
        M = FIXED_STAINS
    conc = separate_stains(od, M)
    muscle = conc[..., 0].astype(np.float32)
    collagen = conc[..., 1].astype(np.float32)

    # -- background ---------------------------------------------------------
    # What this deconvolution reports where there is definitionally no stain:
    # blank slide the microscope did look at.  Measured a good way clear of the
    # section, because the halo around one is part tissue and a couple of those
    # pixels ruin the estimate -- taken raw, one heart section reported 0.83 OD
    # of "counterstain" on its own blank slide.  Median plus MAD rather than a
    # percentile, for the same reason.
    away = cov & ~ndi.binary_dilation(tis, np.ones((3, 3)),
                                      iterations=max(1, int(30.0 / work_px_um)))

    def _background(chan: np.ndarray) -> tuple[float, float]:
        v = chan[away]
        if v.size < 5_000:
            return float("nan"), float("nan")
        med = float(np.median(v))
        return med, float(np.median(np.abs(v - med))) * 1.4826

    bg_c, sd_c = _background(collagen)
    bg_m, sd_m = _background(muscle)
    if p.background_od is not None:
        bg_c = bg_m = float(p.background_od)
    if p.subtract_background and np.isfinite(bg_c):
        collagen = np.clip(collagen - bg_c, 0, None)
        muscle = np.clip(muscle - (bg_m if np.isfinite(bg_m) else 0.0), 0, None)
    else:
        bg_c = bg_m = 0.0
    background_noise = float(np.hypot(sd_c, sd_m)) if np.isfinite(sd_c) else float("nan")

    say(0.7, "measuring")
    # A pixel is fibrotic when collagen outweighs the counterstain in it.
    # Both sides are measured on the same pixel in the same units and both
    # have had the background taken off them, which is what makes the rule
    # comparable between sections.  There is no threshold in it.
    if p.blue_rule == "absolute" and p.blue_threshold is not None:
        fibrotic = tis & (collagen > float(p.blue_threshold))
    else:
        fibrotic = tis & (collagen > muscle)

    area_px = float(work_px_um ** 2)
    covered_um2 = float(cov.sum()) * area_px

    def measure(where: np.ndarray) -> dict[str, Any]:
        """Every number, over whatever part of the tissue is asked for.

        One function so that a region and the whole section are measured the
        same way.  A per-region summary computed by a second, similar-looking
        block is a per-region summary that will disagree with the total in
        some corner nobody checks.
        """
        n_tissue = int(where.sum())
        tissue_area_um2 = n_tissue * area_px
        c_in = collagen[where]
        m_in = muscle[where]
        c_total = float(c_in.sum()) * area_px          # OD.um^2
        m_total = float(m_in.sum()) * area_px
        denom = c_total + m_total
        blue = float((fibrotic & where).sum())
        return {
            # -- the denominator, said out loud --------------------------
            "tissue_area_mm2": tissue_area_um2 / 1e6,
            "imaged_area_mm2": covered_um2 / 1e6,
            "tissue_fraction_of_image": (tissue_area_um2 / covered_um2) if covered_um2 else float("nan"),
            # -- the headline: collagen against the counterstain, both
            #    integrated over the same tissue, so staining strength
            #    divides out.  The counterstain is muscle in a heart and
            #    tubular epithelium in a kidney; the column keeps one name so
            #    the tables pool, and `counterstain` says what it was.
            "collagen_to_muscle": (c_total / m_total) if m_total > 0 else float("nan"),
            "collagen_fraction_of_stain": (c_total / denom) if denom > 0 else float("nan"),
            # -- the interpretable one: how much of the tissue is fibrotic
            "collagen_area_fraction": (blue / n_tissue) if n_tissue else float("nan"),
            "collagen_area_mm2": blue * area_px / 1e6,
            # -- per unit tissue area, for anyone who wants it that way ---
            "collagen_od_um2_per_mm2_tissue": (c_total / (tissue_area_um2 / 1e6)) if n_tissue else float("nan"),
            "muscle_od_um2_per_mm2_tissue": (m_total / (tissue_area_um2 / 1e6)) if n_tissue else float("nan"),
            # -- means, and what the run was told to do ------------------
            "collagen_mean_conc": float(c_in.mean()) if n_tissue else float("nan"),
            "muscle_mean_conc": float(m_in.mean()) if n_tissue else float("nan"),
            "collagen_od_um2_total": c_total,
            "muscle_od_um2_total": m_total,
            "counterstain": counterstain_name(organ),
            "blue_rule": p.blue_rule if p.blue_rule != "absolute" or p.blue_threshold is None
                         else "absolute",
            "blue_threshold": p.blue_threshold if p.blue_rule == "absolute" else None,
            "tissue_od_threshold": thr,
            "collagen_background_od": bg_c,
            "counterstain_background_od": bg_m,
            "background_noise_od": background_noise,
            "analysis_px_um": work_px_um,
            "n_tissue_px": n_tissue,
            "excluded_area_mm2": float(excluded.sum()) * area_px / 1e6,
        }

    def _lesion_totals() -> dict[str, Any]:
        n = len(lesions)
        area = float(sum(l["area_um2"] for l in lesions))
        n_t = int(tis.sum())
        return {
            "n_lesions": n,
            "lesion_area_mm2": area / 1e6,
            "lesion_area_fraction": (area / (n_t * area_px)) if n_t else float("nan"),
            "lesion_area_mean_um2": (area / n) if n else float("nan"),
            "lesions_per_mm2_tissue": (n / (n_t * area_px / 1e6)) if n_t else float("nan"),
            **lesion_info,
        }

    say(0.85, "looking for lesions")
    lesion_labels, lesions, lesion_info = (
        (np.zeros(tis.shape, np.int32), [], {}) if not find_lesions_too else
        find_lesions(collagen, muscle, tis, work_px_um,
                     min_area_um2=lesion_min_area_um2, k_mad=lesion_k_mad,
                     exclude_at=exclude_lesions_at))
    points = measure_points(points_um, collagen, muscle, tis, work_px_um,
                            lesion_labels=lesion_labels)

    totals = {**measure(tis), **_lesion_totals()}
    # One row per drawn region, measured exactly as the whole section is.  A
    # kidney is three tissues with different baseline collagen stacked in one
    # section, and a single number over all of them mostly reports how much of
    # each the knife happened to catch.
    region_totals = [{"name": name, **measure(tis & mask)}
                     for name, mask in region_masks.items()]
    say(1.0, "done")
    return FibrosisResult(px_um=work_px_um, rgb=rgb, tissue=tis, fibrotic=fibrotic,
                          muscle=muscle, collagen=collagen, stain_matrix=M,
                          totals=totals, regions=region_totals,
                          region_masks=region_masks, excluded=excluded,
                          lesions=lesions, lesion_labels=lesion_labels,
                          points=points)


# =============================================================================
# The figure
# =============================================================================
#
# Six panels, and the sixth is the one that matters.  Panels A-E show what was
# measured; panel F shows *where the decision fell* -- the distribution of
# collagen minus muscle over every tissue pixel, with the boundary drawn on
# it.  A fibrosis number is only worth as much as that boundary, and a reader
# who cannot see whether it landed in a valley or through the middle of the
# main peak has to take the number on faith.


#: A little arrowhead, drawn as a downward triangle above whatever it points
#: at.  White with a thin dark edge so it survives both a pale margin and a
#: dark section -- the two places a lesion is most likely to sit.
# Sized to survive being shown small.  The app displays this figure at
# whatever width the panel gets -- often under 200 px for a single panel in a
# six-panel figure -- so a marker that reads clearly in a full-page export
# can vanish to a couple of screen pixels in the app.  This is bigger than
# "tiny" would suggest on paper, and is exactly as tiny as it can be while
# still being findable at the sizes this actually gets looked at.
ARROW_MS = 7.0


def _mark_lesions(ax, lesions, px_um, rotation_deg, src_shape, dst_shape,
                  scale: float = 1.0) -> None:
    """Arrowheads over each lesion, in whatever frame the panel is drawn in."""
    if not lesions:
        return
    # The picture may have been turned for the page; rotate the marks with it
    # about the image centre, the same way scipy.ndimage.rotate does.
    th = np.deg2rad(rotation_deg)
    cs, sn = np.cos(th), np.sin(th)
    cx0, cy0 = src_shape[1] / 2.0, src_shape[0] / 2.0
    cx1, cy1 = dst_shape[1] / 2.0, dst_shape[0] / 2.0
    ms = ARROW_MS * scale
    off = 3.4 * ms
    for les in lesions:
        x = les["x_um"] / px_um - cx0
        y = les["y_um"] / px_um - cy0
        xr = cx1 + x * cs + y * sn
        yr = cy1 - x * sn + y * cs
        ax.plot(xr, yr - off, marker="v", ms=ms, mfc="white", mec="#1a1a1a",
                mew=1.1, zorder=8, linestyle="none")


def _panel_letter(ax, letter: str, fs: float) -> None:
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", ha="right", va="bottom")


def _style(ax, show_grid: bool, show_spines: bool, fs_tick: float) -> None:
    ax.tick_params(labelsize=fs_tick)
    if show_grid:
        ax.grid(True, alpha=0.25, linewidth=0.6)
    if not show_spines:
        # Bottom and left stay: they carry the ticks.  Losing the box is a
        # style choice; losing the axis the numbers sit on is a mistake.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def plot_fibrosis(
    res: FibrosisResult,
    title: str = "",
    organ: str = "",
    font_scale: float = 1.0,
    show_grid: bool = True,
    show_spines: bool = True,
    panel_in: float = 3.0,
    rotation_deg: float = 0.0,
    stain_gain: Sequence[float] = (1.0, 1.0, 1.0),
    svg: bool = False,
):
    """
    The section, what counted as tissue, the two stains, and the decision.

    *rotation_deg* turns the picture panels only.  Nothing measured changes --
    area, optical density and the comparison between two stains are all
    invariant to how the slide was laid on the stage -- so this is the same
    figure at a different angle, not a different analysis.  It is here because
    a section is mounted whichever way it came off the knife, and a figure
    panel has to sit the way the page needs it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fs_title = WA.FS_TITLE * font_scale
    fs_label = WA.FS_LABEL * font_scale
    fs_note = WA.FS_NOTE * font_scale
    fs_tick = fs_title - 1
    fs_panel = fs_title + 1
    # Whatever the red channel is in this organ.  Every label a reader sees
    # says it; the column names underneath do not change.
    cs = res.totals.get("counterstain") or counterstain_name(organ)

    def turn(arr, order=1):
        if not rotation_deg:
            return arr
        from scipy import ndimage as _ndi
        out = _ndi.rotate(np.asarray(arr, dtype=float), rotation_deg, axes=(1, 0),
                          reshape=True, order=order, mode="constant",
                          cval=1.0 if (arr.ndim == 3) else 0.0)
        return (out > 0.5) if arr.dtype == bool else out

    # Display only.  The picture panels can have a stain turned down so the
    # collagen is legible for scoring by eye; every number below came from the
    # concentrations before any of this and is untouched by it.
    rgb_v = np.clip(turn(WA.restain(res.rgb, res.stain_matrix, stain_gain)), 0, 1)
    tis_v = turn(res.tissue, order=0)
    fib_v = turn(res.fibrotic, order=0)
    col_v = turn(res.collagen)
    mus_v = turn(res.muscle)
    exc_v = turn(res.excluded, order=0) if res.excluded is not None else None
    reg_v = {k: turn(v, order=0) for k, v in (res.region_masks or {}).items()}

    h, w = tis_v.shape
    aspect = h / max(w, 1)
    fig, axes = plt.subplots(2, 3, figsize=(3 * panel_in,
                                            2 * panel_in * min(max(aspect, 0.55), 1.4) + 0.6))
    (ax_rgb, ax_tis, ax_cls), (ax_col, ax_mus, ax_hist) = axes

    ax_rgb.imshow(rgb_v)
    ax_rgb.set_title("section", fontsize=fs_title)

    # What counted as tissue, drawn over the section rather than beside it:
    # the question a reader has is whether the mask followed the tissue, and
    # that is answered by looking through it, not at it.
    ax_tis.imshow(rgb_v)
    ax_tis.imshow(np.ma.masked_where(tis_v, np.zeros_like(tis_v, float)),
                  cmap="gray", vmin=0, vmax=1, alpha=0.72)
    # Anything taken out by hand is marked, not silently absent: a mask a
    # reader cannot see is a mask a reader cannot check.
    if exc_v is not None and exc_v.any():
        cut = np.zeros(rgb_v.shape[:2] + (4,), float)
        cut[exc_v] = (0.85, 0.20, 0.15, 0.30)
        ax_tis.imshow(cut)
    for name, mask in reg_v.items():
        ax_tis.contour(mask.astype(float), levels=[0.5], colors="#0b59f2", linewidths=0.9)
        ys, xs = np.nonzero(mask)
        if len(xs):
            # A halo, because the label lands on whatever the section is and
            # blue on dark tissue is not a label anyone can read.
            import matplotlib.patheffects as pe
            ax_tis.text(xs.mean(), ys.mean(), name, fontsize=fs_note, color="#0b59f2",
                        ha="center", va="center", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    cut_note = (f"  −{res.totals['excluded_area_mm2']:.2f} masked"
                if res.totals.get("excluded_area_mm2") else "")
    ax_tis.set_title(f"tissue — {res.totals['tissue_area_mm2']:.2f} mm²{cut_note}",
                     fontsize=fs_title)

    ax_cls.imshow(rgb_v)
    ov = np.zeros(rgb_v.shape[:2] + (4,), float)
    ov[fib_v] = (0.05, 0.35, 0.95, 0.80)
    ax_cls.imshow(ov)
    # Lesions marked, not drawn over.  An outline traces the thing you are
    # trying to look at, and a coloured one puts a second stain on a section
    # that already has two.  A small white arrowhead above each lesion points
    # at it and covers nothing; the shapes themselves are on the editor
    # canvas, where there is room to look at them.
    _mark_lesions(ax_cls, res.lesions, res.px_um, rotation_deg,
                  res.tissue.shape, tis_v.shape, font_scale)
    for pt in (res.points or []):
        if not rotation_deg:
            ax_cls.plot(pt["x_um"] / res.px_um, pt["y_um"] / res.px_um, "+",
                        color="#ffb000", mew=1.6, ms=7)
            ax_cls.text(pt["x_um"] / res.px_um + 6, pt["y_um"] / res.px_um,
                        str(pt["point"]), fontsize=fs_note, color="#ffb000",
                        ha="left", va="center", fontweight="bold")
    ax_cls.set_title(f"collagen > {cs} — "
                     f"{100 * res.totals['collagen_area_fraction']:.1f}%",
                     fontsize=fs_title)

    for ax, arr, cmap, lab in ((ax_col, col_v, "Blues", "collagen"),
                               (ax_mus, mus_v, "Reds", cs)):
        shown = np.where(tis_v, arr, np.nan)
        hi = float(np.nanpercentile(shown, 99.5)) if tis_v.any() else 1.0
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=max(hi, 1e-6))
        ax.set_title(f"{lab} (OD)", fontsize=fs_title)

    for ax in (ax_rgb, ax_tis, ax_cls, ax_col, ax_mus):
        ax.set_xticks([])
        ax.set_yticks([])
        if not show_spines:
            for side in ax.spines:
                ax.spines[side].set_visible(False)

    # -- panel F: the decision, drawn in the plane it is actually taken in ---
    # Every tissue pixel is one point, after background: the diagonal is
    # collagen against the counterstain, and what is counted is everything
    # above it.  A plane rather than a histogram of the difference because the
    # distance of the cloud from the diagonal is the thing worth seeing, and a
    # 1-D projection throws it away.  The dashed line is the background level
    # that was taken off each channel, so how much of the answer rests on that
    # correction is visible rather than taken on trust.
    mus, col = res.muscle[res.tissue], res.collagen[res.tissue]
    bg_c = float(res.totals.get("collagen_background_od", 0.0) or 0.0)
    if mus.size:
        # One limit for both axes, so the diagonal is a true 45 degrees and the
        # distance of the cloud from it can be read off.  Scaling each axis to
        # its own data uses the space better and ruins the panel: the
        # counterstain runs several times higher than the collagen in all of
        # these tissues, so the diagonal would leave through the corner and the
        # criterion it stands for would stop being visible.
        hi = float(np.nanpercentile(np.concatenate([mus, col]), 99.5)) or 1.0
        step = max(1, mus.size // 200_000)      # a plot, not a census
        hb = ax_hist.hexbin(mus[::step], col[::step], gridsize=48, bins="log",
                            extent=(0, hi, 0, hi), cmap="Greys", mincnt=1,
                            linewidths=0)
        hb.set_rasterized(True)                 # a vector file, not 2000 hexagons
        xs = np.linspace(0, hi, 200)
        ax_hist.plot(xs, xs, color="#0b59f2", lw=1.2)
        ax_hist.fill_between(xs, xs, hi, color="#0b59f2", alpha=0.06, lw=0)
        if bg_c > 0:
            ax_hist.axhline(bg_c, color="#d93326", lw=1.0, ls=(0, (3, 2)))
        ax_hist.set_xlim(0, hi)
        ax_hist.set_ylim(0, hi)
    ax_hist.set_xlabel(f"{cs} (OD, background removed)", fontsize=fs_label)
    ax_hist.set_ylabel("collagen (OD, background removed)", fontsize=fs_label)
    ax_hist.set_title("where the boundary falls", fontsize=fs_title)
    handles = [Patch(facecolor="#0b59f2", alpha=0.25,
                     label=f"counted: {100 * res.totals['collagen_area_fraction']:.1f}% of tissue")]
    if bg_c > 0:
        handles.append(Line2D([0], [0], color="#d93326", lw=1.0, ls=(0, (3, 2)),
                              label=f"background removed {bg_c:.3f} OD"))
    ax_hist.legend(handles=handles, fontsize=fs_note, frameon=False, loc="upper right")
    _style(ax_hist, show_grid, show_spines, fs_tick)

    for ax, letter in zip((ax_rgb, ax_tis, ax_cls, ax_col, ax_mus, ax_hist), "ABCDEF"):
        _panel_letter(ax, letter, fs_panel)

    t = res.totals
    note = (f"collagen÷{cs} {t['collagen_to_muscle']:.3f}   ·   "
            f"{100 * t['collagen_area_fraction']:.1f}% of tissue area   ·   "
            f"tissue {t['tissue_area_mm2']:.2f} mm² "
            f"({100 * t['tissue_fraction_of_image']:.0f}% of what was imaged)   ·   "
            f"{t['analysis_px_um']:.2f} µm/px")
    fig.text(0.5, 0.005, ((title + "   —   ") if title else "") + note,
             ha="center", va="bottom", fontsize=fs_note, color="0.25")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


def save_results(res: FibrosisResult, out_dir: str | Path, name: str = "fibrosis",
                 title: str = "", organ: str = "", **style) -> dict[str, str]:
    """The numbers as a table, the figure as a raster to look at and vector to edit."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    pd.DataFrame([res.totals]).to_csv(out_dir / f"{name}_totals.csv", index=False)
    paths["totals"] = str(out_dir / f"{name}_totals.csv")

    if res.lesions:
        pd.DataFrame(res.lesions).to_csv(out_dir / f"{name}_lesions.csv", index=False)
        paths["lesions"] = str(out_dir / f"{name}_lesions.csv")
    if res.points:
        pd.DataFrame(res.points).to_csv(out_dir / f"{name}_points.csv", index=False)
        paths["points"] = str(out_dir / f"{name}_points.csv")

    if res.regions:
        # The whole section is in here too, as a row named "whole section", so
        # the file answers "how do the regions compare, and to what" without
        # anyone having to join it against the totals by hand.
        rows = [{"name": "whole section", **res.totals}] + list(res.regions)
        pd.DataFrame(rows).to_csv(out_dir / f"{name}_regions.csv", index=False)
        paths["regions"] = str(out_dir / f"{name}_regions.csv")

    # The stain vectors the numbers were produced with.  Two runs of the same
    # tissue that used different vectors are not comparable, and the only way
    # to notice that later is to have written them down.
    pd.DataFrame(res.stain_matrix, columns=["R", "G", "B"],
                 index=["muscle", "collagen", "residual"]).to_csv(
        out_dir / f"{name}_stain_vectors.csv")
    paths["stain_vectors"] = str(out_dir / f"{name}_stain_vectors.csv")

    fig = plot_fibrosis(res, title=title, organ=organ, **style)
    fig.savefig(out_dir / f"{name}_figure.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths["figure"] = str(out_dir / f"{name}_figure.png")
    # Real <text>, not every glyph traced out as its own curve, so the labels
    # stay editable in a vector editor.
    with plt.rc_context({"svg.fonttype": "none"}):
        fig = plot_fibrosis(res, title=title, organ=organ, svg=True, **style)
        fig.savefig(out_dir / f"{name}_figure.svg", bbox_inches="tight")
        plt.close(fig)
    paths["figure_svg"] = str(out_dir / f"{name}_figure.svg")

    # A picture to draw regions on, not a second copy of the data: the working
    # image can be eight thousand pixels across and the editor shows it in a
    # box a thousand wide.  The scale it was saved at travels with it, because
    # a polygon drawn here is sent back in microns and microns are the only
    # thing both ends agree about.
    import stitch_core as SC
    ov_scale = min(1.0, 1400.0 / max(res.rgb.shape[:2]))
    overview = np.clip(SC.resize(res.rgb, ov_scale, "area"), 0, 1) if ov_scale < 1 else res.rgb
    plt.imsave(out_dir / f"{name}_overview.jpg", overview)
    paths["overview"] = str(out_dir / f"{name}_overview.jpg")
    paths["overview_px_um"] = str(res.px_um / max(ov_scale, 1e-9))
    return paths


# =============================================================================
# Focal lesions
# =============================================================================


def find_lesions(
    collagen: np.ndarray,
    counterstain: np.ndarray,
    tissue: np.ndarray,
    px_um: float,
    min_area_um2: float = 2_000.0,
    smooth_um: float = 25.0,
    k_mad: float = 3.0,
    exclude_at: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    """
    Patches where collagen is focally higher than this section's own level.

    A lesion is a *local* excess, so what it is measured against is the rest of
    the same section: the collagen share is smoothed to lesion scale, and a
    patch counts when it stands more than *k_mad* robust deviations above that
    section's median.  Median and MAD rather than mean and SD because a section
    with big lesions in it would otherwise raise its own bar and hide them.

    Being relative is the point and also the limitation, and the two are worth
    keeping straight.  It finds *focal* disease -- scars, patches, a fibrotic
    wedge -- against whatever the surrounding tissue happens to be.  It cannot
    see *diffuse* fibrosis, because tissue that is uniformly fibrotic has no
    focus to stand out from and will return no lesions at all while being
    thoroughly diseased.  The area fraction and the collagen-to-counterstain
    ratio are what answer that question; this answers "where, and how many".

    *exclude_at* removes specific lesions after detection -- points in the same
    microns as everything else drawn on the overview, one per lesion a person
    has looked at and rejected.  Nothing about the detector changes: the same
    candidates are found at the same cut, and a point simply erases whichever
    one it falls inside.  That order matters.  Lowering the cut until a false
    positive disappears would also raise or drop lesions nobody has looked at,
    everywhere else in the section; a click removes exactly the one patch a
    person decided was not real and touches nothing else.

    The third return value is what the cut-off was and what the section
    actually reached, and it is there so that **no lesions** is a readable
    answer rather than a silent one.  An aortic media returns nothing at the
    default and should: its collagen is lamellar, interleaved with muscle all
    the way through, so the share distribution is broad and unimodal and
    nothing stands three deviations above it -- median 0.39 against a maximum
    of 0.65, with the cut at 0.76.  Widen the section to include the
    adventitia and it gets worse rather than better, because the tissue is
    then bimodal (collagen-rich adventitia, muscle-rich media) and the spread
    is wider still.  That is not a failure to find lesions; it is diffuse
    disease in a tissue that has no focal lesions to find, and the numbers
    below say so instead of leaving you guessing.
    """
    from skimage.measure import regionprops

    total = collagen + counterstain
    share = np.where(total > 1e-6, collagen / np.maximum(total, 1e-6), 0.0)
    sigma = max(1.0, smooth_um / max(px_um, 1e-6))
    share = WA.gaussian(share, sigma)

    inside = share[tissue]
    if inside.size < 100:
        return np.zeros(tissue.shape, np.int32), [], {}
    med = float(np.median(inside))
    mad = float(np.median(np.abs(inside - med))) * 1.4826
    cut = med + k_mad * max(mad, 1e-6)
    info = {
        "lesion_cut_share": cut,
        "lesion_share_median": med,
        "lesion_share_mad": mad,
        "lesion_share_max": float(inside.max()),
        "lesion_k_mad": float(k_mad),
        "lesion_min_area_um2": float(min_area_um2),
    }

    hot = tissue & (share > cut)
    hot = WA.square_morph(hot, 3, "open")
    hot = WA.fill_holes(hot)
    labels, n = ndi.label(hot)

    area_px = float(px_um ** 2)
    min_px = max(1, int(round(min_area_um2 / area_px)))
    out: list[dict[str, Any]] = []
    keep = np.zeros(labels.shape, np.int32)
    for prop in regionprops(labels):
        if prop.area < min_px:
            continue
        m = labels == prop.label
        idx = len(out) + 1
        keep[m] = idx
        cy, cx = prop.centroid
        c_sum = float(collagen[m].sum())
        m_sum = float(counterstain[m].sum())
        out.append({
            "lesion": idx,
            "area_um2": prop.area * area_px,
            "area_mm2": prop.area * area_px / 1e6,
            "x_um": cx * px_um,
            "y_um": cy * px_um,
            # skimage renamed this property in 0.26 (equivalent_diameter is
            # deprecated, gone in 2.0); area is unambiguous either way, so
            # compute the diameter from it directly rather than depend on
            # whichever spelling the installed version has.
            "equivalent_diameter_um": float(2.0 * np.sqrt(prop.area / np.pi)) * px_um,
            "collagen_share_mean": float(share[m].mean()),
            "collagen_share_peak": float(share[m].max()),
            "collagen_to_counterstain": (c_sum / m_sum) if m_sum > 0 else float("nan"),
            "collagen_od_um2": c_sum * area_px,
        })

    if exclude_at:
        # A rejected lesion's id in `keep` and its "lesion" number in `out` are
        # the same integer -- both came from the one `idx` above -- so a point
        # only has to name the id at that pixel to remove it from both.
        #
        # Not a single exact-pixel lookup, though: "x_um"/"y_um" is a
        # *centroid*, and a centroid can sit outside its own shape whenever
        # the shape is not convex -- an arc of curved wall is exactly that
        # shape, and on real aortic data the reported centroid landed on
        # background, not the lesion, in the very first section tried.  The
        # outline sent to the app for clicking is also simplified by a few
        # microns (see lesion_outlines), so even a click confirmed on-screen
        # to be inside the drawn polygon can land just outside the true mask
        # after that simplification.  A small search radius makes both of
        # those non-issues: it finds whichever lesion is actually nearest the
        # point instead of demanding the point be exactly on one.
        h, w = keep.shape
        drop: set[int] = set()
        search_px = max(1, int(round(20.0 / max(px_um, 1e-6))))
        for pt in exclude_at:
            px, py = int(round(pt[0] / px_um)), int(round(pt[1] / px_um))
            if not (0 <= py < h and 0 <= px < w):
                continue
            if keep[py, px] > 0:
                drop.add(int(keep[py, px]))
                continue
            y0, y1 = max(0, py - search_px), min(h, py + search_px + 1)
            x0, x1 = max(0, px - search_px), min(w, px + search_px + 1)
            window = keep[y0:y1, x0:x1]
            ys, xs = np.nonzero(window)
            if len(ys):
                d2 = (ys + y0 - py) ** 2 + (xs + x0 - px) ** 2
                drop.add(int(window[ys[d2.argmin()], xs[d2.argmin()]]))
        if drop:
            out = [r for r in out if r["lesion"] not in drop]
            # A lookup table indexed by the old id, not a per-pixel Python
            # call: `keep` is a full mosaic and can be tens of millions of
            # pixels, and this is the only step here that touches all of them.
            n_old = int(keep.max())
            lut = np.zeros(n_old + 1, dtype=np.int32)
            for new_id, r in enumerate(out, start=1):
                lut[r["lesion"]] = new_id
                r["lesion"] = new_id
            keep = lut[keep]
            info["lesions_removed_by_hand"] = len(drop)

    return keep, out, info


def lesion_outlines(labels: np.ndarray, px_um: float,
                    simplify_um: float = 4.0) -> list[list[list[float]]]:
    """
    Each lesion as a closed polygon in microns, for drawing on the overview.

    The editor draws on the stitched section, and a lesion is only useful there
    if you can see its shape against the tissue -- a dot at the centroid says a
    lesion was found somewhere near here, which is not the same as showing you
    what was counted.
    """
    from skimage.measure import find_contours, approximate_polygon

    out: list[list[list[float]]] = []
    for k in range(1, int(labels.max()) + 1):
        m = labels == k
        if not m.any():
            continue
        padded = np.pad(m.astype(float), 1)
        for contour in find_contours(padded, 0.5):
            poly = approximate_polygon(contour, tolerance=simplify_um / max(px_um, 1e-6))
            if len(poly) < 4:
                continue
            out.append(((poly[:, ::-1] - 1.0) * px_um).tolist())   # (row,col) -> (x,y) um
    return out


def measure_points(
    points_um: Sequence[Sequence[float]],
    collagen: np.ndarray,
    counterstain: np.ndarray,
    tissue: np.ndarray,
    px_um: float,
    radius_um: float = 25.0,
    lesion_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    What is at each marked point -- the counting tool, made quantitative.

    A click on its own gives a count, which is what a multipoint tool is for.
    A click here also gives the tissue in a small disc around it, so a set of
    marks is a table rather than a tally: how fibrotic each marked spot is, and
    which detected lesion it landed in, if any.

    *radius_um* is a fixed disc rather than the lesion's own extent on purpose.
    Every point is then measured over the same area and the numbers are
    comparable between marks; growing the disc to fit whatever is under it
    would make a big lesion and a small one differ by how much was averaged.
    """
    h, w = tissue.shape
    r = max(1, int(round(radius_um / max(px_um, 1e-6))))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (yy ** 2 + xx ** 2) <= r * r

    out: list[dict[str, Any]] = []
    for i, pt in enumerate(points_um or [], start=1):
        cx, cy = float(pt[0]) / px_um, float(pt[1]) / px_um
        x0, y0 = int(round(cx)) - r, int(round(cy)) - r
        sl = (slice(max(y0, 0), min(y0 + 2 * r + 1, h)),
              slice(max(x0, 0), min(x0 + 2 * r + 1, w)))
        # A mark can land outside the image -- the editor draws on an overview
        # and nothing stops a click near the edge -- and an empty slice is not
        # an error, it is a point with no tissue at it.
        sub = disc[max(0, -y0):max(0, -y0) + (sl[0].stop - sl[0].start),
                   max(0, -x0):max(0, -x0) + (sl[1].stop - sl[1].start)]
        m = (sub & tissue[sl]) if sub.size and sub.shape == tissue[sl].shape \
            else np.zeros((0, 0), bool)
        rec: dict[str, Any] = {
            "point": i, "x_um": float(pt[0]), "y_um": float(pt[1]),
            "radius_um": radius_um, "n_tissue_px": int(m.sum()),
        }
        if m.any():
            c = float(collagen[sl][m].sum())
            s = float(counterstain[sl][m].sum())
            rec.update({
                "collagen_to_counterstain": (c / s) if s > 0 else float("nan"),
                "collagen_share": c / max(c + s, 1e-9),
                "collagen_mean_od": float(collagen[sl][m].mean()),
                "counterstain_mean_od": float(counterstain[sl][m].mean()),
            })
        else:
            rec.update({"collagen_to_counterstain": float("nan"),
                        "collagen_share": float("nan"),
                        "collagen_mean_od": float("nan"),
                        "counterstain_mean_od": float("nan")})
        if lesion_labels is not None:
            iy, ix = int(round(cy)), int(round(cx))
            rec["in_lesion"] = (int(lesion_labels[iy, ix])
                                if 0 <= iy < h and 0 <= ix < w else 0)
        out.append(rec)
    return out
