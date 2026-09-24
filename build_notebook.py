#!/usr/bin/env python3
"""
Write the analysis notebook.  Run this, don't hand-edit the .ipynb.

    python build_notebook.py --data /path/to/sections --out my_analysis.ipynb

The data folder is the one thing in the notebook that is about your machine
rather than about the analysis, so it is given here rather than hardcoded:
whatever is passed becomes the notebook's default, and the notebook still
reads AORTA_DATA at run time, so the same file also works for someone whose
sections live somewhere else.

--out names the file to write.  It defaults to Aorta_wall_analysis.ipynb next
to this script, which is also the name .gitignore knows about; anything else
is yours to keep track of.  A run overwrites whatever is already at that path,
so --out is also how you build a second notebook without losing the first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "Aorta_wall_analysis.ipynb"

_parser = argparse.ArgumentParser(
    description="Write the analysis notebook.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_parser.add_argument(
    "--data", metavar="DIR",
    default=os.environ.get("AORTA_DATA", "~/aorta-sections"),
    help="folder holding one sub-folder per section, each of image fields. "
         "Becomes the notebook's default; defaults itself to $AORTA_DATA")
_parser.add_argument(
    "--out", metavar="FILE", type=Path, default=DEFAULT_OUT,
    help="notebook to write. A relative name is taken from where you are, "
         "not from the app folder; .ipynb is added if you leave it off")
_args = _parser.parse_args()

# Embedded as a Python literal, so a Windows path's backslashes survive.
DATA_DEFAULT = json.dumps(_args.data)
if not Path(_args.data).expanduser().is_dir():
    print(f"note: {_args.data} is not a folder here -- writing it in anyway, "
          f"since the notebook may be meant for another machine.")

OUT = _args.out.expanduser()
# Order matters: a folder has no .ipynb suffix, so testing for one first would
# turn `--out some/dir` into `some/dir.ipynb` beside it instead of a notebook
# inside it.
if OUT.is_dir():
    OUT = OUT / DEFAULT_OUT.name
elif OUT.suffix != ".ipynb":
    # A name given without the extension is a name, not a request for a file
    # Jupyter will not open.
    OUT = OUT.with_suffix(OUT.suffix + ".ipynb")
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    print(f"note: overwriting the existing {OUT}")


def _lines(text: str) -> list[str]:
    """nbformat wants one string per line, newline included on all but the last."""
    parts = text.strip("\n").split("\n")
    return [ln + "\n" for ln in parts[:-1]] + parts[-1:]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


cells: list[dict] = []

cells.append(md(r"""
# ASFQ — stitch, straighten, quantify

This notebook takes a folder of overlapping trichrome fields, assembles them into one
section, unrolls the vessel wall, and measures the collagen in it.

Three numbers come out of it, and it is worth being precise about what each one means:

| readout | units | what it answers |
|---|---|---|
| `collagen_od_um2_per_mm_length` | OD·µm² / mm | how much collagen there is **per millimetre of wall**, so vessels of different size compare |
| `cum_collagen_od_um2` along the wall | OD·µm² | *where* the collagen is: a straight cumulative line means diffuse, steps mean focal |
| `collagen_area_fraction_media` | fraction | proportion of the muscle layer where collagen outweighs muscle — no threshold to choose |

Everything is measured on **stain concentrations** obtained by colour deconvolution, not on
hue or saturation. Optical density is linear in dye amount (Beer–Lambert), so it can be
integrated over distance; a hue score cannot, because it confounds how much dye is present
with how dark the field happened to be.

The same code runs behind the browser app (`python app.py`). Use the app when you need to
place tiles by hand; use this notebook when you want the numbers, the plots, or a batch.
"""))

cells.append(code(r"""
import os, sys, json, time
from pathlib import Path

def _find_app_dir() -> Path:
    # Locate the app next to this notebook, wherever the folder has been moved.
    here = Path.cwd()
    for cand in (here, *here.parents):
        if (cand / "wall_analysis.py").exists() and (cand / "stitch_core.py").exists():
            return cand
    raise FileNotFoundError(
        "Could not find stitch_core.py / wall_analysis.py above the working "
        "directory. Open this notebook from the ASFQ folder, or set "
        "APP_DIR by hand."
    )

APP_DIR = _find_app_dir()
sys.path.insert(0, str(APP_DIR))
print(f"app: {APP_DIR}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import stitch_core as SC
import wall_analysis as WA
import cohort as CO

%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

# Where the sections live: one sub-folder per section, each holding that
# section's image fields.  The default below was set when this notebook was
# built (build_notebook.py --data ...); AORTA_DATA overrides it, or just edit
# this line.  It is the one path here that is about your machine rather than
# about the analysis.
DATA = Path(os.environ.get("AORTA_DATA", __DATA_DEFAULT__)).expanduser()
if not DATA.is_dir():
    raise SystemExit(
        f"{DATA} is not a folder.\n"
        "Set AORTA_DATA to the folder holding one sub-folder per section, "
        "or edit the DATA line above.")
OUT  = APP_DIR / "notebook_out"
OUT.mkdir(exist_ok=True)

# "test" is a scratch copy of the arch sec2 folder, so it is skipped here; drop the
# exclusion if you want it back as a reproducibility check.
sections = sorted(d for d in DATA.iterdir()
                  if d.is_dir() and d.name not in ("__pycache__", "test") and SC.list_images(d))
for d in sections:
    print(f"{len(SC.list_images(d)):4d} fields   {d.name}")
""".replace("__DATA_DEFAULT__", DATA_DEFAULT)))

cells.append(md(r"""
## 1. Assemble one section

`auto_stitch` scores every pair of fields with masked normalised cross-correlation
evaluated by FFT, then adds tiles one at a time, refusing any placement that contradicts
one already made. Tiles it cannot place consistently are parked as separate *islands*
rather than forced in — check `island_sizes` before trusting the result, and open the
browser app to drag the leftovers into place if the largest island is not the whole set.
"""))

cells.append(code(r"""
# The thoracic sections assemble completely, so start with one of those; the arch
# sections are the hard case and are covered in the batch at the end.
section = next((d for d in sections if "thoracic" in d.name), sections[0])
print(section.name)

params = SC.StitchParams(
    coarse_max_dim=320,         # working resolution for the all-pairs sweep
    min_ncc=0.30,               # correlation floor
    min_peak_ratio=1.12,        # how much better the winner must be than a distant rival
    min_overlap_frac=0.06,      # of the smaller tile's tissue area
    rotation_search_deg=0.0,    # raise to a few degrees if the slide was turned between shots
)

t0 = time.time()
ts, poses, matches, stats = SC.auto_stitch(section, params,
                                           progress=lambda f, m: None)
print(f"{len(ts)} fields assembled in {time.time()-t0:.0f} s")
print(json.dumps({k: v for k, v in stats.items() if k != "merge_suggestions"}, indent=1))
"""))

cells.append(md(r"""
### Did it work?

`layout_qc` re-scores every pair of tiles that ends up overlapping, at the offset the
finished layout implies — including pairs that were never matched to each other. A tile
whose neighbours disagree with it is the thing to look at. Hand-checked correct overlaps on
this material score 0.45–0.94 here; contradictions score below 0.2, so there is no grey zone
to interpret.
"""))

cells.append(code(r"""
qc = SC.layout_qc(ts, poses)
print(f"median agreement with neighbours : {qc.score.median():.3f}")
print(f"tiles contradicting a neighbour  : {int(qc.n_conflicts.sum())}")
print(qc.groupby("island").agg(tiles=("index", "size"), median_score=("score", "median")).to_string())

worst = qc.sort_values("score").head(5)[["file", "score", "n_neighbours", "n_conflicts", "island"]]
print("\nweakest tiles:\n", worst.to_string(index=False))
"""))

cells.append(code(r"""
# Render the largest island.  0.35 gives about 0.5 um/px from these 0.17 um/px fields,
# which is ample: the media is tens of microns thick.
RENDER_SCALE = 0.35
main = [i for i, p in enumerate(poses) if p.island == 0]
mosaic = SC.render_mosaic(ts, poses, scale=RENDER_SCALE, indices=main)
mosaic_px_um = ts.px_um / RENDER_SCALE
print(f"mosaic {mosaic.shape[1]} x {mosaic.shape[0]} px at {mosaic_px_um:.3f} um/px")

fig, ax = plt.subplots(figsize=(8, 8 * mosaic.shape[0] / mosaic.shape[1]))
ax.imshow(mosaic); ax.set_axis_off(); ax.grid(False)
ax.set_title(section.name, fontsize=9)
plt.show()
"""))

cells.append(md(r"""
## 2. Straighten the wall

The muscular media is segmented from the muscle-stain concentration, the lumen outline
gives a closed reference curve, and the image is resampled onto (distance along the wall
× depth through the wall). Two things keep that curve on the muscle. Its starting offset is
**local** — the maximum of the media's distance transform along each normal, which is the
middle of the wall at that point; one number for the whole section ties every point to the
thickest muscle anywhere in the mask, and on a branched section that is the junction, where
two walls have merged. It is then **re-centred on the measured middle of the wall** and the
wall sampled again, aiming at the peak of the muscle profile wherever the wall edges cannot
be found — a target that survives the curve being wrong, so a curve that slips off can climb
back rather than staying off and being counted as wall anyway.

Along each sampling line:

* the **luminal surface** is the tissue edge, from total optical density — not from the
  muscle stain, which oscillates by a factor of two inside the media as lamellae and nuclei
  pass under the sampling line and would report a fifty-micron wall as a few microns thick;
* the **media/adventitia junction** comes from one of two rules, `media_edge_rule`, and which
  one is wanted depends on what the section is for — see below;
* the **outer tissue edge** is the far side of the same top-hat, and the junction is clamped
  to it, because the media cannot end outside the tissue.

### Where the media ends, and why it matters which rule

**`collagen-crossing`** ends the media where collagen overtakes muscle *and
stays ahead* for several microns. On a healthy wall that is the external elastic lamina, and
the persistence requirement stops it ending at the first dip. It has one structural flaw, and
no hold length repairs it: it defines the edge of the media **in the same quantity fibrosis is
measured in**. A focal patch inside the media ends the media at the patch, the patch is then
counted as adventitia, and the wall is reported thin — so the more collagen a stretch has, the
less of it is counted. On one thoracic section here, at a patch where collagen ÷ muscle
reaches 1.25 against the vessel's own 0.64, the edge lands 15 µm into a media whose neighbours
are 31 µm thick.

**`muscle-end`**, the default, ends the media where the muscle runs out: the first sustained fall of muscle
below a fraction of this section's *own* muscle level (`muscle_end_fraction`, 0.25 of the
median per-position peak). A fraction rather than an absolute concentration, because staining
intensity varies between slides and an absolute level would make the media thicker on a darker
one. It says nothing about collagen, so the boundary cannot be moved by the thing being
measured and a fibrotic patch keeps its place inside the wall. It is also the steadier of the
two where it counts: muscle falls off sharply at the edge, so anywhere between a quarter and a
twentieth of the plateau moves the answer by about a micron, while the crossing rule compares
two noisy quantities near equality.

Both are the same walk — outwards from the middle of the wall to the first sustained fall of
a profile through a level, interpolated between samples, clamped to the outer tissue edge —
so the two cannot quietly drift apart in anything except the profile and the level.

Which to quote: if the question is fibrosis, `muscle-end`, because `collagen-crossing` is
circular against it. If the question is media thickness on healthy walls, `collagen-crossing`
puts the boundary at the anatomical lamina. Whichever ran is recorded in the totals as
`media_edge_rule`. Section 5's block barely moves between them — `muscle_variation` 0.121
against 0.124 on the same section — because it describes the media itself rather than its
edge; `median_muscle_beyond_edge_fill`, which is defined *against* the edge, swings from 0.72
to 0.60 on that same section, and is the one to read with the rule in mind.
"""))

cells.append(code(r"""
wp = WA.WallParams(
    analysis_px_um=0.5,
    media_quantile=0.55,        # raise to make the muscle mask stricter
    smooth_centerline_um=100.0, # smoothing of the first, lumen-derived curve
    recentre_passes=2,          # then move that curve onto the middle of the wall
    recentre_smooth_um=20.0,    # ...smoothed much more lightly, so it tracks the wall
    arc_step_um=1.0,            # sampling step along the wall
    max_depth_um=260.0,         # half-width of the sampled band
    blue_threshold=None,        # None = the threshold-free rule below
    blue_rule="dominance",      # a pixel is fibrotic when collagen outweighs muscle
    media_edge_rule="muscle-end",   # or "collagen-crossing" -- see above
    muscle_end_fraction=0.25,   # of this section's own muscle, for that rule
    exclude_non_wall=True,      # keep branch junctions out of the totals
)

res = WA.analyze_wall(mosaic, mosaic_px_um, wp)
pd.Series(res.totals).to_frame("value")
"""))

cells.append(code(r"""
strip = WA.straight_view(res)
fig, ax = plt.subplots(figsize=(15, 3.6))
ax.imshow(strip, aspect="auto",
          extent=[0, res.arc_um[-1] / 1000, float(np.nanmax(res.outer_um)) + 60,
                  float(np.nanmin(res.inner_um)) - 60])
ax.plot(res.arc_um / 1000, res.inner_um, color="#1f77b4", lw=0.8, label="luminal surface")
ax.plot(res.arc_um / 1000, res.outer_um, color="#d62728", lw=0.8, label="media / adventitia")
ax.plot(res.arc_um / 1000, res.tissue_outer_um, color="#555", lw=0.6, ls="--", label="outer tissue")
ax.set_xlabel("distance along the wall (mm)"); ax.set_ylabel("depth (µm)")
ax.set_title("the wall, unrolled — real microns"); ax.legend(fontsize=8, loc="lower right")
ax.grid(False)
plt.show()

print(f"sampling curve sits {np.abs((res.inner_um+res.outer_um)/2).mean():.1f} µm from the "
      f"middle of the wall on average")
print(f"unrolling stretch inside the media: {res.totals['stretch_p1']:.2f} to "
      f"{res.totals['stretch_p99']:.2f}   (folded samples: "
      f"{100*res.totals['folded_fraction']:.3f} %)")
"""))

cells.append(md(r"""
### One figure, one horizontal scale

The map at the top carries the traced wall, coloured by distance along it and dotted at whole
millimetres, so a feature in any panel below can be found again on the vessel. Everything under
the map shares the same *distance along the wall* axis: a step in the cumulative curve lines up
with the patch of wall that caused it.

Dividing each sampling line by its own wall thickness turns the media into a true rectangle:
**0 is the luminal surface and 1 the media/adventitia junction everywhere**, whatever the wall
does locally. This is the view for comparing *where in the wall* something sits, because one
row means the same fraction of the media at every position along the vessel — which a row of
the micron-depth image above does not, since the wall is thicker in some places than others.
The axis runs a little past both edges so the lumen and the adventitia stay visible.

Averaging that rectangle down its length gives the section's **curve through the wall** — one
value per relative depth, for each stain. It is an average over *area*, not over sampling
lines: each sample carries the same `1 - κd` weight as the integrals do, so the outside of a
bend is not counted for more than it holds. Positions where the wall was cut are left out, so
the curve and the rectangle agree about which stretches were measured, and `n_rows` records
how many positions reached each depth — not a constant, since a section with a ragged outer
edge has fewer positions reaching 1.0 than reaching 0.5. Section 9 averages these curves
across a cohort, and uses `n_rows` as the weight when it does.
"""))

cells.append(code(r"""
overview = np.clip(SC.resize(mosaic, (mosaic_px_um / res.px_um) * 0.28, "area"), 0, 1)
fig = WA.plot_rectangular(res, overview_rgb=overview)
plt.show()
"""))

cells.append(md(r"""
The wall is **cut** where it stops being one continuous piece — at branches, at stretches with
no image behind them, and wherever the traced path turns over. That last one matters: past a
branch the path comes out on the other wall, so the lumen is then on the opposite side, and
deciding the orientation once for the whole vessel leaves everything after the junction inside
out. Orientation is decided per position instead, and the cuts are drawn as hard vertical
lines because the two sides of a cut are not neighbours on the vessel.

Shaded stretches are where the band is too thick to be a wall — **orange** where the wall
actually forks, **red** where it is thick without forking, which is what a diseased wall would
look like. The fork test is topological rather than metric: the strip that was measured is
subtracted from the media, and a fork is where a substantial piece of wall is left attached.

How far a junction reaches is measured from where the branch's own wall touches this one,
not assumed to be a fixed window around a point: a branch leaves at a shallow angle and runs
alongside its parent for a few hundred microns. Two more things are flagged the same way — a
gap in the band, where a vessel with a 56 µm media measures four, and a band with no muscle
in it, which is adventitia or the collagen wedge at a branch mouth.

By default all of these are left out of the totals, and the cumulative curves follow the same
decision so the curve ends at the reported total. On aortic arch sec2 the junction and the
wedge at its mouth span 20 % of the traced wall and were lifting collagen-per-mm from 26,600
to 39,800 — with them out, the arch's cumulative curve is nearly straight, so the apparent
focal lesion was the branch. Set `exclude_non_wall=False`
to put them back; `collagen_od_um2_per_mm_all_traced` always reports the unfiltered figure,
and `excluded_non_wall` records which way the run went. How much was set aside, and for which
reason, is reported per section rather than summed into one number: `non_wall_length_fraction`
is everything excluded, and `branch_length_fraction`, `thickened_length_fraction`,
`off_wall_length_fraction` (the band lost the wall), `collagen_dominated_length_fraction` (no
muscle in the band — adventitia, or the collagen wedge at a branch mouth) and
`no_data_length_fraction` say what it was. They overlap, so they do not add to the first.

A note on what unrolling costs. Laying a curved band flat cannot preserve area: on the convex
side of a bend the sampling lines fan out and on the concave side they crowd together, by a
factor `1 - κd` at depth `d`. Two consequences are handled rather than ignored:

* every integral through the wall is weighted by that factor, so the outside of a bend is not
  counted more than it should be;
* where the factor goes to zero the sampling lines have crossed and the same tissue would be
  read twice — those samples are dropped, and `folded_fraction` reports how many there were.
  After the sampling curve is re-centred on the middle of the wall it is normally zero.
"""))

cells.append(md(r"""
## 3. Collagen per unit length of wall

At each position along the vessel, the collagen concentration is integrated through the
thickness of the media. The result has units of OD·µm and is a **line density**: it does not
grow just because the wall is thicker there in the trivial sense of counting more pixels —
it is the amount of dye in a one-micron-wide slice through the wall.

Summing that along the vessel gives the cumulative curve. Its *slope* is the local density
and its final value divided by wall length is the per-millimetre figure that compares
across animals.

Three siblings come off the same integral and are worth not confusing. `muscle_od_um_per_um_length`
is the identical quantity for the muscle stain, and is the denominator that makes
`collagen_to_muscle_ratio` a ratio of like with like. `collagen_mean_conc_media` divides the
collagen integral by media *area* instead of by length, so it is a concentration rather than a
line density and does not grow with wall thickness. And thickness is reported twice —
`mean_media_thickness_um` is summed area over summed length, which is what pools correctly,
while `median_media_thickness_um` is the middle position of the wall and is the one to read
when a junction or a lesion has pulled the mean.
"""))

cells.append(code(r"""
p = res.profile
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

axes[0].plot(p.arc_um / 1000, p.media_thickness_um, color="#444", lw=0.8)
axes[0].set_ylabel("media thickness\n(µm)")
axes[0].set_title(f"{section.name}  —  wall {res.totals['wall_length_mm']:.2f} mm", fontsize=10)

axes[1].plot(p.arc_um / 1000, p.collagen_od_um, color="#2c7fb8", lw=0.8, label="in the media")
axes[1].plot(p.arc_um / 1000, p.collagen_adventitia_od_um, color="#9ecae1", lw=0.6,
             label="in the adventitia")
axes[1].set_ylabel("collagen per unit\nlength (OD·µm)")
axes[1].legend(fontsize=8)

axes[2].plot(p.arc_um / 1000, p.cum_collagen_od_um2 / 1000, color="#2c7fb8", lw=1.4)
slope = res.totals["collagen_od_um_per_um_length"]
axes[2].plot(p.arc_um / 1000, p.arc_um * slope / 1000, "k--", lw=0.8,
             label="uniform at the same total")
axes[2].set_ylabel("cumulative collagen\n(10³ OD·µm²)")
axes[2].set_xlabel("distance along the wall (mm)")
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"collagen per mm of wall : {res.totals['collagen_od_um2_per_mm_length']:.0f} OD·µm²/mm")
print(f"total collagen          : {res.totals['collagen_od_um2_total']:.0f} OD·µm²")
"""))

cells.append(md(r"""
Departure of the cumulative curve from the dashed line is the whole point: a straight line
means the collagen is spread evenly around the vessel, and a staircase means it is
concentrated in a few places. Reporting only the total would not distinguish the two.

> **One lumen at a time.** The reference curve is the outline of the *largest* enclosed
> lumen, so a section containing several vessels — an aortic arch with its branches — is
> measured around one of them. `wall_length_mm` is the tell: if it changes when the layout
> changes, the tracer has switched lumen. The thoracic sections have a single lumen and this
> does not arise.
"""))

cells.append(md(r"""
## 4. Blue staining inside the muscle layer

Two complementary measures of the same thing, both restricted to the media:

* **area fraction** — the proportion of the muscle layer where **collagen outweighs muscle**.
  It saturates: a densely fibrotic patch and a very densely fibrotic patch both score 1.
* **stain fraction** — collagen ÷ (collagen + muscle) by integrated concentration. This does
  not saturate, and it is what changes when fibrosis deepens rather than spreads.

The area fraction deliberately has **no threshold in it**. Collagen inside the media is
unimodal, so an automatically chosen threshold has nothing to lock onto: re-centring the
sampling curve on one of these sections moved the collagen percentiles by less than 0.02 and
moved Otsu from 0.778 to 0.632 — taking the reported fibrosis from 9 % to 34 %. A per-section
threshold is also wrong in principle, because a more fibrotic section earns a higher threshold
that cancels part of the difference you are trying to measure. Comparing the two stains pixel
by pixel has neither problem, and survived that same change at 6.6 % → 4.9 %.

The adventitial collagen, which is large and mostly uninteresting, is measured separately so
it cannot leak into either number.
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

axes[0].plot(p.arc_um / 1000, 100 * p.collagen_area_fraction, color="#756bb1", lw=0.7)
axes[0].axhline(100 * res.totals["collagen_area_fraction_media"], color="k", ls="--", lw=0.8)
axes[0].set_xlabel("distance along the wall (mm)"); axes[0].set_ylabel("blue area in media (%)")
axes[0].set_title("where the fibrosis is")

axes[1].plot(p.arc_um / 1000, 100 * p.collagen_fraction_of_stain, color="#2c7fb8", lw=0.7)
axes[1].set_xlabel("distance along the wall (mm)")
axes[1].set_ylabel("collagen / (collagen+muscle)  %")
axes[1].set_title("how blue the media is")

d = res.depth_profile_rel
axes[2].plot(d.depth_relative, d.muscle_mean, color="#d62728", label="muscle")
axes[2].fill_between(d.depth_relative, d.collagen_mean - d.collagen_sd,
                     d.collagen_mean + d.collagen_sd, color="#2c7fb8", alpha=0.18)
axes[2].plot(d.depth_relative, d.collagen_mean, color="#2c7fb8", label="collagen")
axes[2].set_xlabel("relative depth through the media\n(0 = lumen, 1 = adventitia)")
axes[2].set_ylabel("mean concentration"); axes[2].legend(fontsize=8)
axes[2].set_title("across the wall")
plt.tight_layout(); plt.show()

t = res.totals
print(f"blue area fraction of the media : {100*t['collagen_area_fraction_media']:.2f} %")
print(f"collagen fraction of the stain  : {100*t['collagen_fraction_of_stain']:.2f} %")
print(f"rule                            : {t['collagen_rule']}")
print(f"collagen in the adventitia      : {t['collagen_adventitia_od_um2_total']:.0f} OD·µm²"
      f"  ({t['collagen_adventitia_od_um2_total']/max(t['collagen_od_um2_total'],1e-9):.1f}× the media)")
"""))

cells.append(md(r"""
### What an absolute threshold would have done

If you do need the classical absolute-threshold index — to compare with older work, say — pass
`blue_threshold=<number>` and pick the number deliberately; whatever was used comes back as
`collagen_threshold`, left empty when the threshold-free rule ran, so a run cannot be mistaken for
the other kind afterwards. The curve below is why it should
not be picked automatically: the area fraction is on a steep slope across the whole plausible
range, so small changes anywhere upstream move it a long way. The threshold-free rule and the
stain fraction are marked for comparison.
"""))

cells.append(code(r"""
ok = res.media_band & np.isfinite(res.straight_collagen) & np.isfinite(res.straight_muscle)
inside = res.straight_collagen[ok]
grid = np.linspace(np.percentile(inside, 2), np.percentile(inside, 98), 60)
frac = np.array([(inside > g).mean() for g in grid])

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(grid, 100 * frac, color="#756bb1", label="absolute threshold")
ax.axhline(100 * res.totals["collagen_area_fraction_media"], color="#2c7fb8", ls="--", lw=1.1,
           label=f"collagen > muscle: {100*res.totals['collagen_area_fraction_media']:.1f} %")
ax.axhline(100 * res.totals["collagen_fraction_of_stain"], color="#444", ls=":", lw=1.1,
           label=f"stain fraction: {100*res.totals['collagen_fraction_of_stain']:.1f} %")
ax.set_xlabel("collagen threshold"); ax.set_ylabel("blue area in media (%)")
ax.legend(fontsize=8)
plt.show()
"""))

cells.append(md(r"""
### Normalise collagen to muscle

`collagen_od_um2_per_mm_length` is not a fibrosis index. It tracks media thickness at
**r = +0.97** here, and muscle per unit length at +0.98 — arithmetic, not biology, since
integrating a concentration through a thicker wall gives more of everything.

Dividing collagen by muscle cancels it: both are integrated through the same thickness.
`collagen_to_muscle_ratio` drops to r = −0.42 to −0.78 against thickness, and the sign flip
is the tell — a units artefact would have stayed positive.

What is left is a real negative relationship of varying strength: where the media is thicker
it is proportionally less collagenous, clearly so in the thoracic sections (−0.71, −0.78)
and in arch sec4 (−0.64), weakly in the incompletely assembled arch sec2 (−0.42). It is not
bleed from the adventitial collagen band, because restricting to the middle 20–80 % of the
wall leaves it unchanged or slightly stronger.

Where the muscle integral is essentially zero the ratio is undefined rather than large, so
`collagen_to_muscle` is NaN anywhere muscle falls below 5 % of the section's own median.

Quote `collagen_to_muscle_ratio`, or the bounded `collagen_fraction_of_stain`, which is a
monotone transform of the same thing.

### Focal lesions, and what they cannot see

Everything above is an average over the whole wall. A lesion is the other question — *where*,
and *how many* — and it is defined against the same section rather than against an absolute
level. The collagen share, collagen ÷ (collagen + muscle) per pixel, is smoothed to lesion
scale with a 25 µm Gaussian and compared with its own distribution **inside the traced
media**: a patch counts where the smoothed share stands more than `lesion_k_mad` robust
deviations above that section's median, the deviation being 1.4826 × MAD rather than an SD.
Median and MAD, because a section with big lesions in it would otherwise raise its own bar and
hide them. Candidates are opened by a pixel, holes filled, and anything under
`lesion_min_area_um2` — 2 000 µm², about 50 µm across — dropped. Each survivor is reported
with its area, its equivalent diameter, its position *along the wall* in mm of arc so it can
be found again in every other panel, its mean and peak share, and its own collagen-to-muscle
ratio.

Being relative is the point and also the limit, and the two are worth keeping apart. It finds
**focal** disease — a scar, a patch, a fibrotic wedge — against whatever the surrounding
tissue happens to be. It cannot see **diffuse** fibrosis: tissue that is uniformly fibrotic
has no focus to stand out from, and returns no lesions at all while being thoroughly diseased.
That is what the area fraction and the collagen-to-muscle ratio above are for.

An aortic media routinely returns nothing at the default, and should. Its collagen is
lamellar, interleaved with muscle all the way through, so the share distribution is broad and
unimodal and nothing stands three deviations above it. Widening the region to take in the
adventitia makes it worse rather than better, because the tissue is then bimodal —
collagen-rich adventitia against muscle-rich media — and the spread is wider still. So
`lesion_cut_share`, `lesion_share_median`, `lesion_share_mad` and `lesion_share_max` are
reported whether or not anything was found, because **no lesions** has to be a readable answer
and not a silent one.

Read them against each other, because "none" covers two different situations. One section
here gives median 0.39, a maximum anywhere of 0.65 and a cut at 0.76: nothing came close, and
that is a wall with no focal disease in it. The fragmented section 2139 gives median 0.551, a
cut at 0.737 — and a maximum of 0.735, sitting **0.001 under the line**. Same verdict, not
remotely the same confidence: that one is a threshold decision rather than a finding, and
`lesion_k_mad` is the knob. The cell below prints the gap for exactly this reason.

A lesion someone has looked at and rejected is removed **after** detection, by a point clicked
on it, never by lowering the cut. That order is deliberate — lowering the cut would also raise
or drop lesions nobody has looked at, everywhere else in the section, while a click erases
exactly the one patch and touches nothing else. The click is matched to whichever lesion lies
within 20 µm rather than by exact pixel, because the centroid of a curved arc of wall can sit
outside its own shape: on real aortic data the first section tried put it on background.

Unlike section 5's block, lesions would pool the *easy* way across a cohort — a count and an
area are integrals, and `lesion_area_fraction_media` and `lesions_per_mm_wall` rebuild from
summed lesion area over summed media area and summed length exactly as the collagen ratios do
in section 9. They are not in `READOUTS`, though, so today they are carried per section and
stop there.
"""))

cells.append(code(r"""
t = res.totals
if "lesion_cut_share" in t:
    print(f"lesions found                   : {t['n_lesions']}"
          f"   (at {t['lesion_k_mad']:.0f} MAD, minimum {t['lesion_min_area_um2']:.0f} um^2)")
    print(f"  share: median {t['lesion_share_median']:.3f}"
          f" | cut {t['lesion_cut_share']:.3f}"
          f" | highest anywhere {t['lesion_share_max']:.3f}")
else:
    print("no lesion pass -- too little media to set a cut from.")

if t.get("n_lesions"):
    print(f"  area {t['lesion_area_mm2']:.4f} mm^2"
          f"  ({100*t['lesion_area_fraction_media']:.2f} % of the media,"
          f" {t['lesions_per_mm_wall']:.2f} per mm of wall)")
    cols = ["lesion", "arc_mm", "area_um2", "equivalent_diameter_um",
            "collagen_share_peak", "collagen_to_counterstain"]
    print(pd.DataFrame(res.lesions)[cols].round(3).to_string(index=False))
elif "lesion_share_max" in t:
    # Not a failure: read the two numbers above against each other.  A maximum
    # far below the cut is a wall with nothing focal in it; one just under the
    # cut is a decision, not a finding.
    head = t["lesion_cut_share"] - t["lesion_share_max"]
    print(f"  nothing focal -- the highest share anywhere sat {head:.3f} below the cut.")
"""))

cells.append(code(r"""
t = res.totals
print(f"collagen / muscle               : {t['collagen_to_muscle_ratio']:.3f}")
print(f"  vs thickness                  : r = {t['r_thickness_vs_collagen_to_muscle']:+.3f}")
print(f"collagen per length vs thickness : r = {t['r_thickness_vs_collagen_per_length']:+.3f}")
print(f"collagen fraction  vs thickness : r = {t['r_thickness_vs_collagen_fraction']:+.3f}")
print(f"image coverage                  : {100*t['image_coverage_fraction']:.1f} %")
print(f"no-data stretches               : {100*t['no_data_length_fraction']:.1f} % of the traced wall")
"""))

cells.append(md(r"""
## 5. Is the media intact?

Everything above describes the collagen. These describe the muscle: whether the media is
still a solid block, whether it is still layered, and — the measure without which none of the
rest can be read — how sharply this section resolves anything at all. None of them moves a boundary
or enters a collagen total; they are descriptive.

### Muscle unevenness across the wall

Walk outwards from the lumen through an intact media and the muscle stain stays near its own
average. Walk through one that has broken up and the same average arrives with holes in it, so
the walk rises and falls. That is the coefficient of variation of the radial profile — over
its own mean, so a darker slide does not read as a more broken one.

Taken literally the measure does not work, and the reason is worth stating because it is not
a flaw in the staining. **An aortic media is meant to be layered.** Elastic lamellae alternate
with muscle on a 4–5 µm pitch, measured the same on every section tried here, so a section
that resolves them crisply varies more across its wall than one that smears them together.
Raw, this figure ranked a cleanly-cut, intact middle aorta (0.29) almost level with a visibly
fragmented one (0.33). It was reading focus and staining, not pathology.

So the profile is **split by scale**. Smoothing it with a uniform kernel of
`variation_scale_um` — 12 µm, about two and a half lamellae — removes the banding. What
survives is `muscle_variation`: dropouts wider than a lamella, and the fragmentation readout.
What the smoothing removed is `lamellar_contrast`, the banding itself. The split separated
those same two sections 2.0× where the raw figure had separated them 1.2×, and it separates
them *because* their banding is identical — 0.190 against 0.191 — leaving only the coarse
component to tell them apart.

Both are taken over the **middle 70%** of the wall, not all of it. Muscle ramps down at each
boundary because that is what a boundary is, so a profile run edge to edge partly measures
where the edge was drawn — and a thin wall is mostly ramp. Untrimmed, stretches 16–20 µm thick
read 0.44 against 0.20 for stretches 30–60 µm thick *on the same section*. Dropping 15% at
each end removes most of that bias and sharpens the separation; past 20% the trim starts
eating media and both get worse. Requiring a thicker wall instead is the wrong fix: it
discards the thin positions, which are the low-variation ones, so the median rises.

`muscle_variation` describes the media itself, so unlike the pair below it does not rest on
the outer edge having been put in the right place — across both `media_edge_rule` settings it
moves 0.121 → 0.124, while `muscle_beyond_edge_fill` swings 0.72 → 0.60.

### Muscle found past the media edge

Neither edge rule can describe an outer media that has broken up, because both are clamped to
the tissue mask and fragments sit in pale gaps that drag the region under the tissue
threshold. `muscle_beyond_edge_um` is how far muscle reaches past the edge and
`muscle_beyond_edge_fill` how much of that reach is muscle. **The fill is the discriminating
one.** Reach alone separates nothing: an arch with 15 µm of solid muscle past its edge merely
had its edge stop short, and reads the same as a middle aorta with 12 µm of fragments.
`fragmented_length_fraction` is the share of counted wall where muscle reaches more than 12 µm
out and less than 80% of that reach is muscle. This one is deliberately **not** smoothed along
the wall — intact sections have no jitter in it at all, so the unsteadiness is the signal, and
a 10 µm smooth halved the separation.

### The layers themselves

`lamellar_spacing_um` is the distance between successive lamellae, from the peaks of the same
residual `lamellar_contrast` measures — standard aortic morphometry, and the robust one here
(5.0, 5.8 and 6.0 µm on the three well-resolved sections). `lamellar_disorder` is the spread
of those gaps over their mean: evenly stacked layers give a low figure whatever the staining,
layers that have merged, split or gone missing give a high one.

A peak counts as a lamella only if it stands out by `lamella_prominence` (10%) of the wall's
own mean muscle level. **Prominence must be absolute, never relative to the profile's own
spread** — a spread-relative threshold is self-scaling, so a flat, blurred wall still yields
the largest wiggles in its own noise and reads as beautifully layered.

`lamellar_disorder` is the weakest measure in this section and is reported with its
`lamellar_coverage` for that reason: it needs three lamellae in the profile, so on a soft
section most of the wall cannot be judged at all. It separates the fragmented section from the
crisply layered one 1.8×, but only about 1.3× from the intact group as a whole, at every
threshold tried. It is worth keeping because it is independent of everything else here —
r ≈ 0 against `muscle_variation` on every section — not because it is decisive.

### What none of the above can be read without

Every figure in this section is attenuated by blur, and blur is not a constant of the study:
across four sections of the same experiment the luminal edge rises in 2.0, 2.0, 4.0 and 4.0 µm.
`section_edge_width_um` is that rise, 10% to 90%, measured across the luminal boundary — the
lumen is empty and the media starts at once, so it reports the section's own resolution
without reference to any lamella. It computes on effectively the whole wall.

It is what keeps the layering figures interpretable, and the case that makes the point is
real: of two sections that rise equally sharply at 2.0 µm, one bands at 0.191 and the other at
0.107, so **that gap is structure, not optics** — while a third section's feeble 0.074 arrives
with a 4.0 µm edge and is mostly optics. Without an edge width beside them those two cases are
indistinguishable. Compare layering numbers between sections only at a similar edge width;
where they differ, the softer section's figures are floors, not measurements.

Three other formulations were measured and rejected, so they are not worth retrying: depth
autocorrelation (0.20–0.22 everywhere, no separation), profile continuity along the wall (it
ranks the *intact* sections lowest — it reads contrast), and gradient anisotropy in the
straightened frame (it puts the blurred section below the fragmented one, the same focus
confound as the raw variance).

Whether a high `muscle_variation` is real medial degeneration or a worse section is a question
about the samples, not about the images. The edge width is what lets that question be asked.
"""))

cells.append(code(r"""
t = res.totals
print(f"muscle variation across the wall : {t['muscle_variation']:.3f}")
print(f"  lamellar contrast (excluded)   : {t['lamellar_contrast']:.3f}")
print(f"lamellar spacing                 : {t['lamellar_spacing_um']:.2f} um")
print(f"lamellar disorder                : {t['lamellar_disorder']:.3f} "
      f"(measurable on {100*t['lamellar_coverage']:.0f} % of the wall)")
print(f"section edge width               : {t['section_edge_width_um']:.2f} um")
print()
print(f"muscle past the media edge       : {t['median_muscle_beyond_edge_um']:.1f} um, "
      f"{100*t['median_muscle_beyond_edge_fill']:.0f} % solid")
print(f"wall with a fragmented outer media: {100*t['fragmented_length_fraction']:.1f} %")
"""))

cells.append(md(r"""
The two blocks above are near-independent: per position, `muscle_variation` and
`muscle_beyond_edge_fill` correlate at r = +0.02. One looks inside the wall and the other
outside it, so a section can fail either on its own.
"""))

cells.append(md(r"""
## 6. More than one vessel in the frame

`analyze_vessels` measures every vessel in the frame as its own wall, and this is what the
app and the batch both run: an arch section routinely catches the vessel twice — the
ascending and descending limbs of the same arch — and measuring one and dropping the other
throws away half the wall. The batch writes one row per vessel and the pooling step adds them
back together by segment, which is the unit the comparison is made on anyway.

A vessel is found in one of two ways, and both are needed. A wall photographed whole
**encloses a lumen**, and the hole is what finds it. A hole counts on two tests — **size**,
and **the wall around it**: every hole in the band is enclosed by media by construction, so
being surrounded says nothing, while being surrounded by 50 µm of muscle rather than a
two-pixel sliver says a great deal.

A wall that does not close encloses nothing, so there is no hole to find it by, and it is
found instead as a **separate piece of the band**. Those are weighed on the matching pair of
tests: **size**, the same threshold the band itself was built with, and **shape**, long
compared with its thickness — a disc scores 1.8 whatever its size, an arc of aorta 40 to 60,
which is what keeps a lump of fat or dense stain out of the count. Rings are settled first
and their pieces taken out of the running, so a ring broken by a fold stays one vessel.

This matters on exactly the sections where it is easiest to miss. One arch caught at a
branch holds two walls, neither of which closes; the band is two pieces of 395,000 and
141,000 µm². Enumerating lumina alone finds none, so the whole band was skeletonised as one
and the longest path won — and it was the *smaller* vessel that won it, because that walk
starts from whichever skeleton endpoint the raster reaches first. The section reported one
2.9 mm wall. It now reports two, of 3.7 and 2.9 mm.

Nothing is dropped in silence either way: rejected holes and rejected pieces both come back
with the numbers that decided them.

A branch cut near its origin shares the parent's lumen and stays part of the parent's band;
those are reported by `find_branch_nodes` as attachment points on the parent instead.

When the automatic answer is wrong, `analyze_wall(..., guide_um=...)` takes an outline drawn
by hand, in microns from the top-left of the image; the **Vessel wall** tab has an editor for
drawing it. It does not have to be accurate — the re-centring pass pulls a rough line onto
the middle of the media exactly as it does the automatic one.

The editor cuts as well as redraws, and the two answer different questions. Redrawing a
stretch fixes a trace that went the wrong way. Cutting the ends fixes one that went the right
way and then kept going — into the junction two limbs of an arch share, or off along the
vessel next to it. There is nothing to redirect there: the wall stops, and where it stops is
a judgement about the specimen. Cutting is the same mechanism, since a shorter `guide_um` is
just a shorter wall; on the arch section above it takes vessel 1 from 4.645 mm traced to
3.435, and from four branch nodes to one.

An outline belongs to **one** wall, and the section's other walls are still found the usual
way while it is edited — the app sends their ranks alongside the outline and the server
measures them automatically, so they come back identical to the run they came from. They are
deliberately *not* re-measured from the outlines already on screen: a curve does not survive
a round trip through the re-centring untouched. Feeding vessel 2's own trace straight back
moves its traced length by 5 % and its blue area by a quarter, so a vessel nobody touched
would drift every time its neighbour was redrawn.

Each vessel's own row records the bookkeeping: `n_vessels` found in that section, of which
`n_lumina` were found by an enclosed hole and `n_lumina_rejected` holes were turned down,
`lumen_rank` which vessel this row is (rings first, then arcs), `guided` whether the outline
was drawn by hand, and `closed_ring` whether the wall came round on itself — an open trace is a section that was not fully assembled, and it measures
fine but it is not a complete ring.
"""))

cells.append(code(r"""
vessels = WA.analyze_vessels(mosaic, mosaic_px_um, wp)
pd.DataFrame([{
    "vessel": v.totals["lumen_rank"] + 1,
    "wall_mm": round(v.totals["wall_length_mm"], 3),
    "media_um": round(v.totals["mean_media_thickness_um"], 1),
    "collagen_per_mm": round(v.totals["collagen_od_um2_per_mm_length"]),
    "blue_area_%": round(100 * v.totals["collagen_area_fraction_media"], 2),
    "branches": v.totals["n_branches"],
} for v in vessels])
"""))

cells.append(md(r"""
## 7. Save everything
"""))

cells.append(code(r"""
name = section.name.replace(" ", "_").replace(",", "")
paths = WA.save_results(res, OUT / name, name, overview_rgb=overview)
SC.save_layout(OUT / name / "layout.json", ts, poses, matches, params)
qc.to_csv(OUT / name / "tile_quality.csv", index=False)

for k, v in paths.items():
    print(f"{k:24s} {v}")
"""))

cells.append(md(r"""
## 8. All sections at once

Runs the whole thing unattended and collects one row per section. Read `n_islands` and
`conflicts` before reading anything else: a section that did not assemble into a single
island has not been measured in full, and the honest move is to open it in the browser app,
drag the stray pieces into place, press **A** to let correlation settle them, and re-run
from the saved layout.
"""))

cells.append(code(r"""
# Every section's curve through the wall, stacked as they are measured.  This
# is what section 9 averages into one curve per region, age and genotype.
depth_frames = []


def analyse_folder(folder: Path, render_scale: float = 0.35) -> dict:
    ts, poses, matches, stats = SC.auto_stitch(folder, params)
    qc = SC.layout_qc(ts, poses)
    main = [i for i, p in enumerate(poses) if p.island == 0]
    mosaic = SC.render_mosaic(ts, poses, scale=render_scale, indices=main)

    d = OUT / folder.name.replace(" ", "_").replace(",", "")
    d.mkdir(parents=True, exist_ok=True)
    SC.save_layout(d / "layout.json", ts, poses, matches, params)

    meta = CO.parse_section_name(folder.name, folder=str(folder))
    for field, value in CO.read_overrides(folder.parent).get(folder.name, {}).items():
        setattr(meta, field, value)
    row = {"section": folder.name, "name": folder.name,
           "mouse": meta.mouse, "genotype": meta.genotype, "age": meta.age,
           "segment": meta.segment, "slide": meta.slide, "section_no": meta.section,
           "n_fields": len(ts),
           "placed": stats["island_sizes"][0], "n_islands": stats["n_islands"],
           "conflicts": int(qc.n_conflicts.sum()),
           "median_agreement": round(float(qc.score.median()), 3)}
    try:
        res = WA.analyze_wall(mosaic, ts.px_um / render_scale, wp)
        stem = folder.name.replace(" ", "_").replace(",", "")
        saved = WA.save_results(res, d, stem,
                                overview_rgb=np.clip(SC.resize(mosaic, 0.30), 0, 1))
        # Section 9 rebuilds the media-quality medians from these positions.
        row[CO.PROFILE_COLUMN] = saved["profile"]
        dr = res.depth_profile_rel.copy()
        for pos, key in enumerate(("segment", "age", "genotype", "mouse", "name")):
            dr.insert(pos, key, row.get(key, ""))
        depth_frames.append(dr)
        row.update({k: res.totals[k] for k in (
            # The integrals first: these are what section 9 pools over.
            "wall_length_um", "media_area_um2",
            "collagen_od_um2_total", "muscle_od_um2_total",
            "collagen_adventitia_od_um2_total", "collagen_area_um2_per_mm_length",
            "wall_length_mm", "traced_length_mm", "mean_media_thickness_um",
            "sd_media_thickness_um", "collagen_od_um2_per_mm_length",
            "collagen_od_um2_per_mm_all_traced", "collagen_area_fraction_media",
            "collagen_fraction_of_stain", "media_area_um2",
            "n_branches", "branch_length_fraction", "thickened_length_fraction",
            "collagen_to_muscle_ratio", "r_thickness_vs_collagen_to_muscle",
            "n_segments", "turned_over_fraction", "no_data_length_fraction",
            "arc_step_um",
            "r_thickness_vs_collagen_per_length", "r_thickness_vs_collagen_fraction",
            # Section 5's media-quality block.  Kept per section here, and
            # averaged rather than summed on the way up: these are medians,
            # not integrals, so section 9 weights them by wall length instead
            # of adding them.
            "muscle_variation", "lamellar_contrast", "lamellar_spacing_um",
            "lamellar_disorder", "lamellar_coverage", "section_edge_width_um",
            "median_muscle_beyond_edge_um", "median_muscle_beyond_edge_fill",
            "fragmented_length_fraction",
            # Section 4's lesions.  The diagnostics travel with the count,
            # because "none found" only means something beside the cut it was
            # not found against.
            "n_lesions", "lesion_area_mm2", "lesion_area_fraction_media",
            "lesions_per_mm_wall", "lesion_cut_share", "lesion_share_max")})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


rows = []
for folder in sections:
    print("…", folder.name, flush=True)
    rows.append(analyse_folder(folder))

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "all_sections.csv", index=False)
summary
"""))

cells.append(code(r"""
ok = summary[summary.get("error").isna()] if "error" in summary else summary
if len(ok):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    lbl = [s[:26] for s in ok.section]
    axes[0].barh(lbl, ok.collagen_od_um2_per_mm_length, color="#2c7fb8")
    axes[0].set_xlabel("collagen per mm of wall\n(OD·µm²/mm)")
    axes[1].barh(lbl, 100 * ok.collagen_area_fraction_media, color="#756bb1")
    axes[1].set_xlabel("blue area in media (%)")
    axes[2].barh(lbl, ok.mean_media_thickness_um, xerr=ok.sd_media_thickness_um,
                 color="#d62728", error_kw={"lw": 0.8})
    axes[2].set_xlabel("media thickness (µm)")
    for a in axes[1:]:
        a.set_yticklabels([])
    plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""
## 9. Sections, mice, genotypes

One section is not an experiment. Three things have to happen between the table above and a
claim about a genotype, and each is somewhere a result can be quietly lost:

* **Metadata.** Folder names are read for mouse, genotype, age, segment, slide and section.
  Whatever the parser makes of them is printed, and a `metadata.csv` beside the folders
  overrides it — a naming convention is a convention, not a guarantee.
* **Pool by adding integrals, never by averaging ratios.** Two slices of one segment in one
  mouse are one wall sampled twice, so the pooled ratio is *total* collagen over *total*
  muscle. Averaging the two section ratios weights a 0.4 mm off-cut the same as a 4 mm ring.
  `mean_media_thickness_um` comes back as summed area over summed length, which is exactly
  the length-weighted mean thickness.
* **Medians are not added; they are rebuilt from positions.** Section 5's block is the
  exception to the bullet above, because nothing in it is an integral: every one of the
  nine is a median or a spread over arc positions, and two medians added together mean
  nothing. `CO.summarise(measured, medians=...)` offers two ways up and one way out.

  `medians="exact"` is the default here and is the same move as summing integrals, one
  level down. Each section's per-position values are kept — that is what `profile_csv` in
  the batch row points at, the along-wall table written beside its results — and a mouse's
  sections are stacked into one long list of positions, which is reduced once. Pooling
  three sections gives the median of all their positions, not the mean of three medians.
  A single section reproduces its own reported number exactly, which is the check that the
  two reductions agree: run `test_cohort_pooling.py`, and there is a real-section version
  of it that agrees to float32 precision on all nine. The cost is one pass over files
  already on disk.

  `medians="weighted"` needs nothing but the table, and is what any group whose profiles
  cannot all be read falls back to. Each section is weighted by the wall behind it, so the
  0.4 mm off-cut counts a tenth of the 4 mm ring rather than equally. Two of the nine come
  out *exactly* right that way too: `lamellar_coverage` and `fragmented_length_fraction`
  are already fractions of wall length. `lamellar_spacing_um` and `lamellar_disorder` are
  weighted by length × coverage rather than length, because only the part of the wall that
  yielded three lamellae produced them — a section 47% covered would otherwise weigh as
  much as one 90% covered. The other five are approximations, and it is worth being clear
  about what kind: a weighted mean of medians is not the median of the pooled positions.
  It is close while a mouse's sections agree and wrong in the tail when one of them is
  bimodal — one bad section among three good ones, which is exactly the case these measures
  exist to catch. Two equally long sections, one clean at 0.05 throughout and one half
  clean and half ruined at 0.40, give 0.05 from the positions (three quarters of that
  mouse's wall is clean, and it says so) and 0.1375 from the section medians, which is a
  value no position in that mouse has.

  Which of the two happened is in the pooled table, as `medians_from`, not only in the
  call. A group falls back whole or not at all: rebuilding from two of a mouse's three
  profiles would quietly describe a different mouse. Sections run at different
  `arc_step_um` also fall back, because a position is only interchangeable with another
  while it is the same length of wall. `medians="off"` leaves the block out of the pooled
  tables entirely, rather than let an approximation be read as exact.
* **The mouse is the unit.** Sections and slides are repeated measures of one animal.
  Counting them as independent inflates n and shrinks p for reasons that have nothing to do
  with the biology, so genotypes are compared across per-mouse values with `n_mice` printed
  beside `n_sections`.
* **The curve through the wall goes up the same ladder, held apart by age.** Each section's
  curve (section 2) is averaged into its mouse *at every depth separately*, weighted by
  `n_rows` — the positions of that section which reached that depth — so a 0.4 mm off-cut
  does not pull a mouse's curve as hard as a 4 mm ring, and a section whose outer edge is
  ragged stops contributing where it ran out of wall rather than dragging the curve down
  there. Mice are then averaged into genotypes unweighted, because the mouse is the unit: n
  is animals and the shaded band is ± SD **between animals**, not the variation along one
  wall, which the per-section file carries under the same name.

  The grouping is region × age × genotype. Age is in it because baseline wall composition
  moves with age far more than most genotypes move it within one age, so a group mixing a
  12-week and a 32-week animal has a spread that is mostly calendar. It is read out of the
  folder name like everything else and overridden by `metadata.csv` the same way; sections
  whose name says nothing about age land together in `unknown` rather than being spread
  across the others. **Note that age is a key for this curve only** — `by_genotype` and
  `comparison` still group on region and genotype alone, so a multi-age study pools ages in
  those two tables.

  The figure is two panels per group, muscle and collagen, rather than one carrying both:
  four curves in two different meanings of colour is not a figure anyone reads. Written as
  `cohort_depth_relative.csv`, `cohort_depth_plot.png` and `.svg`.
"""))

cells.append(code(r"""
measured = summary[summary.get("error").isna()] if "error" in summary else summary
tables = CO.summarise(measured, medians="exact")   # see the bullet above
CO.write_summary(tables, OUT)

cols = ["genotype", "mouse", "segment", "n_sections", "wall_length_mm",
        "mean_media_thickness_um", "collagen_to_muscle_ratio", "collagen_area_fraction_media"]
print("per mouse, per segment")
print(tables["by_mouse_segment"][cols].round(3).to_string(index=False))
print("\nper mouse (all segments pooled)")
print(tables["by_mouse"][cols].round(3).to_string(index=False))

# Section 5's block, if the batch carried it.  Rebuilt from every counted
# position of each mouse's sections, not summed and not averaged -- and
# `medians_from` says so per row, since a mouse whose profiles could not all
# be read falls back to the weighted mean.
media = [c for c in CO.SECTION_MEDIANS if c in tables["by_mouse"].columns]
if media:
    print("\nmedia quality per mouse")
    print(tables["by_mouse"][["genotype", "mouse", "n_sections", "medians_from"] + media]
          .round(3).to_string(index=False))
"""))

cells.append(code(r"""
# What pooling by sums buys over averaging the section ratios.  They agree when the
# slices are the same length and part company when they are not, which is the usual case.
g = ["genotype", "mouse", "segment"]
naive = measured.groupby(g, dropna=False)["collagen_to_muscle_ratio"].mean().rename("mean_of_ratios")
pooled = tables["by_mouse_segment"].set_index(g)["collagen_to_muscle_ratio"].rename("pooled")
pd.concat([pooled, naive], axis=1).assign(
    difference=lambda d: d.pooled - d.mean_of_ratios).round(4)
"""))

cells.append(code(r"""
# One panel per readout; bars are group means, dots are animals.  With a handful
# of mice the dots are the figure: a bar cannot say whether a difference is every
# animal agreeing or one animal carrying the group.
fig = CO.plot_cohort(tables, unit="mouse_segment", error="sd",
                     title=DATA.name)
fig.savefig(OUT / "cohort_plot.png", dpi=150, bbox_inches="tight")
with plt.rc_context({"svg.fonttype": "none"}):
    fig.savefig(OUT / "cohort_plot.svg", bbox_inches="tight")   # vector, text still text
plt.show()
"""))

cells.append(code(r"""
# The curve through the wall, region by region and age by age, genotypes
# overlaid.  Sections into a mouse weighted by how much wall each contributed
# at that depth, then mice into a genotype -- so the band is the spread
# between animals and n is the number of them.
if not depth_frames:
    print("no section produced a depth profile -- nothing to average.")
else:
    depth_long = pd.concat(depth_frames, ignore_index=True)
    depth_avg = CO.average_depth_profile(depth_long)
    depth_long.to_csv(OUT / "batch_depth_relative.csv", index=False)
    depth_avg.to_csv(OUT / "cohort_depth_relative.csv", index=False)

    # `stains_together=True` turns the grid the other way up: one panel per
    # condition with both stains in it, for reading where the collagen sits
    # relative to the muscle inside one wall rather than comparing genotypes.
    # `spec=PlotSpec(axes_width_mm=50.8, axes_height_mm=50.8)` pins the
    # plotting area itself at 2 x 2 in, whatever the labels around it need.
    fig = CO.plot_depth_profiles(depth_avg, title=DATA.name)
    fig.savefig(OUT / "cohort_depth_plot.png", dpi=150, bbox_inches="tight")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / "cohort_depth_plot.svg", bbox_inches="tight")
    plt.show()
"""))

cells.append(code(r"""
comp = tables["comparison"]
if len(comp):
    show = comp[comp.readout.isin(["collagen_to_muscle_ratio", "mean_media_thickness_um"])]
    print(show[["segment", "readout", "genotype_a", "genotype_b", "n_a", "n_b",
                "mean_a", "mean_b", "difference", "p"]].round(4).to_string(index=False))
else:
    print("only one genotype in this batch -- nothing to compare.")
    print(tables["by_genotype"][["segment", "genotype", "readout", "n_mice",
                                 "n_sections", "mean", "sd"]].round(4).to_string(index=False))
"""))

cells.append(md(r"""
---

### Where the assembly can still go wrong

Vessel wall looks much the same everywhere, and perivascular fat looks much the same as
other perivascular fat, so some pairs of fields genuinely cannot be told apart from the
image alone. The assembler is built to say so rather than to guess: it refuses placements
that contradict what is already down, and reports a proposed island merge with its
supporting evidence and its conflicts instead of applying it.

When `stats["merge_suggestions"]` is non-empty, it is telling you that a group of fields has
strong evidence for a position that is already occupied — which means one of the two groups
is misplaced. That is a question about the specimen, not about the software, and it is
answered fastest in the browser app with the two islands on screen.
"""))

# nbformat 4.5 wants a unique id per cell; numbered, so a rebuild writes the same file.
nb = {
    "cells": [dict(c, id=f"cell-{i:03d}") for i, c in enumerate(cells)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
