# Aortic Straightener and Fibrosis Quantifier

**ASFQ** is a browser app and a Python library for measuring fibrosis in Masson's-trichrome
sections, and for building the sections in the first place:

1. **Assembling** a folder of overlapping microscope fields into one image. Generic — it
   knows nothing about any particular organ and works for any specimen photographed as a
   set of overlapping snapshots.
2. **Measuring the aortic wall** in the assembled section: straightening it, collagen per
   unit length of wall, and blue staining through the muscle layer.
3. **Measuring fibrosis by area** in heart and kidney, where there is no wall to follow.

One tab assembles; each organ then has its own tab, with its own batch at the bottom of it.

Nothing below assumes anything is already installed — not even Python. Start here:

---

## Setting it up on a new machine

**Python 3.10 or newer**, and nothing else installed by hand. Everything below comes from
`requirements.txt`: `numpy scipy scikit-image opencv-python pandas matplotlib flask pillow
statsmodels czifile tifffile imagecodecs`. macOS ships 3.9 at `/usr/bin/python3` and Windows
ships none at all, so on a machine that has never run Python this is the one thing to install
first — from [python.org](https://www.python.org/downloads/), and on Windows tick **Add
python.exe to PATH** in the installer. The launchers check the version before they do
anything else and say so rather than failing inside `pip` several minutes later.

### Getting it

```bash
git clone https://github.com/righeile/asfq.git
cd asfq
```

Or **Code → Download ZIP** on that page and unzip it anywhere. A ZIP does not carry the
execute bit, so after unzipping on macOS or Linux:

```bash
chmod +x run_mac.command
```

### The short way

Double-click `run_mac.command` (macOS) or `run_windows.bat` (Windows); on Linux,
`bash run_mac.command`. Each builds a `.venv` next to the app on first run, installs the
dependencies into it, starts the server and opens the browser on it. Nothing is installed
system-wide and nothing outside the app folder is touched. Close the window, or press
**ctrl+C** in it, to stop the app; your work is saved in the session, not in the window.

The launcher rebuilds the environment whenever `requirements.txt` changes, so pulling a
newer version and double-clicking again is the whole upgrade. It also checks who already
holds port 8765 before starting: a copy left running from an earlier day keeps serving the
current interface off disk with older code behind it, which looks like a broken app rather
than a stale one, so it stops that copy and starts a fresh one instead of quietly handing you
back to it. And it opens the browser only once the server answers, rather than after a
fixed wait that elapsed whether or not anything had started.
Set `PORT` to use a different port.

### When the first double-click does nothing

Both systems refuse to run a file that arrived from the internet until they are told twice,
and neither says which file it means.

* **macOS** — *"cannot be opened because it is from an unidentified developer"*, or nothing
  happens at all: **right-click `run_mac.command` → Open**, then **Open** again in the
  dialog. Once per copy of the app. If the Terminal opens and closes immediately instead,
  the execute bit is missing: `chmod +x run_mac.command` (see above).
* **Windows** — *"Windows protected your PC"*: **More info → Run anyway**. If a Microsoft
  Store page opens instead, Python is not installed — the `python` on a fresh PC is a stub
  that opens the Store and does nothing else. Install it from python.org as above.

Everything else the launcher can hit, it explains in its own window, which is why the window
stays open and waits for **Enter** instead of vanishing with the message in it.

### The manual way

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/python app.py                        # then open http://127.0.0.1:8765
```

An existing conda or system environment works too, as long as it satisfies
`requirements.txt` — the app imports nothing else and writes nothing outside the folder it
is told to write to.

### Checking it works

There is no bundled test dataset — the sections this was built on are not in the repo. Any
folder of overlapping microscope fields exercises the **Assemble** tab; any single image of
a vessel cross-section exercises **Vessel wall** via *image file*, which is the faster check
because it skips stitching entirely. `curl -s localhost:8765/api/health` answers
`{"app": "asfq", ...}` and is what the launchers use to tell this app apart from anything
else on the port — including a copy of it left running from an earlier day.

---

## Assembling

Point the app at a folder, press **Auto-stitch**, then deal with whatever is left over.

| | |
|---|---|
| drag a tile | move it |
| **shift**+drag, or **I** then drag | move its whole island |
| **A** | align the selection against its neighbours |
| **shift+A** | align everything |
| **L** | lock — locked tiles are never moved again, by anything |
| **[** **]** | rotate (with **shift**, 5° steps) |
| arrows | nudge one pixel (with **shift**, twenty) |
| **F** | fit the view |
| **V** / **R** / **Q** | move · rectangle select · lasso select |
| **⌘A** | select all · **E** grows the selection to whole islands |
| **Esc** | back to the move tool, deselect |
| **⌘Z** / **ctrl+Z** | undo · **⇧⌘Z** or **ctrl+Y** redo |

**Selecting several pieces.** The rectangle takes every tile it touches; the lasso takes
every tile whose centre or a corner falls inside the loop, because a lasso is thrown round a
cluster rather than traced along tile edges. Tick *take only fully enclosed tiles* when you
mean exactly the pieces inside the line. **shift** adds to the selection and **alt**
subtracts, so a rough sweep can be corrected instead of redrawn; pressing an
already-selected tile drags the selection rather than starting a new region, so
marquee-then-move needs no mode switch; **space**+drag pans in any tool. Align, lock, nudge
and drag all act on the whole selection.

Undo covers every change to the layout — drags, nudges, rotations, locks, alignments, an
applied merge, and auto-stitch itself — eighty steps deep. Only the layout is recorded, so
undo puts the pieces back and leaves your view, selection and settings alone; runs of
nudges collapse into one step so a held arrow key does not bury the history.

**Drag, then align.** You supply the topology — which piece belongs roughly where, the part
software gets wrong on self-similar tissue. Correlation then supplies the last few pixels,
the part people get wrong. After a drag, the tile is re-matched against its new neighbours
within a small radius of where you dropped it, so your decision is refined and never
overruled. A test: displacing a placed field by 150 × −120 px and pressing **A** returns it
to within **0.07 px**, on five neighbour votes at correlation 0.95–0.96.

Dragging a whole island and pressing **A** aligns it as a **rigid body** — every member that
touches something outside votes on one common shift and the weighted median is applied to
all of them. Aligning members individually would take a correctly solved group and pull it
apart.

**Settle islands** is the pass for after the hand work: once the pieces are roughly in
place, it settles every island against the others, each one still moving as a whole, and
repeats until nothing shifts. **Align all** is the wrong tool at that point — it nudges
tiles one at a time, which is exactly what pulls a solved island apart — and aligning one
island at a time cannot converge when three pieces all want to move, since each is aligned
against the others' un-settled positions. The largest island holds still so the mosaic does
not drift bodily, and an island holding a **locked** tile holds still too: a hand-placed
anchor is the one thing not up for revision. Displacing half of a solved thoracic mosaic by
6, 18 or 40 px and settling it returns it to the same 1.3 px residual in either direction —
the answer does not depend on how wrong the starting point was. Islands keep their identity;
merging is a claim that two pieces are one, and it stays with **A**, made deliberately,
rather than happening as a side effect of tidying up.

**Merge selected islands** is that deliberate claim, made by hand. Automatic
merging is a judgement — **A** joins an island only when the settled result
contradicts nothing, and the suggestion panel offers a shift it worked out
itself — and neither can help when the evidence is genuinely ambiguous and you
can nonetheless see where the piece belongs. Select a tile in each piece and
merge: the tiles stay exactly where you put them and only the island label
changes, so there is no placement for it to get wrong. Conflicts are still
reported afterwards rather than being silenced by the decision, since forcing
the merge settles what the pieces *are*, not whether they agree. The largest
piece keeps its name, so the main mosaic stays island 0 and the export and
analysis pickers do not renumber underneath you.

Positioning by hand is the other half and already works: tick **drag whole
island** to move a piece bodily, arrow keys nudge a pixel at a time (twenty
with **shift**), and **Settle islands** takes it from roughly right to
pixel-right. The usual order is drag, settle, then merge.

Tile borders are coloured by how well each field agrees with its neighbours in the finished
layout; red lines join fields that contradict each other.

### Island merge suggestions

Press **Look for island merges**. Each suggestion reports how much tissue would agree after
the move, and what it would collide with. A suggestion with strong support *and* conflicts
means one of the two islands is internally misplaced — the app says so rather than guessing.
On the arch section it reports, correctly, that the six parked fields want a position
occupied by fields 7136–7176, with 59,606 px of agreeing tissue behind the claim.

---

## The wall analysis

Everything is measured on **stain concentrations** from colour deconvolution, with the stain
vectors estimated from the section itself (Macenko). Optical density is linear in dye
amount, so it can be integrated over distance; a hue-and-saturation score cannot, because it
confounds how much dye is present with how dark the field was.

The media is segmented from the muscle-stain concentration, the lumen outline provides a
closed reference curve, and the image is resampled onto (distance along the wall × depth
through the wall). That curve is then **re-centred on the measured middle of the wall** and
the wall sampled again.

Two things keep the curve on the muscle, and both had to be there:

* **The starting offset is local.** Walking outwards from the lumen edge, the distance
  transform of the media rises to a maximum halfway through the band and falls again; that
  maximum is the middle of the wall *at that point*. Using one high percentile of the same
  transform for the whole section instead ties every point's offset to the thickest muscle
  anywhere in the mask, and on a section with a branch that is the junction, where two walls
  have merged. On the aortic arch it gave 60 µm against a wall whose own half-thickness is
  28: the starting curve began outside its own media at 93 % of positions, and a constant
  offset of a concave outline also loops back on itself at every branch mouth.
* **Re-centring can recover.** Its usual target is the midpoint between the measured wall
  edges, which is the right target while the curve is on the wall and no target at all once
  it is not — with no muscle underneath, the edge finder reports a band a few microns thick
  and the correction is refused as implausible. The peak of the muscle profile is a target
  that survives the curve being wrong, because it is a fact about the image rather than
  about boundaries just measured. Without it, a curve that slipped off stayed off: on arch
  sec4 that locked a quarter of the traced length into the adventitia, 18 µm from a media
  43 µm thick, and it was still counted as wall.

Together these take arch sec4 from 64 % of positions on the muscle band to **100 %**, with
the measured thickness agreeing with the segmentation to 4 % (55 µm against 57 µm). On a
plain thoracic ring, where the old curve was already right, nothing moves by more than
0.3 %. Re-centring also takes the off-centre error on the thoracic aorta from 8.5 µm RMS to
3.0 µm and the folded fraction of the wall to zero.

A stretch that still measures no wall is not counted as one. Three things are flagged and
kept out of the totals rather than averaged into them: a **fork**, over the span the
branch's own wall actually touches this one rather than a fixed window around a point; a
**gap**, where a vessel with a 56 µm media measures four; and a stretch where the band is
there but has **no muscle in it**, which is adventitia or the collagen wedge at a branch
mouth. That last test is the media/adventitia rule applied to the band as a whole, and it is
not a threshold on fibrosis — collagen laid down between muscle cells leaves muscle dominant
through the band, whereas these stretches have no muscle at all. On arch sec2 one of them
was reporting a local collagen ÷ muscle of 13 against a vessel whose own is 0.35.

* the **luminal surface** is the tissue edge, from total optical density;
* the **media/adventitia junction** is where the **muscle runs out** — the first sustained
  fall of muscle below a quarter of this section's own muscle level (`media_edge_rule`,
  default `muscle-end`). It says nothing about collagen, so a patch of fibrosis inside the
  media keeps its place in the wall instead of truncating the measurement of itself. The
  alternative, `collagen-crossing`, puts the junction where collagen overtakes muscle and
  *stays ahead* for several microns — the external elastic lamina on a healthy wall, and the
  better answer when the question is media thickness rather than fibrosis.

### Readouts

| column / total | units | meaning |
|---|---|---|
| `collagen_od_um` | OD·µm | collagen integrated through the wall at one position — a line density, independent of local thickness |
| `cum_collagen_od_um2` | OD·µm² | that quantity accumulated along the vessel |
| `collagen_od_um2_per_mm_length` | OD·µm²/mm | the headline per-length figure, comparable between animals |
| `collagen_area_fraction_media` | fraction | media area where collagen outweighs muscle — no threshold |
| `collagen_fraction_of_stain` | fraction | collagen ÷ (collagen + muscle) in the media; no threshold, does not saturate |
| `media_thickness_um` | µm | per position along the wall |
| `media_area_per_um` | µm²/µm | true cross-sectional area per unit length, Jacobian-corrected |
| `collagen_adventitia_od_um` | OD·µm | adventitial collagen, kept separate so it cannot leak in |
| `stretch_p1` / `stretch_p99` | — | how much unrolling had to squash or spread the wall (1 = none) |
| `folded_fraction` | fraction | samples where the sampling lines crossed and were discarded |
| `non_wall_length_fraction` | fraction | traced length the totals leave out, for any of the reasons below |
| `branch_length_fraction` / `thickened_length_fraction` | fraction | a fork, measured over the span the branch actually touches this wall; or thickening away from any fork |
| `off_wall_length_fraction` | fraction | a few microns of media where the vessel has tens: a gap in the band |
| `collagen_dominated_length_fraction` | fraction | the band is there but has no muscle in it — adventitia, or the collagen wedge at a branch mouth |
| `collagen_od_um2_per_mm_all_traced` | OD·µm²/mm | the per-length figure with nothing excluded |
| `n_branches` / `n_lumina` | count | forks found on this wall; vessels in the frame |

### Normalise collagen to muscle

`collagen_od_um2_per_mm_length` is **not** a fibrosis index. It integrates a concentration
through the wall, so a thicker wall gives more of everything — arithmetic, not biology.
Dividing collagen by muscle cancels it, because both are integrated through the same
thickness.

Pearson r against media thickness, from the batch over all four sections:

| section | collagen ÷ muscle | collagen per unit length |
|---|---|---|
| thoracic sec1 | −0.78 | **+0.97** |
| thoracic sec3 | −0.71 | **+0.97** |
| arch sec4 | −0.64 | **+0.96** |
| arch sec2 | −0.42 | **+0.73** |

All four sections show the per-length figure tracking thickness at r ≥ 0.73; on arch sec2 it
is weakest because that mosaic is incomplete and cut into five short segments. The ratio
removes it in every case, and the **sign flips**, which is the tell — a units artefact would
have stayed positive.

What is left is a real negative relationship of varying strength: where the media is
thicker it is proportionally less collagenous, clearly so in the thoracic sections (−0.71,
−0.78) and in arch sec4 (−0.64), weakly in the incompletely assembled arch sec2 (−0.42). It
is not bleed from the adventitial collagen band — restricting the measurement to the middle
20–80 % of the wall leaves it unchanged or slightly stronger.

The ratio is undefined, not large, where the muscle integral is essentially zero: those
positions have no media to divide through. Anywhere muscle falls below 5 % of the section's
own median the ratio is NaN, which keeps a handful of junction positions from dominating
both the panel and the correlation.

Quote `collagen_to_muscle_ratio`, or the bounded `collagen_fraction_of_stain`, which is a
monotone transform of it. Section ratios: thoracic 0.493 and 0.464, arch 0.346 and 0.409 —
so by composition the arch media is the *less* collagenous of the two, the opposite of what
the per-length figure says. `r_thickness_vs_collagen_to_muscle` and
`r_thickness_vs_collagen_per_length` are in `totals` so the confound is visible without
opening the figure.

The bottom panel of the figure plots this normalised: `collagen_to_muscle` position by
position along the wall, with the *running* value — collagen accumulated so far over muscle
accumulated so far — over it on the same axis, settling on the whole-wall ratio. Both are
the same quantity, so a drift of the running curve away from the dashed whole-wall line is
a real gradient along the vessel and not a thick patch.

### How reproducible the trace is

The whole pipeline is deterministic: assembling and measuring the same folder twice in one
environment gives bit-identical mosaics and totals.

It used not to be stable across environments. The rendered mosaic differs by about one part
in 10⁶ between numpy 2.5.2 / scikit-image 0.26 and numpy 1.26.4 / scikit-image 0.23 —
resampling and morphology are not bit-identical between versions — and on **aortic arch
sec4** that 10⁻⁶ difference used to move mean media thickness by 8.3 % and traced length by
5.3 %. The cause was the reference curve: it was the lumen outline pushed out by one
constant, which left it balanced between routes that a vanishingly small change could tip.

Anchoring the curve to the medial axis of the muscle, point by point, removes that. The
same mosaic now gives identical totals in both environments, and perturbing the image
directly — which is a sharper test than swapping libraries — barely moves the answer:

| perturbation of every pixel | traced length | media thickness | collagen ÷ muscle |
|---|---|---|---|
| ±10⁻⁶ (the size of the library difference) | 0.000 % | +0.002 % | +0.003 % |
| ±10⁻⁵ | 0.000 % | +0.019 % | +0.14 % |

Thoracic rings, being closed and unbranched, never had the problem. Mosaic *rendering* may
still differ slightly between environments; it now costs three parts in a hundred thousand
rather than eight per cent.

### Missing fields

A section that is missing a field or two is still measurable, because every headline number
is per unit length. Removing consecutive fields from a complete 38-field thoracic section
and re-measuring:

| fields kept | traced | media thickness | collagen/mm | collagen fraction |
|---|---|---|---|---|
| 38 | 2.765 mm | — | — | — |
| 35 | 2.765 mm | +0.0 % | −0.0 % | +0.0 % |
| 33 | 2.702 mm | +0.2 % | +0.5 % | +0.4 % |
| 30 | 2.515 mm | +2.5 % | +3.7 % | +0.6 % |
| 27 | 2.410 mm | +4.2 % | +6.2 % | +0.3 % |
| 24 | 1.964 mm | +5.0 % | +5.3 % | −0.3 % |

Losing wall costs length and biases the density by a few per cent, because the missing
stretch is not representative of the rest. `collagen_fraction_of_stain` barely moves at all
— within 0.6 % even with a third of the section gone — which makes it the readout to quote
when a mosaic is incomplete.

The coverage mask is used to keep the difference between "no tissue here" and "no image
here" straight: stretches with no image behind them are marked `no_data`, excluded rather
than measured as thin wall, and reported as `no_data_length_fraction` and
`image_coverage_fraction`. Pass the mask from
`render_mosaic(..., return_coverage=True)`; it is inferred from the image otherwise.

There is deliberately **no gap-bridging**. Two versions were built and both removed:
dilating the band into uncovered territory swallowed the empty corners of the mosaic and
reported a 37 mm vessel with five lumina, and pairing up the band's free ends never fired on
a real gap — with eight fields removed the media mask has 36 free ends, of which one is
anywhere near the missing data, because most are mask spurs rather than the two sides of a
hole. Placing the missing fields in the app is the fix that works.

### Reading it back to the vessel

`plot_rectangular(res, overview_rgb=...)` is one column on one horizontal scale: the section
with the traced wall on top — a quarter-point white dash, dotted at whole millimetres, with
the start and end of the strip marked — then the unrolled wall, the two stain channels, the
local collagen density, and the cumulative curve. A step in the cumulative curve lines up with
the patch of wall that caused it, and the numbered marks say where on the vessel that is.

### Cutting the wall at branches

Which side of a sampling line is the lumen is decided **per position**, not once for the
vessel. It has to be: where the traced path runs through a branch it comes out on the other
wall, and from there the lumen is on the opposite side. Deciding globally leaves everything
past the junction inside out, with adventitia drawn where the lumen belongs — which is
exactly what the arch figure showed. The per-position votes are median-filtered over a
window much wider than a junction (and wrapped on a closed vessel, or the seam where the ring
is cut open votes against its own neighbours).

The wall is then **cut** into segments at branches, at no-data stretches, at gaps in the
band, and wherever the path turns over. Arch sec2 comes out as five segments, of which two
carry the vessel — 0.31 and 2.53 mm — with three slivers between the junction flags; the two
thoracic sections are one segment each. The cuts are drawn as hard vertical lines on every
panel that runs along the wall, because the two sides of a cut are not neighbours on the
vessel and nothing should be read across the join. `profile["segment"]` carries the labels
and `totals["n_segments"]` the count.

### Branches

A branch is detected as a *topological* fact, not a thickness: rasterise the strip of wall the
straightening actually measured, subtract it from the media, and see whether a substantial
piece of wall is left attached. At a fork the branch's own wall is media the strip never
covered; a merely thick wall lies entirely inside it and leaves nothing. That distinction
matters, because a genuinely thickened wall is the thing you would be looking for in a
diseased vessel and must not be written off as a junction.

(The textbook answer — skeletonise the media and look for degree-three nodes — does not work
here. A wall band is tens of microns thick, so its skeleton is a mesh of small loops: on a
plain thoracic ring with no branch at all it reported 182 forks.)

How far a junction reaches is measured, not assumed. A branch leaves at a shallow angle and
runs alongside its parent for a few hundred microns, so the flagged span is the stretch over
which the branch's own wall touches this one, and not a fixed window around a single point.
A window would either fall short of that or overshoot it, and which of the two it does is a
fact about the specimen.

Flagged stretches are shaded on the figure — orange for a junction, red for wall that is
thick without forking, grey for a gap in the band, blue for a band with no muscle in it.
**By default all of these are left out of the totals** (`exclude_non_wall=True`), and the
cumulative curves follow the same decision, so the curve ends at the reported total instead
of climbing through a stretch the total omits. On arch sec2 this is not a small effect: the
junction and the collagen wedge at its mouth span 20 % of the traced wall and were lifting
collagen-per-mm from 26,600 to 39,800. With them out, the arch's cumulative curve is nearly
a straight line — the apparent focal lesion was the branch.

`totals` reports it either way: `collagen_od_um2_per_mm_length` follows the setting,
`collagen_od_um2_per_mm_all_traced` never excludes anything, and `n_branches`,
`branch_length_fraction`, `thickened_length_fraction`, `off_wall_length_fraction` and
`collagen_dominated_length_fraction` say what was found.

### Redirecting the outline by hand

No rule gets every section right, and the ones it gets wrong are usually not
defects so much as judgements: which of two touching vessels to follow, whether
to go up a branch or past it, where a torn wall ought to be joined. Those are
questions about the specimen, and the person looking at it can answer them in a
second.

The **Vessel wall** tab draws the outline on the section and lets you drag along
it to redraw a stretch. A stroke replaces only the span between the two points
it starts and ends nearest — the trace is usually right nearly everywhere — and
on a closed ring the shorter way round is the one that gets replaced, because
nobody redraws the long way to fix a wobble. Undo and Reset are there; **Re-measure
with this outline** runs the same analysis with the same settings, differing only
in where the curve started.

The drawing does not have to be accurate. Everything downstream is unchanged, so
the re-centring pass pulls a rough line onto the middle of the media exactly as
it does the automatic one. Deliberately mangling a good outline — 90 points,
±15 µm of jitter, and one stretch dragged 40 µm off the wall, so that only 81 %
of it still lay on the muscle — and re-measuring from it gives back a curve
100 % on the muscle and totals within 0.3 % of the automatic trace.

**A second vessel can be drawn as well as redirected.** *Add another vessel*
keeps the outline you have and clears the canvas for the next one; on an empty
canvas the first stroke *is* the outline rather than an edit to one, so a
vessel the detector missed entirely can be drawn from scratch. Kept outlines
stay on screen in white, the live one in the accent colour. Re-measuring then
returns one result per outline — its own `lumen_rank`, its own row in the
vessel picker, its own `..._vessel2_*` files — so a hand-drawn vessel is a
vessel everywhere downstream rather than a special case. Drawing back the two
outlines the detector finds on arch sec4 reproduces its own numbers to 0.1 %
(3.733 and 3.307 mm against 3.738 and 3.309). Neither outline has to be
accurate: re-centring pulls each onto the middle of its own media exactly as
it does an automatic trace, and each carries its own ring/arc flag.

In the library it is `analyze_wall(..., guide_um=...)`, an (x, y) array in microns
from the top-left of the image. Microns rather than pixels so a drawing survives
a change of render scale or analysis resolution. `totals["guided"]` records that
a run came from one. Over the API, `guides` takes a list of
`{um, closed}` and measures one wall each; a lone `guide` still behaves as it
always did.

### More than one vessel

**Every vessel in the frame is measured**, not just the biggest. An arch section routinely
catches the vessel twice — the ascending and descending limbs of the same arch, joined
through the section by a stalk of tissue — and measuring one and dropping the other throws
away half the wall. `analyze_vessels(rgb, px_um, params)` returns one result per enclosed
lumen, largest first, with `totals["lumen_rank"]` naming each; the batch writes one row per
vessel and the cohort step pools them back together by segment, which is the right unit
anyway. Arch sec4 goes from 3.74 mm of measured wall to 3.74 + 3.31 = 7.05 mm, and the two
limbs agree closely on the readout that matters — collagen ÷ muscle 0.399 and 0.434.

The **Vessel wall** tab shows a button per vessel above the figure, labelled with its own
wall length, so switching between them is explicit rather than something to notice by
counting lumina against what the cropped map happens to show.

A hole counts as a vessel on two tests, and both are needed. **Size**, because a lumen
smaller than a cell cluster is not a vessel anyone sectioned on purpose. And **the wall
around it**, measured the same way the trace is placed: every hole here is enclosed by media
by construction — that is what makes it a hole — so being surrounded says nothing, while
being surrounded by 50 µm of muscle rather than a two-pixel sliver says a great deal.
Rejected holes are reported in `n_lumina_rejected` and in `res.lumina` with the numbers that
decided it, so a vessel that should have been measured can be seen not to have been.

A vessel must *enclose* its lumen to be found. An incompletely assembled section has an open
wall and no enclosed lumen — arch sec2 is one — and a branch cut near its origin usually
shares the parent's lumen rather than enclosing its own; those appear as attachment points
on the parent instead.

### Two views of the straightened wall

**Real microns** keeps the true thickness, so wall thickening is visible. **Thickness-normalised**
divides each sampling line by its own wall thickness, so 0 is the luminal surface and 1 the
media/adventitia junction *everywhere* and the media is a rectangle. Use the second to compare
where in the wall something sits — one row means the same fraction of the media at every
position along the vessel, which a row of the micron view does not. `plot_rectangular` draws
it alongside the two stain channels.

Laying a curved band flat cannot preserve area: at depth `d` the sampling lines are spread by
`1 - κd`. Every integral through the wall is weighted by that factor, so the convex side of a
bend is not counted more than it should be, and samples where it reaches zero — the lines have
crossed and would read the same tissue twice — are discarded and counted in `folded_fraction`.

The cumulative curve is the reason to prefer this over a single number: a straight line means
diffuse fibrosis, a staircase means focal lesions, and the total alone cannot tell them
apart.

**One lumen per result, but every lumen gets a result** — see "More than one vessel" above.
`wall_length_mm` names which vessel a given result is: compare it against the vessel picker's
own figure to confirm the two line up. A branch cut near its own origin usually shares the
parent's lumen rather than enclosing one of its own, so it cannot be measured as a separate
vessel; restrict the input to an island containing only that branch if it needs its own trace.

**Why the area fraction has no threshold in it.** Collagen inside the media is unimodal, so an
automatically chosen threshold has nothing to lock onto. Re-centring the sampling curve on one
thoracic section moved the collagen percentiles by less than 0.02 — and moved Otsu from 0.778
to 0.632, and with it the reported fibrosis from 9 % to 34 %. A per-section threshold is also
wrong in principle: a more fibrotic section earns a higher threshold, which cancels part of the
difference being measured. Asking instead whether each pixel holds more collagen than muscle
has no free parameter, is invariant to overall staining intensity, and survived the same
geometry change at 6.6 % → 4.9 %. Pass `blue_threshold=<number>` for the classical absolute
index if you need it; the notebook plots it against threshold so you can see the slope.

### A figure you can rotate and resize

The full analysis figure itself — the same `plot_rectangular` panel with the
map, both unrolled bands, both depth profiles and the collagen regression — is
editable in place, with the **Figure style** controls above it. **Rotate
stitched image** spins the section to whatever orientation reads best.
**Row height** is one number for all seven panels below the map, rather than
a size to set on each; **Left width**/**Right width** size the two columns
independently — the three unrolled images plus the local/running plot share
the left one, the three depth profiles and the regression share the right —
since the two hold different kinds of content and rarely want to grow
together. The map spans both, so it's sized by their sum and needs no control
of its own; its height still follows its own aspect ratio, unrelated to the
rows below it. **Font size (pt)** sets the panel titles directly and scales
labels and legend text with
them, on the same reasoning `FS_TITLE` and its neighbours already follow —
panels that disagree about what a title looks like read as separate figures
pasted together — while tick labels stay one point smaller than whatever the
titles end up at, rather than their own independent size to keep in step;
**gridlines** toggles the grid on the panels that have one (an unrolled image
never gets a grid drawn over it, on or off — a grid line through a picture is
illegible either way, and the toggle doesn't override that); **plot boxes**
switches between a full frame and an open one — bottom and left kept, top and
right dropped — never no frame at all, since a panel with no axis lines reads
as unfinished rather than as clean. The figure carries no title of its own
any more; what section and vessel it's from is the filename it's saved under,
not text baked into the image.

The map (panel A) marks only the trace itself, the mm ticks and where it
starts and stops — not branches and not other lumina in the same section any
more. Both are still measured and still in the totals and the CSVs; they just
aren't annotated onto this picture. Use the vessel buttons above the figure
to move between lumina instead of reading their positions off the map.

Every panel is lettered A through H, and the flagged stretches shaded into
the four along-the-wall panels — a branch, a stretch with no image behind it,
one thickened without forking — get one shared legend at the bottom of panel
H rather than a colour code with no key: built from whichever flags actually
appear in this trace, so it says what's shaded here rather than what could be
in general.

Every numerical panel has a CSV behind it, all listed next to the totals
table. C and E (the depth profiles) are `..._depth_absolute_um.csv` and
`..._depth_relative.csv`; G (the regression) is `..._along_wall.csv` filtered
to `counted`; H (local/running collagen ÷ muscle) is the same file's
`collagen_to_muscle` and `collagen_to_muscle_running` columns, the second one
added there for exactly this — it used to exist only as a line in the figure,
computed from two columns that were already saved but never combined into the
number the panel actually plots. The three image panels (B, D, F) are arrays,
not tables, and are saved as arrays in `..._maps.npz` instead.

A batch additionally pools **panel E** into one long table,
`batch_depth_relative.csv`: every section's profile through the media, one row
per depth per vessel, carrying the same `genotype` / `mouse` / `segment` /
`section` columns as `batch_summary.csv` so it groups the same way. The
per-section files are still written beside each mosaic and the pooled values
are identical to them; what the pooled file adds is that the question panel E
exists to answer — does the profile through the media differ by genotype — is
asked across sections, and answering it from fifty separate files is not an
analysis. Filter to `in_media` for the wall itself; the grid runs a little
past both edges so the lumen and adventitia stay visible.

`cohort_depth_relative.csv` is the same thing averaged, one curve per genotype
per segment. It climbs the ladder the rest of the cohort step uses: a mouse's
sections are averaged first, at every depth, and only then are mice averaged
into groups, so **n is animals**. Averaging the sections directly would let a
mouse that happened to yield six of them count six times over — the mistake
`per_genotype` exists to avoid, and harder to spot here because the output is
a curve rather than a number. `muscle_sd` and `collagen_sd` in this file are
therefore the spread *between animals*, not the variation along one wall that
the per-section file carries under the same name, and `n_mice` says how many
went in.

There is no second copy of this figure to keep in sync. A style change posts
to `/api/replot`, which looks up the `WallResult` behind whichever
`..._rectangular_channels.png` is on screen — kept server-side from the run
that produced it, keyed by that PNG's own path — and calls `plot_rectangular`
again with the new rotation, size, font scale, grid and spines, overwriting
the same PNG and SVG a plain analysis run would have saved. Every restyle
starts over from that original cached result rather than the previously
rotated output, so turning the dial back and forth never compounds — 30° then
another 30° lands exactly where a single 60° would, not a degree short from
interpolating a raster twice. Rotation itself works by turning the source
image and carrying the traced centreline, the branch marks and the other
lumina through the same rotation, rather than rotating a finished picture with
the annotations already burned into it — so they still land exactly on the
anatomy at any angle, including on the automatic quarter-turn a portrait
section already gets before your own rotation is added on top of it.

Because restyling redraws from the cached result rather than the display,
switching vessels or re-running the analysis keeps whatever style you've set:
the server-side figure a fresh run just saved is redrawn once more, in the
background, the moment its picture appears on screen. If the process
restarts, that cache is gone and the controls will say so — rerun the
analysis and the next restyle has something to redraw again.

---

## From sections to genotypes

Every organ tab carries a **Batch** block at the bottom of it. Point it at a *tray* — one
parent folder, one sub-folder per section — and
press **…** to scan it. The listing shows every readable sub-folder, not only the ones
holding images, because most of the path to a tray runs through folders that hold none.
Clicking a folder's **name** opens it; clicking the **○** beside a section ticks it into
the run, and ticks survive walking to another folder, so a run can be built from several
trays. **Choose all here** takes the whole level. A folder the process cannot open — a
`.Trash`, a cloud-drive mount — is named and stepped over rather than failing the scan.

Every folder name is read for mouse, genotype, age, segment, slide and section:

```
4_1749_L8D-E-KO, 32W_aortic arch_slide 9_sec2
^ ^    ^                ^          ^       ^
| mouse genotype, age   segment    slide   section
order
```

Nothing insists on that order. Each field is recognised by what it looks like — `sec\d+`,
`slide \w+`, a bare 3+ digit number for the animal, a comma-joined `genotype, age`, and a
vocabulary of regions for whichever organ's tab you are in: aortic segments (root,
ascending, arch, middle, descending thoracic, thoracic, abdominal, supra/infrarenal, plus
carotid, femoral and mesenteric), heart chambers, or kidney regions. A field that cannot be
read costs you only that field: the section is still measured and lands in an `unknown`
group where you can see it. The scan reports how many folders are missing something before
you run anything. A **`metadata.csv`** beside the folders, with a `folder` column and any of
`mouse, genotype, age, segment, slide, section`, overrides the parser — so a mis-read is
fixed without renaming data on disk.

You do not have to write that file by hand. Tick the sections in the tray browser, pick the
field in the box under **Choose all here** — segment, mouse, genotype, age, slide or section —
type the value and press **Set**: it is written into the `metadata.csv` beside them, and the
list redraws from the file so you can see what was written. The suggestions follow the field:
the segment vocabulary above for a segment, and for anything else whatever this tray already
says, so a typed value lands on a spelling that groups rather than starting a second group of
its own. Anything you type is kept, though — which is how a dataset labels a segment this app
has never heard of, without anyone editing its code. A blank value clears that field and the
folder name has its say back. Sections ticked across several trays are written to each tray's
own file.

The batch then produces four tables, written as `cohort_*.csv` next to `batch_summary.csv`:

| table | one row per | how it is built |
|---|---|---|
| `sections` | section | the measurement itself |
| `by_mouse_segment` | mouse × segment | slices of that segment pooled |
| `by_mouse` | mouse | every segment pooled — the whole-aorta number |
| `by_genotype` / `comparison` | genotype × segment × readout | mean ± SD across mice, Welch's t |

**Pooling adds integrals; it does not average ratios.** Two sections of one segment in one
mouse are one wall sampled twice, so the pooled collagen ÷ muscle is *total* collagen over
*total* muscle. Averaging the two section ratios would weight a 0.4 mm off-cut the same as a
4 mm ring. On the arch of mouse 1749 the difference is 0.378 pooled against 0.380 averaged —
small here because the two sections are of similar length, and unbounded when they are not.
`mean_media_thickness_um` comes back as summed media area over summed wall length, which is
exactly the length-weighted mean thickness. Only extensive quantities are summed —
`wall_length_um`, `media_area_um2`, `collagen_od_um2_total`, `muscle_od_um2_total`,
`collagen_area_um2_total` — and every intensive readout is rebuilt from those sums.

**Plots.** The **Plot** controls draw the tables: one panel per readout, bars for the group
mean, and **a dot for every animal**. The dots are the point of it — with three or four mice
a side, a bar and an error bar cannot tell you whether a difference is four animals agreeing
or one animal carrying the group. A dot can be a mouse (per segment, or pooled across
segments) or a section; choose *section* and the figure says in red that sections within an
animal are repeated measures, because a bar drawn over them is not an n. `p` is printed as a
number, not as stars: at these group sizes 0.04 and 0.06 are the same result. Error bars are
SD, SEM or none; width and font size are set in millimetres and points. The figure is
written as `cohort_plot.png` and `cohort_plot.svg`, and **Download SVG** hands you the
vector with its text still text. Plots are drawn from the `cohort_*.csv` already on disk, so
turning a control costs nothing — the measurement is the expensive part and it is done.

Under **Axes** you can pin the **plotting area** itself rather than the whole picture — 2 ×
2 in is a sensible panel — and the canvas becomes whatever the labels need around it. That
is the setting that makes two panels comparable: `width_mm` sizes the picture, so a longer
tick label eats into the plot instead of the margin.

**Through the wall** is drawn beside the readout grid, from `cohort_depth_relative.csv`,
and has two layouts. By default a column is a stain and the genotypes are overlaid in it,
for comparing groups. Tick **both stains in one panel** and it turns the other way up: one
panel per condition with the counterstain and the collagen in it, coloured as the section
figure colours them, for reading where the collagen sits relative to the muscle within one
wall. Only conditions with data get a panel, so the panels wrap rather than leaving the
cross product's holes. **source data (CSV)** under the figure is the table it was drawn
from: one row per depth per condition, mean, SD between animals, and n.

**The mouse is the unit of analysis.** Sections and slides are repeated measures of one
animal; counting them as independent inflates n and shrinks p for reasons that have nothing
to do with biology. Genotypes are compared across per-mouse values, with `n_mice` and
`n_sections` printed side by side so the difference is impossible to miss — twenty sections
from three animals is an n of 3. The test is Welch's, which does not assume equal variances;
with three or four animals a side you cannot check that assumption, so not making it is
free. `p` is blank below two mice per group, which is the honest answer: a difference
between two single animals has a direction and no p-value.

## Files

| file | |
|---|---|
| `stitch_core.py` | the assembler: loading, masked-NCC matching, verified layout, mosaic rendering, layout I/O |
| `wall_analysis.py` | stain deconvolution, media segmentation, straightening, the wall measurements |
| `fibrosis.py` | fibrosis by area, for tissue with no wall to follow — heart, kidney |
| `cohort.py` | folder names to metadata, pooling by region and mouse, genotype comparison |
| `app.py` | Flask server |
| `sessionfile.py` | saved sessions: the whole working state, with the pictures referenced rather than copied |
| `zones.py` | the hand-drawn regions an analysis can be restricted to |
| `open_when_ready.py` | opens the browser once the server answers, for the launchers |
| `static/` | the browser front end |
| `build_notebook.py` | writes `Aorta_wall_analysis.ipynb`: the same pipeline as a notebook, plus a batch over all sections. The generator is the source — edit it, not the `.ipynb`, which is not in the repo |

### Library use

```python
import stitch_core as SC, wall_analysis as WA

ts, poses, matches, stats = SC.auto_stitch("/path/to/fields")
qc = SC.layout_qc(ts, poses)                 # check before trusting
main = [i for i, p in enumerate(poses) if p.island == 0]
mosaic = SC.render_mosaic(ts, poses, scale=0.35, indices=main)

res = WA.analyze_wall(mosaic, ts.px_um / 0.35, WA.WallParams())
print(res.totals["collagen_od_um2_per_mm_length"])
WA.save_results(res, "out", "section01", overview_rgb=overview)
```

`save_results` writes the tables, the two straightened images, the maps as `.npz`, and the
one figure as both `_rectangular_channels.png` and `_rectangular_channels.svg`. The SVG is
its own render, not a second save of the PNG's figure: every label is real text rather than
matplotlib's default of a glyph traced out as a curve, so a panel can be restyled for a paper
without going back through the analysis, and the map's traced wall is one selectable path
in both, since a single stroke colour needs no per-segment fragments. The numbered marks
along it say where on the wall each point is. In the app the same
file is one click behind **Download SVG** above the figure. Pass `overview_rgb` to get the
map panel — the section with the traced wall
drawn on it, turned a quarter turn if it is taller than it is wide, so the page is used
across rather than down.

Nothing drawn on the section carries meaning in its colour. The trace is white, the
millimetre marks and the start/end markers are white with a black edge, and start and end are
told apart by shape — a circle and a square — rather than by hue. A trichrome section is
already red and blue; a green dot or a red square on top of it reads as tissue, and the whole
point of the map panel is to be looked *through* while fibrosis is scored by eye. **Red
intensity** in the figure style controls turns the red stain down for the same reason, and
touches only the picture: every number comes from the concentrations before any gain is
applied. **Wall trace** hides the line altogether, and the panel then says "Section" rather
than "Traced wall" so a saved figure never claims something it is not showing.

`SC.snap_selection(ts, poses, indices)` aligns a hand-picked set against everything outside
it, in passes, so a settled member becomes an anchor for the rest without any member ever
voting against a sibling that is still displaced.

`SC.save_layout` / `SC.load_layout` round-trip a layout as JSON plus a CSV of tile
positions, so a session finished by hand in the app can be re-analysed later without
repeating the work.

## The generated notebook

`python build_notebook.py` writes a notebook that runs the same pipeline as the app, plus a
batch over all your sections. It is generated rather than kept in the repo, so run it once
if you want one.

Where your sections live is the one path that is about your machine rather than about the
analysis, so it is never hardcoded. Give it to the generator and it becomes the notebook's
default; `--out` names the file:

```bash
python build_notebook.py --data /path/to/sections --out my_analysis.ipynb
```

Either may be left off: `--data` falls back to `$AORTA_DATA` and then to `~/aorta-sections`,
`--out` to `Aorta_wall_analysis.ipynb` next to the app. A relative `--out` is taken from
where you are rather than from the app folder, `.ipynb` is added if you leave it off, a
folder gets the default name inside it, and missing parent folders are created.

The generated notebook still reads `AORTA_DATA` at run time and falls back to that baked-in
default, so the same file also works for someone whose sections are somewhere else.

A run overwrites whatever is already at that path — and says so before it does — which is
the other use for `--out`: building a second notebook without losing the first. `.gitignore`
covers `*.ipynb` rather than the one default name, since every notebook here is a generated
artefact carrying results and `--out` can call them anything; `git add -f` if you ever do
want to track one.
