"""
The fragmentation measure has to tell fragments from a solid layer.

``muscle_beyond_edge`` exists because neither media-edge rule can describe an
outer media that has broken up.  It reports how far muscle reaches past the
edge and how much of that reach is muscle, and the second number is the whole
point: a wall whose edge merely stopped short of solid muscle and a wall whose
outer media is in pieces both reach a long way out, and only ``fill`` tells
them apart.  An earlier version walked out to the first gap, which made
``fill`` ~1 by construction and could not distinguish them at all.

``radial_variation`` measures the other half: not what lies past the edge but
how evenly the muscle is laid down inside it.  Its whole design is the scale
split, because an aortic media is meant to be layered and a raw variance
across the wall mostly reports how crisply a section resolves its elastic
lamellae.  The test below builds the two walls that a raw variance confuses --
a strongly lamellated intact one and a uniform one with a hole in it -- with
their raw figures deliberately set equal.

Run it directly:  python test_wall_measures.py
"""
from __future__ import annotations

import numpy as np

import wall_analysis as WA


def _band(pattern, depth_step=0.5, reach_um=25.0, n_pos=40):
    """A band whose media ends at depth 0, with *pattern* laid out beyond it."""
    dep = np.arange(-40.0, 60.0 + depth_step, depth_step, dtype=np.float32)
    mu = np.zeros((n_pos, len(dep)), np.float32)
    mu[:, (dep >= -30) & (dep < 0)] = 1.0            # the compact media
    for lo, hi in pattern:
        mu[:, (dep >= lo) & (dep < hi)] = 1.0
    return mu, dep, np.zeros(n_pos, np.float32)


def test_solid_and_fragmented_are_told_apart() -> None:
    p = WA.WallParams()
    win = float(p.fragment_reach_um)

    # Nothing past the edge.  Not exactly zero: the 2 um depth smooth is the
    # muscle-end rule's, so the compact media bleeds one sample across the
    # boundary.  A micron is far below the 12 um a fragment has to reach.
    reach, fill = WA.muscle_beyond_edge(*_band([]), p)
    assert np.all(reach <= 2.0), reach[:3]

    # Solid muscle filling the window: reaches the end, and it is all muscle.
    reach, fill = WA.muscle_beyond_edge(*_band([(0.0, win)]), p)
    assert np.allclose(reach, win, atol=1.0), reach[:3]
    # Above the 0.8 that `fragmented_length_fraction` gates on, so this wall
    # is not counted as fragmented however far its muscle reaches.
    assert np.all(fill > 0.8), fill[:3]

    # Fragments spread over the same window: reaches just as far, but the gaps
    # between them are counted.  This is the case the first version missed.
    frag = [(0.0, 3.0), (8.0, 11.0), (16.0, 19.0), (win - 2.0, win)]
    reach, fill = WA.muscle_beyond_edge(*_band(frag), p)
    assert np.allclose(reach, win, atol=1.0), reach[:3]
    # Below the gate, so this one is.  Same reach, opposite verdict -- which
    # is the only reason the measure reports two numbers instead of one.
    assert np.all(fill < 0.8), fill[:3]

    # A gap wider than the muscle-end hold does not stop the walk -- the
    # fragments past it are exactly what is being measured.
    reach, _ = WA.muscle_beyond_edge(*_band([(0.0, 2.0), (18.0, win)]), p)
    assert np.allclose(reach, win, atol=1.0), reach[:3]


def test_level_follows_the_section_not_an_absolute() -> None:
    """Staining intensity varies between slides; the measure may not."""
    p = WA.WallParams()
    mu, dep, outer = _band([(0.0, 12.0)])
    a = WA.muscle_beyond_edge(mu, dep, outer, p)
    b = WA.muscle_beyond_edge(mu * 3.7, dep, outer, p)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1], equal_nan=True)



def _wall(profile, depth_step=0.5):
    """A band *profile* thick, laid out from the lumen at 0 to the outer edge."""
    dep = np.arange(-20.0, 80.0 + depth_step, depth_step, dtype=np.float32)
    mu = np.zeros((12, len(dep)), np.float32)
    j0 = int(np.searchsorted(dep, 0.0))
    mu[:, j0:j0 + len(profile)] = profile
    inner = np.zeros(12, np.float32)
    outer = np.full(12, len(profile) * depth_step, np.float32)
    return mu, dep, inner, outer


def _raw_cv(profile):
    return float(np.std(profile) / np.mean(profile))


def test_lamellae_are_not_mistaken_for_fragmentation() -> None:
    p = WA.WallParams()
    n, step = 120, 0.5                              # a 60 um wall at 0.5 um
    d = np.arange(n) * step

    # An intact wall with strong 4.5 um banding, and a uniform wall with a
    # 10.8 um dropout in it.  Set so a raw variance across the wall cannot
    # tell them apart -- which is the failure this split exists to fix.
    lamellar = 1.0 + 0.5 * np.cos(2 * np.pi * d / 4.5)
    dropout = np.ones(n)
    dropout[int(0.41 * n):int(0.59 * n)] = 0.2
    assert abs(_raw_cv(lamellar) - _raw_cv(dropout)) < 0.02, (
        _raw_cv(lamellar), _raw_cv(dropout))

    cl, ll = WA.radial_variation(*_wall(lamellar), p)
    cd, ld = WA.radial_variation(*_wall(dropout), p)
    cl, ll, cd, ld = (float(np.nanmedian(x)) for x in (cl, ll, cd, ld))

    # The banding smooths away; the hole does not.
    assert cd > 4 * cl, f"dropout {cd:.3f} vs lamellar {cl:.3f}"
    assert cl < 0.05, cl
    # And the split is not one-way: the banding is reported as what it is, so
    # the two columns rank these walls opposite ways round.  Direction is the
    # claim, not a margin -- a real hole has square edges, which carry
    # fine-scale energy of their own, so how close the two come is a fact
    # about the test's own shapes and nothing to tune against.
    assert ll > ld, f"lamellar contrast {ll:.3f} vs dropout {ld:.3f}"


def test_variation_ignores_staining_level() -> None:
    """A darker slide is not a more broken one: CV is taken over its own mean."""
    p = WA.WallParams()
    d = np.arange(120) * 0.5
    prof = 1.0 + 0.35 * np.sin(2 * np.pi * d / 30.0)      # coarse, survives
    a, _ = WA.radial_variation(*_wall(prof), p)
    b, _ = WA.radial_variation(*_wall(prof * 4.3), p)
    assert np.allclose(np.nanmedian(a), np.nanmedian(b), atol=1e-5)
    assert np.nanmedian(a) > 0.1, np.nanmedian(a)         # and it sees it


def test_spacing_recovers_a_known_lamellar_period() -> None:
    p = WA.WallParams()
    for period in (4.5, 6.0, 8.0):
        d = np.arange(140) * 0.5
        prof = 1.0 + 0.4 * np.cos(2 * np.pi * d / period)
        sp, _, _ = WA.lamellar_geometry(*_wall(prof), p)
        got = float(np.nanmedian(sp))
        # The profile is sampled at 0.5 um, so the peak positions quantise.
        assert abs(got - period) <= 0.5, f"period {period} read as {got}"


def test_noise_alone_is_not_layering() -> None:
    """The prominence gate is absolute, so a flat wall yields no lamellae.

    This is the regression that matters: a threshold set against the
    residual's own spread always finds the largest wiggles in whatever it is
    given, so a blurred, featureless media reads as beautifully layered.
    """
    p = WA.WallParams()
    rng = np.random.default_rng(4)
    flat = 1.0 + 0.01 * rng.standard_normal(140)          # 1% ripple: noise
    sp, _, _ = WA.lamellar_geometry(*_wall(flat), p)
    assert not np.isfinite(sp).any(), f"found lamellae in noise: {np.nanmedian(sp)}"

    # The same noise passes a spread-relative threshold, which is the point.
    from scipy import ndimage as ndi, signal
    r = flat - ndi.uniform_filter1d(flat, 24)
    assert len(signal.find_peaks(r, prominence=0.5 * r.std(), distance=4)[0]) >= 3


def test_edge_width_tracks_blur() -> None:
    """The luminal rise widens with blur and does not care about the lamellae."""
    from scipy import ndimage as ndi
    p = WA.WallParams()
    d = np.arange(140) * 0.5
    prof = 1.0 + 0.4 * np.cos(2 * np.pi * d / 5.0)
    widths = []
    for sigma_um in (0.0, 1.0, 2.0):
        mu, dep, inn, out = _wall(prof)
        if sigma_um:
            mu = ndi.gaussian_filter1d(mu, sigma_um / 0.5, axis=1)
        _, _, ew = WA.lamellar_geometry(mu, dep, inn, out, p)
        widths.append(float(np.nanmedian(ew)))
    assert widths[0] < widths[1] < widths[2], widths
    assert widths[0] <= 1.0, widths          # a step edge is one sample wide


def _peaks_at(offsets, n=140, step=0.5, width=1.2):
    """A profile with a bump at each offset (um from the wall's inner edge)."""
    d = np.arange(n) * step
    v = np.full(n, 1.0)
    for o in offsets:
        v += 0.5 * np.exp(-0.5 * ((d - o) / width) ** 2)
    return v


def test_disorder_separates_even_stacking_from_ragged() -> None:
    """Same number of layers, same median gap -- only the evenness differs."""
    p = WA.WallParams()
    even = _peaks_at([10, 15, 20, 25, 30, 35, 40])          # every 5 um
    ragged = _peaks_at([10, 13, 21, 25, 30, 38, 43])        # 3,8,4,5,8,5
    sp_e, di_e, _ = WA.lamellar_geometry(*_wall(even), p)
    sp_r, di_r, _ = WA.lamellar_geometry(*_wall(ragged), p)
    sp_e, di_e, sp_r, di_r = (float(np.nanmedian(x)) for x in (sp_e, di_e, sp_r, di_r))
    assert np.isfinite(di_e) and np.isfinite(di_r), (di_e, di_r)
    # The gaps have the same middle, so spacing alone cannot tell them apart.
    assert abs(sp_e - sp_r) <= 0.5, (sp_e, sp_r)
    assert di_r > 3 * di_e, f"ragged {di_r:.3f} vs even {di_e:.3f}"


SHAPE = (1250, 1000)


def _arc(cy, cx, r, half, a0=0.0, span=360.0):
    """A band of wall at radius *r*, covering *span* degrees from *a0*."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(float)
    band = np.abs(np.hypot(yy - cy, xx - cx) - r) <= half
    if span < 360.0:
        ang = np.degrees(np.arctan2(yy - cy, xx - cx))
        band &= ((ang - a0) % 360.0) <= span
    return band


def test_two_open_walls_are_two_vessels_and_the_larger_is_first() -> None:
    """An arch caught at a branch: two walls, neither closed.

    With no lumen to enumerate them by, the whole band used to be skeletonised
    as one and the longest path won -- and which wall it came from was settled
    by whichever skeleton endpoint the raster reached first.  The small arc is
    put above the large one here for exactly that reason: it wins that race,
    and a real section came back as one vessel, the wrong one, 2.9 mm of wall
    with 3.7 mm dropped in silence.
    """
    px = 1.0
    small = _arc(200, 500, 150, 20, a0=120, span=300)   # highest in the picture
    big = _arc(800, 500, 300, 22, a0=120, span=320)
    blob = _arc(600, 900, 0, 85)                        # a solid disc: fat, not wall
    mask = small | big | blob

    pieces = WA.wall_pieces(mask, (), px, 20_000.0)
    told = [(round(q["area_um2"]), round(q["length_um"]), q["vessel"]) for q in pieces]
    assert [q["vessel"] for q in pieces] == [True, True, False], told
    assert pieces[2]["area_um2"] > 20_000.0, (
        f"the disc is {pieces[2]['area_um2']:.0f} um2 and was turned down on size: "
        "this picture no longer tests the shape rule it was built to test")
    assert "blob" in pieces[2]["why"], pieces[2]["why"]
    # Length from area and thickness, against the geometry it was built from.
    assert abs(pieces[0]["length_um"] - 2 * np.pi * 300 * 320 / 360) < 120, told
    assert abs(pieces[1]["length_um"] - 2 * np.pi * 150 * 300 / 360) < 120, told

    arcs = [q["mask"] for q in pieces if q["vessel"]]
    traces = [WA.centerline_from_mask(mask, px_um=px, lumina=[], arcs=arcs,
                                      lumen_rank=k)[0] for k in (0, 1)]
    ys = [float(t[:, 1].mean()) for t in traces]
    assert abs(ys[0] - 800) < 60 and abs(ys[1] - 200) < 60, (
        f"vessel 1 sits at y={ys[0]:.0f} and vessel 2 at y={ys[1]:.0f}; "
        "they should be the big arc then the small one")
    lens = [float(np.sum(np.hypot(*np.diff(t, axis=0).T))) for t in traces]
    assert lens[0] > lens[1] > 600, lens

    # Asking past the last one says so rather than quietly repeating a vessel.
    try:
        WA.centerline_from_mask(mask, px_um=px, lumina=[], arcs=arcs, lumen_rank=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a third vessel was traced out of two arcs")

    # The race this exists to settle: told nothing, the whole band is one arc
    # and the small wall wins it.
    whole, _ = WA.centerline_from_mask(mask, px_um=px, lumina=[])
    assert abs(float(whole[:, 1].mean()) - 200) < 60, (
        "tracing the whole band no longer lands on the small arc, so this "
        "picture no longer reproduces the race the pieces were written for")


def test_a_ring_is_one_vessel_not_a_second_piece() -> None:
    """The piece that encloses a lumen is already that vessel, and is not counted twice."""
    px = 1.0
    ring = _arc(300, 300, 200, 25)
    arc = _arc(800, 700, 220, 22, a0=120, span=300)
    mask = ring | arc
    lumina = WA.enclosed_lumina(mask, min_hole_px=400, px_um=px)
    assert len(lumina) == 1, f"{len(lumina)} lumen(s) in a picture with one ring"

    pieces = WA.wall_pieces(mask, lumina, px, 20_000.0)
    assert len(pieces) == 1 and pieces[0]["vessel"], [
        (round(q["area_um2"]), q["vessel"]) for q in pieces]
    assert not (pieces[0]["mask"] & ring).any(), "the ring came back as an arc as well"


def _patched_wall(patch=None, step=0.5, n_pos=60):
    """A 41 um media centred on depth 0, adventitia beyond it, lumen below.

    ``patch`` is a depth range inside the media where collagen outweighs
    muscle -- a focal fibrotic patch, with muscle still plainly present.
    """
    dep = np.arange(-40.0, 60.0 + step, step, dtype=np.float32)
    tissue = np.zeros((n_pos, dep.size), np.float32)
    tissue[:, (dep >= -20) & (dep <= 45)] = 1.0
    muscle = np.zeros_like(tissue)
    muscle[:, (dep >= -20) & (dep <= 20)] = 1.0
    collagen = np.full_like(tissue, 0.15)
    collagen[:, dep < -20] = 0.0
    collagen[:, (dep > 20) & (dep <= 45)] = 1.0          # the adventitia
    if patch:
        inside = (dep >= patch[0]) & (dep < patch[1])
        muscle[:, inside], collagen[:, inside] = 0.60, 1.20
    return muscle, collagen, tissue, dep


def test_a_fibrotic_patch_does_not_shorten_its_own_wall() -> None:
    """Why ``muscle-end`` is the default, in the one case the rules disagree.

    ``collagen-crossing`` ends the media in the same quantity fibrosis is
    measured in, so a focal patch inside the media ends the media *at the
    patch*: the patch is then counted as adventitia and the wall is reported
    thin, and the more collagen a stretch has the less of it is counted.
    ``muscle-end`` asks where the muscle runs out, which collagen cannot move.

    On a clean wall the two agree, which is the other half of the check -- the
    difference below has to come from the patch and not from the rules simply
    disagreeing about where a healthy media ends.
    """
    assert WA.WallParams().media_edge_rule == "muscle-end", \
        "the default rule decides what every thickness in the tool means"

    def edge(rule, patch):
        muscle, collagen, tissue, dep = _patched_wall(patch)
        p = WA.WallParams(arc_step_um=1.0).copy_with(media_edge_rule=rule)
        return float(np.median(WA.find_wall_edges(
            muscle, collagen, tissue, dep, 0.05, p, False)[1]))

    clean = {rule: edge(rule, None) for rule in ("muscle-end", "collagen-crossing")}
    assert abs(clean["muscle-end"] - clean["collagen-crossing"]) < 2.0, \
        f"the rules disagree on a healthy wall too: {clean}"
    assert abs(clean["muscle-end"] - 20.0) < 2.0, f"a 41 um media, edge at {clean}"

    patch = (4.0, 16.0)
    assert abs(edge("muscle-end", patch) - clean["muscle-end"]) < 2.0, \
        "the patch moved the muscle-end edge, which only muscle may do"
    lost = clean["collagen-crossing"] - edge("collagen-crossing", patch)
    assert lost > 10.0, \
        (f"collagen-crossing lost only {lost:.1f} um of media to the patch -- "
         "either the patch is not fibrotic enough to test anything, or the "
         "rule is no longer reading collagen")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
