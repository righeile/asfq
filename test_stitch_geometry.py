"""
Refinement has to land on the offset it is refining.

``refine_pair`` crops both tiles down to the region they are predicted to
share, correlates the crops, and converts the answer back.  Both conversions
involve the crop origins, and getting a sign wrong there is invisible: the
pipeline still produces a mosaic, because every refined pair quietly fails its
confidence test and the coarse answer is kept.  So this checks the one thing
that catches it -- hand ``refine_pair`` a pair whose true offset is already
known and see whether it comes back.

``_wall_track`` is here for the same reason.  It measures the step between two
pieces of one wall across the gap between them, and ``settle_sliver_joins``
slides a whole group of tiles until that step reads zero.  The two sides are
walked in opposite directions, so a sign dropped in the flip would not raise
anything -- it would just settle the group the wrong way and let the conflict
check quietly refuse the move.  The other two checks here are the reasons the
walk is not one straight fit: a branch inside the window, and a border that
runs off the edge of its own tile and is flat along the cut.

Run it directly:  python test_stitch_geometry.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


import stitch_core as SC

TRUE_DX, TRUE_DY = 420, 260          # tile 1 sits here, relative to tile 0

WALL_H, WALL_W = 460, 900            # a scrap of mosaic raster, for the wall walk
WALL_SLOPE = 0.15
CONTACT = (450.0, 230.0)
_SEEN = np.ones((WALL_H, WALL_W), bool)   # nothing cut off, unless a test cuts it


def _two_overlapping_tiles(tmp: Path) -> SC.TileSet:
    """Two crops of one blotchy picture, overlapping by a few hundred pixels."""
    rng = np.random.default_rng(7)
    big = SC.gaussian(rng.random((1100, 1400)).astype(np.float32), 6.0)  # blobs
    big = (big - big.min()) / (float(big.max() - big.min()) + 1e-9)
    canvas = np.full((1100, 1400, 3), 245, np.uint8)
    body = slice(60, 1040), slice(60, 1340)
    canvas[body] = (245 - 150 * big[body][..., None] * [1.0, 0.55, 0.75]).astype(np.uint8)

    h, w = 800, 900
    paths = []
    for k, (x, y) in enumerate(((0, 0), (TRUE_DX, TRUE_DY))):
        p = tmp / f"tile{k}.png"
        Image.fromarray(canvas[y:y + h, x:x + w]).save(p)
        paths.append(p)
    return SC.TileSet(SC.StitchParams()).load_paths(paths)


def test_refine_pair_recovers_a_known_offset() -> None:
    with tempfile.TemporaryDirectory() as d:
        ts = _two_overlapping_tiles(Path(d))
        # Start from the truth, off by a couple of pixels: refinement is a local
        # search, and this is the state it is actually asked to improve.
        for nudge in ((0, 0), (3, -2), (-4, 5)):
            m = SC.PairMatch(i=0, j=1, tx=TRUE_DX + nudge[0], ty=TRUE_DY + nudge[1],
                             ncc=0.9, peak_ratio=2.0, n_overlap=1e5, overlap_frac=0.4,
                             quality=0.9)
            r = SC.refine_pair(ts, m)
            assert r.stage == "refined", f"refinement bailed out, nudge {nudge}"
            off = np.hypot(r.tx - TRUE_DX, r.ty - TRUE_DY)
            assert off < 1.0, f"nudge {nudge}: came back {off:.1f} px from the truth"
            # And it must not have thrown away the coarse ambiguity verdict:
            # inside a +-12 px window every peak looks like a ridge.
            assert r.peak_ratio == m.peak_ratio


def _wall(x0: float, x1: float, shift: float = 0.0, branch: bool = False) -> np.ndarray:
    """A straight band of wall between x0 and x1, optionally with a branch on it."""
    yy, xx = np.mgrid[0:WALL_H, 0:WALL_W].astype(float)
    line = CONTACT[1] + WALL_SLOPE * (xx - CONTACT[0]) + shift
    band = (np.abs(yy - line) < 22.0) & (xx >= x0) & (xx < x1)
    if branch:
        band |= ((np.abs((yy - 120) - 1.9 * (xx - 250)) < 60)
                 & (xx > 150) & (xx < 330) & (yy < line))
    return band


def test_wall_track_measures_the_step_across_a_gap() -> None:
    """Two pieces of one wall, a gap between them, and a known step at the gap.

    ``settle_sliver_joins`` slides a group until this step reads zero, so the
    step has to come back with the right size *and* the right sign from either
    side -- the two walls are walked in opposite directions, and a sign lost in
    that flip would leave the settle confidently sliding the wrong way.
    """
    along = np.array([1.0, 0.0])
    for step in (+60.0, -60.0, 0.0):
        left = _wall(0, 400)
        right = _wall(500, WALL_W, shift=step)
        back = SC._wall_track(left, _SEEN, *CONTACT, along, -1.0, 300.0)
        forward = SC._wall_track(right, _SEEN, *CONTACT, along, +1.0, 300.0)
        assert back is not None and forward is not None, f"lost the wall at step {step}"
        got = forward[0] - back[0]
        assert abs(got - step) < 2.0, f"step {step:+.0f} came back as {got:+.1f}"
        for name, w in (("back", back), ("forward", forward)):
            assert abs(w[1] - WALL_SLOPE) < 0.02, f"{name} slope {w[1]:+.3f}"


def test_wall_track_is_not_dragged_off_by_a_branch() -> None:
    """A branch inside the window must not move where the wall says it is.

    This is what the walk exists for.  One straight fit over everything within
    reach of the contact reads the branch as part of the wall and leaves the
    tissue altogether, and the step it then reports is a step the section does
    not have.
    """
    along = np.array([1.0, 0.0])
    perp = np.array([-along[1], along[0]])
    moved = []
    for branch in (False, True):
        band = _wall(0, 400, branch=branch)
        walked = SC._wall_track(band, _SEEN, *CONTACT, along, -1.0, 300.0)
        fitted = SC._band_axis(band, *CONTACT, 300.0)
        assert walked is not None and fitted is not None
        moved.append((walked[0], float((fitted[0] - np.array(CONTACT)) @ perp)))
    walk_shift = abs(moved[1][0] - moved[0][0])
    chord_shift = abs(moved[1][1] - moved[0][1])
    assert walk_shift < 10.0, f"the walk moved {walk_shift:.1f} px when the branch appeared"
    assert chord_shift > 3 * walk_shift, (
        f"a straight fit moved {chord_shift:.1f} px and the walk {walk_shift:.1f}: "
        "the branch is no longer the problem this walk was written for")


def test_wall_track_extrapolates_a_border_the_picture_cuts_off() -> None:
    """A border along the edge of the picture is the tile's border, not the wall's.

    Where a piece runs out of image the band stops with it, and the border it
    draws there is the cut: straight, convincing, and lying across exactly the
    place the two pieces are compared.  Worse, the cut is usually nearest the
    contact -- the wall leaves the tile through the same corner the next tile
    was supposed to continue from -- so the real border only comes out from
    behind it at the far end of the walk, and has to be carried back.
    """
    along = np.array([1.0, 0.0])
    band = _wall(0, 400)
    yy, xx = np.mgrid[0:WALL_H, 0:WALL_W].astype(float)
    seen = yy >= 228.0 + 0.25 * (xx - CONTACT[0])   # this tile's picture stops here
    truth = SC._wall_track(band, _SEEN, *CONTACT, along, -1.0, 300.0)
    honest = SC._wall_track(band & seen, seen, *CONTACT, along, -1.0, 300.0)
    naive = SC._wall_track(band & seen, _SEEN, *CONTACT, along, -1.0, 300.0)
    assert truth and honest and naive
    want = truth[5]["minus"][0]
    assert honest[5]["minus"] is not None, "the cut border was thrown away, not extrapolated"
    assert honest[5]["minus"][4] > 100.0, (
        "the border was read from inside the cut: this picture no longer hides "
        "the near end of it, which is the case the walk was written for")
    off_honest = abs(honest[5]["minus"][0] - want)
    off_naive = abs(naive[5]["minus"][0] - want)
    assert off_honest < 1.5, f"the extrapolated border came back {off_honest:.1f} px off"
    assert off_naive > 3 * max(off_honest, 1.0), (
        f"reading the cut literally was only {off_naive:.1f} px off: "
        "this picture no longer cuts the border it was built to cut")
    # The border the cut never touched has to be left exactly where it was.
    assert abs(honest[5]["plus"][0] - truth[5]["plus"][0]) < 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
