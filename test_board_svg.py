"""The SVG beside each vector figure, which is what omnifig's board takes.

Two ways this quietly stops working: the figure saves with its text outlined,
so the board gets shapes nobody can retype, or the page asks the server for a
path the server never sends, so the button simply never appears.  Both are
silent -- a missing button looks like a design decision -- so they are checked
here rather than noticed in a tab.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np

import fibrosis as FB

HERE = Path(__file__).resolve().parent


def test_a_saved_figure_keeps_its_text_live_beside_its_picture():
    rng = np.random.default_rng(0)
    h = w = 320
    rgb = np.ones((h, w, 3), float)
    yy, xx = np.mgrid[0:h, 0:w]
    tissue = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (0.42 * w) ** 2
    rgb[tissue] = (0.80, 0.45, 0.60)                    # red counterstain
    rgb[tissue & (np.abs(yy - h / 2) < 22)] = (0.35, 0.45, 0.80)   # collagen
    rgb = np.clip(rgb + rng.normal(0, 0.01, rgb.shape), 0, 1)

    res = FB.analyze_fibrosis(rgb, 1.0, organ="heart")
    with tempfile.TemporaryDirectory() as d:
        paths = FB.save_results(res, d, "t", title="a title", organ="heart")
        assert "figure_svg" in paths, sorted(paths)
        svg = Path(paths["figure_svg"]).read_text()
    # Live text, not outlines: svg.fonttype "none" writes the words, the
    # default writes a <path> per glyph and the board cannot edit them.
    assert svg.count("<text") > 10, svg.count("<text")
    assert "a title" in svg
    # And the picture itself came along, as the one thing in the figure that
    # cannot be vector.
    assert "<image" in svg


def test_every_svg_the_page_asks_for_is_one_the_server_sends():
    js = (HERE / "static" / "app.js").read_text()
    wanted = set(re.findall(r"\br\.(svg|\w+_svg)\b", js))
    assert wanted, "no board buttons left in the page?"
    sent = set(re.findall(r'"(\w*svg)":', (HERE / "app.py").read_text()))
    # The cohort route returns save_cohort_plot's dict whole, and the depth
    # figure's under a "depth_" prefix, so those two names are its own.
    sent |= {k for k in re.findall(r'"(\w*svg)": str\(', (HERE / "cohort.py").read_text())}
    sent |= {f"depth_{k}" for k in sent}
    assert not wanted - sent, sorted(wanted - sent)


if __name__ == "__main__":
    test_a_saved_figure_keeps_its_text_live_beside_its_picture()
    test_every_svg_the_page_asks_for_is_one_the_server_sends()
    print("ok")
