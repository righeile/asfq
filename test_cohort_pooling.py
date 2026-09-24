#!/usr/bin/env python3
"""
What happens to the media-quality medians when sections are pooled.

Everything else in ``cohort.pool`` is an integral and adds up; this block is
medians and spreads and does not.  The tests below are about the one thing
that can still go wrong quietly: a mouse's 0.4 mm off-cut counting as much as
its 4 mm ring.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import cohort as CO


def _sections(**cols) -> pd.DataFrame:
    """Two sections of one mouse, with whatever columns the test needs."""
    n = len(next(iter(cols.values())))
    base = {
        "name": [f"sec{i}" for i in range(n)],
        "genotype": ["KO"] * n,
        "mouse": ["1749"] * n,
        "segment": ["aortic arch"] * n,
        # The integrals pooling requires; their values do not matter here.
        "media_area_um2": [1e5] * n,
        "collagen_od_um2_total": [1e4] * n,
        "muscle_od_um2_total": [1e4] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_a_short_section_does_not_count_as_much_as_a_long_one():
    df = _sections(wall_length_um=[4000.0, 400.0],
                   muscle_variation=[0.10, 0.50])
    got = CO.pool(df, ["genotype", "mouse"])["muscle_variation"].iloc[0]
    # 4 mm at 0.10 and 0.4 mm at 0.50: the wall is mostly the first section.
    assert abs(got - (4000 * 0.10 + 400 * 0.50) / 4400) < 1e-9
    assert abs(got - 0.30) > 0.15, "this is the unweighted mean, not a pooled one"


def test_a_fraction_of_wall_length_pools_exactly():
    # fragmented_length_fraction is a fraction *of wall length*, so the
    # length-weighted mean is not an approximation -- it is the pooled
    # fraction, total fragmented length over total length.
    lengths = np.array([4000.0, 400.0, 1200.0])
    fracs = np.array([0.05, 0.60, 0.20])
    df = _sections(wall_length_um=list(lengths),
                   fragmented_length_fraction=list(fracs))
    got = CO.pool(df, ["genotype", "mouse"])["fragmented_length_fraction"].iloc[0]
    assert abs(got - (lengths * fracs).sum() / lengths.sum()) < 1e-12


def test_spacing_is_weighted_by_the_wall_that_produced_it():
    # The second section is long but only a tenth of it yielded three
    # lamellae, so it must not drag the pooled spacing to its own value.
    df = _sections(wall_length_um=[1000.0, 4000.0],
                   lamellar_spacing_um=[5.0, 9.0],
                   lamellar_coverage=[0.90, 0.10])
    row = CO.pool(df, ["genotype", "mouse"]).iloc[0]
    w = np.array([1000 * 0.90, 4000 * 0.10])
    assert abs(row["lamellar_spacing_um"] - np.average([5.0, 9.0], weights=w)) < 1e-9
    # Coverage itself is a fraction of length, so it pools on length alone.
    assert abs(row["lamellar_coverage"] - (1000 * 0.90 + 4000 * 0.10) / 5000) < 1e-12


def test_a_section_that_measured_nothing_is_not_a_zero():
    df = _sections(wall_length_um=[4000.0, 400.0],
                   muscle_variation=[0.10, float("nan")])
    row = CO.pool(df, ["genotype", "mouse"]).iloc[0]
    assert abs(row["muscle_variation"] - 0.10) < 1e-12
    df2 = _sections(wall_length_um=[4000.0, 400.0],
                    muscle_variation=[float("nan")] * 2)
    assert not np.isfinite(CO.pool(df2, ["genotype", "mouse"])["muscle_variation"].iloc[0])


def test_switching_it_off_leaves_the_column_out_rather_than_guessing():
    df = _sections(wall_length_um=[4000.0, 400.0], muscle_variation=[0.10, 0.50])
    pooled = CO.pool(df, ["genotype", "mouse"], medians="off")
    assert "muscle_variation" not in pooled.columns
    # And the whole ladder still runs with it off.
    tables = CO.summarise(df, medians="off")
    assert "muscle_variation" not in tables["by_mouse"].columns
    assert "muscle_variation" not in set(tables["by_genotype"]["readout"])


def test_the_block_reaches_the_genotype_comparison():
    df = pd.concat([
        _sections(wall_length_um=[4000.0], muscle_variation=[0.12]),
        _sections(wall_length_um=[3000.0], muscle_variation=[0.05]).assign(
            genotype="WT", mouse="1750", name="sec9"),
    ], ignore_index=True)
    tables = CO.summarise(df)
    assert "muscle_variation" in set(tables["by_genotype"]["readout"])
    assert "muscle_variation" in set(tables["comparison"]["readout"])


# -- the exact rebuild: positions, not section medians -----------------------


def _profile(tmp: Path, name: str, **cols) -> str:
    """An along-wall table of the shape the batch writes, with every position
    counted."""
    n = len(next(iter(cols.values())))
    pd.DataFrame({"counted": [True] * n, **cols}).to_csv(tmp / name, index=False)
    return str(tmp / name)


def test_one_section_re_derives_to_its_own_number():
    # The strongest check there is: pooling a single section must reproduce
    # what analyze_wall already reported for it, or the two reductions differ.
    # Exact here because the values are written and read as float64.  On a
    # real section the profile arrays are float32 and the rebuild is float64
    # after the CSV, so they agree to ~1e-7 relative, not to the bit -- do not
    # tighten a real-data comparison past that.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        v = list(np.linspace(0.02, 0.30, 501))
        df = _sections(wall_length_um=[501.0], arc_step_um=[1.0],
                       muscle_variation=[float(np.median(v))],
                       profile_csv=[_profile(tmp, "a.csv", muscle_variation=v)])
        row = CO.pool(df, ["genotype", "mouse"], medians="exact").iloc[0]
        assert row["medians_from"] == "positions"
        assert abs(row["muscle_variation"] - float(np.median(v))) < 1e-12


def test_one_bad_section_does_not_drag_a_mouse_with_it():
    # The case the weighted mean cannot do.  Two equally long sections: one
    # clean throughout, one half clean and half ruined.  Three quarters of
    # this mouse's wall is clean, and the median of its positions says so;
    # the mean of the two section medians lands where no position is.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean = [0.05] * 1000
        mixed = [0.05] * 500 + [0.40] * 500
        df = _sections(
            wall_length_um=[1000.0, 1000.0], arc_step_um=[1.0, 1.0],
            muscle_variation=[float(np.median(clean)), float(np.median(mixed))],
            profile_csv=[_profile(tmp, "a.csv", muscle_variation=clean),
                         _profile(tmp, "b.csv", muscle_variation=mixed)])
        exact = CO.pool(df, ["genotype", "mouse"], medians="exact").iloc[0]
        weighted = CO.pool(df, ["genotype", "mouse"]).iloc[0]
        assert abs(exact["muscle_variation"] - 0.05) < 1e-12
        assert abs(weighted["muscle_variation"] - 0.1375) < 1e-9
        assert weighted["medians_from"] == "section medians, by wall length"


def test_coverage_and_the_fragmented_fraction_are_rebuilt_too():
    from wall_analysis import FRAGMENT_FILL, FRAGMENT_REACH_UM
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 400 positions: 100 fragmented (far enough out, not solid), 100 solid
        # past the edge, 100 close in, 100 with nothing measured at all.
        reach = [20.0] * 100 + [20.0] * 100 + [1.0] * 100 + [20.0] * 100
        fill = [0.2] * 100 + [0.95] * 100 + [0.2] * 100 + [float("nan")] * 100
        dis = [0.1] * 250 + [float("nan")] * 150
        df = _sections(wall_length_um=[400.0], arc_step_um=[1.0],
                       profile_csv=[_profile(
                           tmp, "a.csv", muscle_beyond_edge_um=reach,
                           muscle_beyond_edge_fill=fill, lamellar_disorder=dis)])
        row = CO.pool(df, ["genotype", "mouse"], medians="exact").iloc[0]
        assert FRAGMENT_REACH_UM < 20.0 and FRAGMENT_FILL > 0.2, "test data no longer fits"
        # 100 fragmented out of the 300 positions where fill was measurable.
        assert abs(row["fragmented_length_fraction"] - 100 / 300) < 1e-12
        assert abs(row["lamellar_coverage"] - 250 / 400) < 1e-12


def test_a_group_it_cannot_read_whole_falls_back_rather_than_shrinking():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df = _sections(wall_length_um=[1000.0, 1000.0], arc_step_um=[1.0, 1.0],
                       muscle_variation=[0.05, 0.40],
                       profile_csv=[_profile(tmp, "a.csv",
                                             muscle_variation=[0.05] * 10),
                                    str(tmp / "gone.csv")])
        row = CO.pool(df, ["genotype", "mouse"], medians="exact").iloc[0]
        assert row["medians_from"] == "section medians, by wall length"
        assert abs(row["muscle_variation"] - 0.225) < 1e-9


def test_positions_of_different_lengths_are_not_interchangeable():
    # One section stepped at 1 um and one at 4 um: a position is not the same
    # amount of wall in the two, so stacking them would weight the coarse one
    # four times over.  That group goes up the weighted way instead.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df = _sections(wall_length_um=[1000.0, 1000.0], arc_step_um=[1.0, 4.0],
                       muscle_variation=[0.05, 0.40],
                       profile_csv=[_profile(tmp, "a.csv", muscle_variation=[0.05] * 1000),
                                    _profile(tmp, "b.csv", muscle_variation=[0.40] * 250)])
        row = CO.pool(df, ["genotype", "mouse"], medians="exact").iloc[0]
        assert row["medians_from"] == "section medians, by wall length"


# -- the curve through the wall ---------------------------------------------


def _curve(mouse, genotype, age, muscle, n_rows, name):
    """One section's depth profile, of the shape the batch stacks."""
    depths = np.linspace(0.0, 1.0, 5)
    return pd.DataFrame({
        "segment": "aortic arch", "age": age, "genotype": genotype,
        "mouse": mouse, "name": name, "depth_relative": depths,
        "muscle_mean": float(muscle), "collagen_mean": 1.0 - float(muscle),
        "n_rows": n_rows})


def test_a_short_section_does_not_pull_the_curve_like_a_long_one():
    # 4000 positions at 0.10 and 400 at 0.50, one mouse.  The same arithmetic
    # as pooling the readouts, applied at every depth.
    long = pd.concat([_curve("A", "KO", "12W", 0.10, 4000, "s1"),
                      _curve("A", "KO", "12W", 0.50, 400, "s2")],
                     ignore_index=True)
    avg = CO.average_depth_profile(long)
    got = float(avg.muscle_mean.iloc[0])
    assert abs(got - (4000 * 0.10 + 400 * 0.50) / 4400) < 1e-9
    assert abs(got - 0.30) > 0.15, "this is the unweighted mean of the two sections"


def test_a_depth_a_section_never_reached_does_not_dilute_it():
    # The second section's outer half is missing -- a ragged outer edge.  It
    # must stop contributing there, not pull the curve towards nothing.
    a = _curve("A", "KO", "12W", 0.10, 1000, "s1")
    b = _curve("A", "KO", "12W", 0.50, 1000, "s2")
    b.loc[b.depth_relative > 0.5, ["muscle_mean", "n_rows"]] = [np.nan, 0]
    avg = CO.average_depth_profile(pd.concat([a, b], ignore_index=True))
    deep = avg[avg.depth_relative > 0.5]
    assert np.allclose(deep.muscle_mean, 0.10), "the missing half still counted"
    assert abs(float(avg.muscle_mean.iloc[0]) - 0.30) < 1e-9


def test_ages_are_not_averaged_together():
    long = pd.concat([_curve("A", "KO", "12W", 0.10, 100, "s1"),
                      _curve("B", "KO", "32W", 0.50, 100, "s2")],
                     ignore_index=True)
    avg = CO.average_depth_profile(long)
    assert set(avg.age) == {"12W", "32W"}
    assert sorted(avg.groupby("age").muscle_mean.first()) == [0.10, 0.50]
    # And n is animals, not sections: one mouse each here.
    assert set(avg.n_mice) == {1}


def test_n_is_mice_not_sections():
    long = pd.concat([_curve("A", "KO", "12W", 0.10, 100, f"s{i}") for i in range(6)]
                     + [_curve("B", "KO", "12W", 0.20, 100, "s9")], ignore_index=True)
    avg = CO.average_depth_profile(long)
    assert set(avg.n_mice) == {2}, "six sections from one mouse counted six times"
    assert abs(float(avg.muscle_mean.iloc[0]) - 0.15) < 1e-9


def test_every_segment_word_in_use_is_recognised():
    """A folder whose segment nothing claims pools with the unknowns.

    That is silent -- the section is measured, reported and averaged, just
    against the wrong neighbours -- so the vocabulary is checked against the
    words the folders actually use rather than against itself.
    """
    for folder, want in [("3690_middle aorta_sec1c", "middle aorta"),
                         ("2139_mid-aorta_sec2", "middle aorta"),
                         ("1805_aortic arch_sec3", "aortic arch"),
                         ("1971_thoracic aorta_sec2", "thoracic aorta"),
                         ("4_1749_KO, 32W_thoracic_slide 3_sec1", "thoracic aorta")]:
        got = CO.parse_section_name(folder).segment
        assert got == want, f"{folder!r} parsed as {got!r}, not {want!r}"
    # And it stays a vocabulary, not a substring hunt.
    assert CO._find_segment("middle ear") == ""


def test_the_depth_figure_lays_out_one_panel_per_condition_at_the_size_asked_for():
    """Both stains together, and a plotting area that is the size it was told.

    The cross product of region, age and genotype is mostly empty on a real
    cohort -- six conditions in eighteen slots -- so the panels wrap instead.
    And the size control is the point of the figure: two panels only compare
    if the *data area* matches, which is not what ``figsize`` pins.
    """
    import matplotlib
    matplotlib.use("Agg")

    long = pd.concat([_curve("A", "KO", "12W", 0.10, 100, "s1"),
                      _curve("B", "WT", "32W", 0.50, 100, "s2")], ignore_index=True)
    avg = CO.average_depth_profile(long)
    spec = CO.PlotSpec(width_mm=180, font_pt=8,
                       axes_width_mm=50.8, axes_height_mm=50.8)   # 2 x 2 inches

    fig = CO.plot_depth_profiles(avg, spec=spec, stains_together=True)
    fig.canvas.draw()
    drawn = [ax for ax in fig.axes if ax.get_subplotspec() is not None and ax.axison]
    assert len(drawn) == 2, f"two conditions, {len(drawn)} panels"
    assert all(len(ax.get_lines()) == 2 for ax in drawn), "both stains in every panel"
    for ax in drawn:
        box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert abs(box.width - 2.0) < 0.02 and abs(box.height - 2.0) < 0.02, \
            f"asked for a 2 x 2 in panel, got {box.width:.3f} x {box.height:.3f}"

    # The default figure is the other way up: a column per stain, and the
    # genotypes overlaid in it.
    fig = CO.plot_depth_profiles(avg, spec=spec)
    fig.canvas.draw()
    drawn = [ax for ax in fig.axes if ax.get_subplotspec() is not None and ax.axison]
    assert len(drawn) == 4, f"two conditions x two stains, {len(drawn)} panels"
    assert {ax.get_ylabel() for ax in drawn} == {"muscle (OD)", "collagen (OD)"}


def test_the_group_labels_stay_apart_on_a_panel_too_narrow_for_them():
    """A 25 mm panel with breeding-scheme names under it, measured on the PNG.

    Tilting the labels 30 degrees is enough at 50 mm and is not enough here:
    "L8A-Sox10-Cre" and "L8D-E-KO (n=1)" printed straight through each other,
    which is the way a size control must never fail -- asking for a smaller
    panel has to make the panel smaller, not make the figure wrong.  So the
    check is the one nobody does by eye: render it, then ask every pair of
    neighbouring tick labels whether their boxes touch.
    """
    import matplotlib
    matplotlib.use("Agg")

    rng = np.random.default_rng(0)
    rows = []
    for genotype, n in [("L8A-Sox10-Cre", 1), ("L8D-E-KO", 1), ("unknown", 7)]:
        for i in range(n):
            rows.append({"mouse": f"{genotype}-{i}", "genotype": genotype,
                         "segment": "aortic arch",
                         "collagen_to_muscle_ratio": 0.5 + rng.normal(0, 0.05),
                         "mean_media_thickness_um": 42 + rng.normal(0, 4)})
    tables = {"by_mouse": pd.DataFrame(rows)}

    with tempfile.TemporaryDirectory() as tmp:
        fig = CO.plot_cohort(
            tables, unit="mouse", panel_mm=25.0, panel_h_mm=25.0,
            readouts=["collagen_to_muscle_ratio", "mean_media_thickness_um"])
        fig.savefig(Path(tmp) / "cohort.png")
        fig.draw_without_rendering()

    panels = [ax for ax in fig.axes if ax.get_subplotspec() is not None and ax.axison]
    assert len(panels) == 2, f"two readouts, {len(panels)} panels"
    for ax in panels:
        # The panel is still the size that was asked for: the labels are
        # allowed to make the *canvas* taller, never the plotting area smaller.
        box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert abs(box.width / CO.LKP.MM - 25.0) < 0.3, \
            f"asked for a 25 mm panel, got {box.width / CO.LKP.MM:.2f} mm"
        ticks = [t for t in ax.get_xticklabels() if t.get_text()]
        assert len(ticks) == 3, f"three genotypes, {len(ticks)} labels"
        boxes = sorted(((t.get_window_extent(), t.get_text()) for t in ticks),
                       key=lambda b: b[0].x0)
        for (a, left), (b, right) in zip(boxes, boxes[1:]):
            assert a.x1 <= b.x0, \
                f"{left!r} runs to {a.x1:.1f} px and {right!r} starts at {b.x0:.1f}"


def test_a_region_typed_by_hand_survives_a_round_trip_and_keeps_the_rest():
    """
    Writing a segment must leave the file readable by the thing that reads it,
    and must not eat the parts of it nobody asked about.

    ``metadata.csv`` is the user's file, not ours: the whole point of writing
    it from the browser is that a hand-made one and a typed one are the same
    file.  So the round trip is checked both ways, an unrelated column has to
    come back, and a blank has to *clear* the override rather than record an
    empty region -- otherwise clearing one would pool the section into a group
    called "".
    """
    with tempfile.TemporaryDirectory() as tmp:
        tray = Path(tmp)
        (tray / "metadata.csv").write_text(
            "folder,segment,notes\n2139_sec1,aortic arch,cut thin\n")

        CO.write_overrides(tray, {"2139_sec1": {"mouse": "2139"},
                                  "3690_sec2": {"segment": "middle aorta"}})
        back = CO.read_overrides(tray)
        assert back["2139_sec1"] == {"mouse": "2139", "segment": "aortic arch"}
        assert back["3690_sec2"] == {"segment": "middle aorta"}
        assert "cut thin" in (tray / "metadata.csv").read_text(), "kept the notes column"

        # What the browser actually sends, folder paths and all.
        CO.write_overrides(tray, {str(tray / "2139_sec1"): {"segment": ""}})
        assert "segment" not in CO.read_overrides(tray)["2139_sec1"], "blank clears it"

        # And the override is what the section ends up carrying, whatever its
        # name said -- which is the only reason any of this exists.
        meta = CO.parse_section_name("3690_thoracic_sec2")
        assert meta.segment == "thoracic aorta"
        for field, value in CO.read_overrides(tray).get("3690_sec2", {}).items():
            setattr(meta, field, value)
        assert meta.segment == "middle aorta"
        assert "middle aorta" in CO.segment_names(), "and it is in the offered vocabulary"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all good")
