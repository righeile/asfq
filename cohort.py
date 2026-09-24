#!/usr/bin/env python3
"""
cohort.py -- from a tray of section folders to a comparison between genotypes.

One section is not an experiment.  A study is many sections, cut from a few
slides, from a handful of aortic segments, from a number of mice, in two or
more genotypes, and the arithmetic that takes you from one to the other is
where results are quietly won and lost.  This module does that arithmetic
explicitly, in three steps:

1. **Name to metadata.**  ``parse_section_name`` reads mouse, genotype, age,
   segment, slide and section out of a folder name like
   ``4_1749_L8D-E-KO, 32W_aortic arch_slide 9_sec2``.  Whatever it works out is
   shown back to you, and a ``metadata.csv`` next to the folders overrides it,
   because a naming convention is a convention and not a guarantee.

2. **Pool by summing integrals, never by averaging ratios.**  Two sections of
   the same segment in the same mouse are two samples of one wall, and the
   honest pooled ratio is total collagen over total muscle -- not the mean of
   the two section ratios, which silently weights a 0.5 mm off-cut the same as
   a 4 mm ring.  Every pooled number here is recomputed from summed extensive
   quantities (stain integrals, wall length, media area).  ``mean_media_
   thickness_um`` comes back as summed area over summed length, which *is* the
   length-weighted mean thickness.

3. **Compare with the mouse as the unit.**  Sections and slides are repeated
   measures of one animal, so counting them as independent inflates n and
   shrinks p by a factor that has nothing to do with the biology.  Genotypes
   are compared across per-mouse values, n = number of mice, and the section
   count is reported alongside so the pseudoreplication is visible rather than
   buried.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# 1.  Naming conventions
# =============================================================================

# Longest and most specific first: "descending thoracic" must win over
# "thoracic", and "aortic arch" over a bare "aorta".
SEGMENT_PATTERNS: list[tuple[str, str]] = [
    ("aortic root",         r"\b(aortic\s+root|aortic\s+sinus|sinus\s+of\s+valsalva)\b"),
    ("ascending aorta",     r"\bascending\b"),
    ("aortic arch",         r"\b(aortic\s+arch|arch)\b"),
    # The stretch between the arch and the thoracic section, which these
    # folders call "middle aorta".  Nothing else in the list claimed it, so
    # until it was added those sections carried no segment at all and pooled
    # with the unknowns -- which is the one thing segments exist to prevent.
    ("middle aorta",        r"\bmid(dle)?[-\s]*aorta\b"),
    ("descending thoracic", r"\bdescending\s+thoracic\b"),
    ("thoracic aorta",      r"\bthoracic\b"),
    ("descending aorta",    r"\bdescending\b"),
    ("suprarenal aorta",    r"\bsuprarenal\b"),
    ("infrarenal aorta",    r"\binfrarenal\b"),
    ("abdominal aorta",     r"\babdominal\b"),
    ("carotid artery",      r"\bcarotid\b"),
    ("femoral artery",      r"\bfemoral\b"),
    ("mesenteric artery",   r"\bmesenteric\b"),
]

# The same idea for the other organs.  "Segment" is the aorta's word for it;
# in a heart or a kidney the equivalent is the region the field was taken
# from, and it plays exactly the same role -- the thing you must hold constant
# before comparing genotypes, because baseline collagen differs between
# regions far more than genotype moves it within one.
#
# A kidney field usually spans several regions at once, and the folder names
# say which ("Pap, Med, Ctx").  Those combinations are kept as they are rather
# than reduced to one region, because a field of cortex alone and a field
# running from papilla to cortex are not the same measurement and pooling them
# together is precisely the mistake this column exists to prevent.
HEART_PATTERNS: list[tuple[str, str]] = [
    ("left ventricle",   r"\b(lt|left|l)\.?\s*vent"),
    ("right ventricle",  r"\b(rt|right|r)\.?\s*vent"),
    ("septum",           r"\b(septum|ivs)\b"),
    ("atrium",           r"\batri"),
    ("whole heart",      r"\bwhole\s+heart\b"),
]

_KIDNEY_REGIONS = [
    ("papilla",       r"\b(pap|papilla)\b"),
    ("inner medulla", r"\binner\s+med(ulla)?\b"),
    ("outer medulla", r"\bouter\s+med(ulla)?\b"),
    ("medulla",       r"\bmed(ulla)?\b"),
    ("cortex",        r"\b(ctx|cortex|cortical)\b"),
]

# A genotype token is recognised by the vocabulary breeders actually use.
_GENOTYPE_HINT = re.compile(
    r"(\bKO\b|\bWT\b|\bHET\b|\bHOM\b|\bCTRL\b|\bcontrol\b|"
    r"[-+]/[-+]|\bfl/fl\b|\bflox)", re.IGNORECASE)


@dataclass
class SectionMeta:
    """What a folder name says about the section inside it."""

    folder: str
    name: str
    mouse: str = ""
    genotype: str = ""
    age: str = ""
    segment: str = ""
    slide: str = ""
    section: str = ""
    # A grade written into the folder name by whoever cut the section
    # ("minimal fibrosis", "huge fibrosis").  Carried so it can be looked at
    # beside the measurement, never used in one: it is an eyeball label, and a
    # measurement that has been tuned until it agrees with one is no longer an
    # independent check on it.
    grade: str = ""
    unparsed: str = ""       # the parts of the name nothing claimed

    def missing(self) -> list[str]:
        return [f for f in ("mouse", "genotype", "segment") if not getattr(self, f)]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["missing"] = self.missing()
        return d


def _find_kidney_region(text: str) -> str:
    """
    Every kidney region the text names, in anatomical order.

    Order is imposed rather than taken from the name so that "Med, Ctx" and
    "Ctx, Med" are one group and not two -- otherwise the same field written
    two ways lands in two different rows of the summary.  A more specific
    medulla wins over the general one, so "Outer Med, Ctx" does not also
    report a bare medulla.
    """
    low = text.lower()
    found = [name for name, pat in _KIDNEY_REGIONS if re.search(pat, low)]
    if "inner medulla" in found or "outer medulla" in found:
        found = [f for f in found if f != "medulla"]
    return ", ".join(found)


def _find_segment(text: str, organ: str = "aorta") -> str:
    """
    The region this text names, for whichever organ is being measured.

    Kept organ-scoped on purpose: "Med" is a kidney region and also three
    letters that could turn up in an aorta folder name, and a pattern list
    that grows to cover every organ would start claiming tokens in names it
    was never meant to read.
    """
    low = text.lower()
    if organ == "kidney":
        return _find_kidney_region(text)
    table = HEART_PATTERNS if organ == "heart" else SEGMENT_PATTERNS
    for canonical, pattern in table:
        if re.search(pattern, low):
            return canonical
    return ""


def parse_section_name(name: str, folder: str = "",
                       organ: str = "aorta") -> SectionMeta:
    """
    Read metadata out of one folder name.

    The convention this is written against is underscore-separated fields with
    the genotype and age comma-joined, e.g.::

        4_1749_L8D-E-KO, 32W_aortic arch_slide 9_sec2
        ^ ^    ^                ^          ^       ^
        | mouse genotype, age   segment    slide   section
        order

    Nothing here insists on that.  Each field is found by what it looks like,
    so fields in a different order, or missing, cost you only that field: the
    section still gets measured, it just lands in an "unknown" group where you
    can see it rather than being silently folded into a real one.
    """
    meta = SectionMeta(folder=folder or name, name=name)
    tokens = [t.strip() for t in name.split("_") if t.strip()]
    claimed: set[int] = set()

    for i, tok in enumerate(tokens):
        # "huge fibrosis", "minimal fibrosis" -- and "buge fibrosis", which is
        # what a typo in a folder name looks like.  Matching on the noun and
        # keeping whatever adjective precedes it means a misspelt grade is
        # still recognisably a grade rather than silently unparsed.
        m = re.fullmatch(r"(.*?)\s*fibrosis", tok, re.IGNORECASE)
        if m and not meta.grade:
            meta.grade, _ = (m.group(1).strip().lower() or "graded"), claimed.add(i)
            continue
        if re.fullmatch(r"all\s+regions", tok, re.IGNORECASE):
            claimed.add(i)          # says nothing the region field does not
            continue
        if re.fullmatch(r"p\d+", tok, re.IGNORECASE) and not meta.slide:
            meta.slide, _ = tok.lower(), claimed.add(i)
            continue
        m = re.fullmatch(r"sec\w*\s*(\d+[a-z]?)", tok, re.IGNORECASE)
        if m and not meta.section:
            meta.section, _ = m.group(1), claimed.add(i)
            continue
        m = re.fullmatch(r"slide\s*(\w+)", tok, re.IGNORECASE)
        if m and not meta.slide:
            meta.slide, _ = m.group(1), claimed.add(i)
            continue
        if _find_segment(tok, organ) and not meta.segment:
            meta.segment, _ = _find_segment(tok, organ), claimed.add(i)
            continue
        # "L8D-E-KO, 32W" -- genotype and age travelling together.
        if "," in tok:
            head, _, tail = tok.partition(",")
            age = re.search(r"(\d+)\s*([WwMmDd])\b", tail)
            if age:
                meta.age = f"{age.group(1)}{age.group(2).upper()}"
                meta.genotype = meta.genotype or head.strip()
                claimed.add(i)
                continue
        if not meta.age:
            age = re.fullmatch(r"(\d+)\s*([WwMmDd])", tok)
            if age:
                meta.age, _ = f"{age.group(1)}{age.group(2).upper()}", claimed.add(i)
                continue
        # A bare animal number: three or more digits and nothing else.
        if not meta.mouse and re.fullmatch(r"\d{3,}", tok):
            meta.mouse, _ = tok, claimed.add(i)
            continue
        if not meta.genotype and _GENOTYPE_HINT.search(tok):
            meta.genotype, _ = tok, claimed.add(i)
            continue

    # A segment named across two fields ("descending_thoracic") only shows up
    # when the whole name is searched at once.
    if not meta.segment:
        meta.segment = _find_segment(name, organ)

    meta.unparsed = " | ".join(t for i, t in enumerate(tokens) if i not in claimed)
    return meta


#: The fields ``metadata.csv`` may carry, and the only ones written back into
#: it.  Blank means "no override": whatever the name parser worked out stands.
OVERRIDE_FIELDS = ("mouse", "genotype", "age", "segment", "slide", "section")


def segment_names(organ: str = "aorta") -> list[str]:
    """
    Every region name this organ's parser can produce, in anatomical order.

    Offered as the suggestions beside the hand-typed field, so a region typed
    for a folder whose name never said one lands on the same spelling as the
    folders that did -- "arch" and "aortic arch" in one column are two groups
    where the tissue is one.  Typing something else is allowed: this vocabulary
    is what gets *read* out of a name, not what a region may be called.
    """
    if organ == "kidney":
        return [name for name, _ in _KIDNEY_REGIONS]
    return [n for n, _ in (HEART_PATTERNS if organ == "heart" else SEGMENT_PATTERNS)]


def read_overrides(parent: str | Path) -> dict[str, dict[str, str]]:
    """
    Metadata typed by hand, keyed by folder name, from ``<parent>/metadata.csv``.

    Any of ``mouse``, ``genotype``, ``age``, ``segment``, ``slide``, ``section``
    may appear; blank cells fall through to whatever the name parser worked
    out.  This is the escape hatch for the sections whose names do not follow
    the convention, and it means a mis-parse never has to be fixed by renaming
    data on disk.
    """
    path = Path(parent) / "metadata.csv"
    try:
        if not path.exists():
            return {}
        df = pd.read_csv(path, dtype=str).fillna("")
    except OSError:
        return {}
    key = next((c for c in ("folder", "name", "section_folder") if c in df.columns), None)
    if key is None:
        return {}
    fields = [c for c in OVERRIDE_FIELDS if c in df.columns]
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        k = str(r[key]).strip()
        if not k:
            continue
        out[Path(k).name] = {f: str(r[f]).strip() for f in fields if str(r[f]).strip()}
    return out


def write_overrides(parent: str | Path,
                    updates: dict[str, dict[str, str]]) -> Path:
    """
    Metadata typed by hand, merged into ``<parent>/metadata.csv``.

    The counterpart of :func:`read_overrides`, and what lets the tray browser
    label a section at all.  The name parser only knows the regions in its own
    vocabulary, and a dataset is free to use one it has never seen -- so rather
    than teach it a new word per cohort, the answer is written down beside the
    data, where it outlives the app and the next scan reads it straight back.

    The file belongs to whoever made it, so it is merged and never rewritten
    from scratch: rows and columns this module has no opinion about are carried
    through untouched, and a blank value *clears* an override instead of
    recording an empty one.
    """
    path = Path(parent) / "metadata.csv"
    try:
        df = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()
    except (OSError, ValueError):
        df = pd.DataFrame()
    key = next((c for c in ("folder", "name", "section_folder") if c in df.columns),
               "folder")
    rows: list[dict[str, str]] = [{k: str(v) for k, v in r.items()}
                                  for _, r in df.iterrows()]
    by_name = {Path(r.get(key, "")).name: r for r in rows}
    for folder, fields in updates.items():
        row = by_name.get(Path(folder).name)
        if row is None:
            row = {key: Path(folder).name}
            rows.append(row)
            by_name[row[key]] = row
        row.update({f: str(v).strip() for f, v in fields.items()
                    if f in OVERRIDE_FIELDS})
    # A row that now says nothing goes, and a column nobody filled in is not
    # written: otherwise clearing a region leaves the file filling up with
    # blank rows that read like data the next time something opens it.
    rows = [r for r in rows if any(str(v).strip() for f, v in r.items() if f != key)]
    cols = [key] + [c for c in dict.fromkeys([*df.columns, *OVERRIDE_FIELDS])
                    if c != key and any(str(r.get(c, "")).strip() for r in rows)]
    pd.DataFrame(rows).reindex(columns=cols).fillna("").to_csv(path, index=False)
    return path


def list_children(parent: str | Path, list_images=None,
                  skipped: list[str] | None = None,
                  organ: str = "aorta") -> list[dict[str, Any]]:
    """
    Every readable sub-folder of ``parent``, section or not.

    A tray is rarely where you start.  Getting to it means passing through
    folders that hold no images of their own -- ``Documents``, ``Apps``, a lab
    folder -- and a browser that lists only image-bearing folders is a browser
    you cannot walk anywhere with.  So everything readable is returned, each
    flagged ``is_section`` if it holds images directly, with its parsed
    metadata when it does.
    """
    parent = Path(parent).expanduser()
    overrides = read_overrides(parent)
    out: list[dict[str, Any]] = []
    for child in sorted(parent.iterdir(), key=lambda q: q.name.lower()):
        try:
            if not child.is_dir() or child.name.startswith((".", "__")):
                continue
            n = len(list_images(child)) if list_images is not None else 0
        except OSError:
            if skipped is not None:
                skipped.append(child.name)
            continue
        meta = parse_section_name(child.name, folder=str(child), organ=organ)
        for field, value in overrides.get(child.name, {}).items():
            setattr(meta, field, value)
        entry = meta.as_dict()
        entry["images"] = n
        entry["is_section"] = n > 0
        out.append(entry)
    return out


def scan_tray(parent: str | Path, list_images=None,
              skipped: list[str] | None = None,
              organ: str = "aorta") -> list[SectionMeta]:
    """
    Every sub-folder of ``parent`` that holds images, with its metadata read.

    This is the "point me at the tray" entry point: one folder per section,
    all of them under one parent, which is how these are kept anyway.

    A folder the process cannot open is skipped, not fatal.  Any real home
    directory has a few -- ``.Trash``, a cloud-drive mount, anything macOS
    guards -- and one of them refusing to be listed is no reason to refuse the
    other forty.  Pass a list as ``skipped`` to be told which ones they were,
    so a section that is quietly absent can be told from one that is not there.
    """
    parent = Path(parent).expanduser()
    overrides = read_overrides(parent)
    out: list[SectionMeta] = []
    for child in sorted(parent.iterdir(), key=lambda q: q.name.lower()):
        try:
            if not child.is_dir() or child.name.startswith((".", "__")):
                continue
            if list_images is not None and not list_images(child):
                continue
        except OSError:
            if skipped is not None:
                skipped.append(child.name)
            continue
        meta = parse_section_name(child.name, folder=str(child), organ=organ)
        for field, value in overrides.get(child.name, {}).items():
            setattr(meta, field, value)
        out.append(meta)
    return out


# =============================================================================
# 2.  Pooling
# =============================================================================

# Quantities that are integrals over the wall, and so may be added together.
EXTENSIVE = [
    "wall_length_um",
    "media_area_um2",
    "collagen_od_um2_total",
    "muscle_od_um2_total",
    "collagen_adventitia_od_um2_total",
    "collagen_area_um2_total",
]

# Per-section medians and spreads -- the media-quality block that
# ``wall_analysis`` produces.  None of these is an integral, so the summing
# above cannot touch them: two medians added together mean nothing.  With
# ``medians="weighted"`` they are averaged over a mouse's sections instead,
# weighted by how much wall each section measured, which is the most
# defensible thing available from the table alone.  ``medians="exact"`` goes
# back to the positions and reduces once, which is better and needs the
# along-wall tables kept; see ``pool``.
#
# Two of them come out exact under the weighting too, not just approximate:
# ``lamellar_coverage`` and ``fragmented_length_fraction`` are already
# fractions *of wall length*, so length-weighting reconstructs the pooled
# fraction precisely.  The other seven are medians over arc positions, and a
# weighted mean of medians is not the median of the pooled positions -- close
# when the sections agree, wrong in the tail when one of them is bimodal.
SECTION_MEDIANS = [
    "muscle_variation",
    "lamellar_contrast",
    "lamellar_spacing_um",
    "lamellar_disorder",
    "lamellar_coverage",
    "section_edge_width_um",
    "median_muscle_beyond_edge_um",
    "median_muscle_beyond_edge_fill",
    "fragmented_length_fraction",
]

# Where a section's per-position values live: the along-wall table the batch
# wrote beside its results, whose path the batch row records.  Only
# ``medians="exact"`` needs it, and a table without the column still pools the
# weighted way, so an older batch loses nothing it used to have.
PROFILE_COLUMN = "profile_csv"

# Which column of that table rebuilds which total.  Each is a median over the
# counted positions -- the same reduction ``analyze_wall`` does per section,
# which is the point: a mouse with one section re-derives to exactly that
# section's own number, and a mouse with three gets one median over all their
# positions instead of a mean of three medians.  There is a test for both.
_MEDIAN_OF = {
    "muscle_variation":               "muscle_variation",
    "lamellar_contrast":              "lamellar_contrast",
    "lamellar_spacing_um":            "lamellar_spacing_um",
    "lamellar_disorder":              "lamellar_disorder",
    "section_edge_width_um":          "edge_width_um",
    "median_muscle_beyond_edge_um":   "muscle_beyond_edge_um",
    "median_muscle_beyond_edge_fill": "muscle_beyond_edge_fill",
}

# Spacing and disorder only exist where three lamellae were found, so the wall
# behind them is the section's length times its coverage, not its length.  A
# section covered 47% would otherwise weigh as much as one covered 90%.
_COVERED_BY = {"lamellar_spacing_um": "lamellar_coverage",
               "lamellar_disorder": "lamellar_coverage"}

# What is quoted, and how each is rebuilt from the sums above.
READOUTS = [
    "collagen_to_muscle_ratio",
    "collagen_fraction_of_stain",
    "mean_media_thickness_um",
    "collagen_od_um2_per_mm_length",
    "muscle_od_um2_per_mm_length",
    "collagen_area_fraction_media",
]

# -- what differs between organs, and what does not --------------------------
#
# An aorta is measured along a wall and a heart is measured over an area, so
# the integrals differ: one divides by millimetres of wall, the other by
# square millimetres of tissue.  Everything above that is the same argument
# and so is the same code -- pool by summing integrals rather than averaging
# ratios, then compare with the mouse as the unit.  A scheme is just the pair
# of organ-specific lists, so the ladder itself is written once.


@dataclass(frozen=True)
class Scheme:
    """Which columns are integrals, and how the quoted numbers rebuild."""

    name: str
    extensive: tuple[str, ...]
    readouts: tuple[str, ...]
    required: tuple[str, ...]
    pooled_label: str = "all segments"
    # Columns that are averaged rather than added, and what weighs them.
    weighted: tuple[str, ...] = ()
    weight: str = "wall_length_um"

    def derive(self, row: dict[str, float]) -> dict[str, float]:
        return (_derive_area(row) if self.name == "area" else _derive_wall(row))

    def fill_extensive(self, df: pd.DataFrame) -> pd.DataFrame:
        return (_extensive_area(df) if self.name == "area" else _extensive_wall(df))


WALL_SCHEME = Scheme(
    name="wall",
    extensive=tuple(EXTENSIVE),
    readouts=tuple(READOUTS),
    # Without these there is nothing to add up, and every pooled number would
    # come back NaN -- which reads like a measurement that failed rather than
    # a table that was never given the columns.
    required=("wall_length_um", "media_area_um2",
              "collagen_od_um2_total", "muscle_od_um2_total"),
    pooled_label="all segments",
    weighted=tuple(SECTION_MEDIANS),
)

AREA_SCHEME = Scheme(
    name="area",
    extensive=("tissue_area_um2", "imaged_area_um2",
               "collagen_od_um2_total", "muscle_od_um2_total",
               "collagen_area_um2_total"),
    readouts=("collagen_to_muscle_ratio",
              "collagen_fraction_of_stain",
              "collagen_area_fraction_tissue",
              "collagen_od_um2_per_mm2_tissue",
              "muscle_od_um2_per_mm2_tissue",
              "tissue_fraction_of_image"),
    required=("tissue_area_um2", "collagen_od_um2_total", "muscle_od_um2_total"),
    pooled_label="all regions",
    # An area analysis has no media to measure the quality of; the field is
    # named anyway so that nothing here weighs a heart by a wall length.
    weight="tissue_area_um2",
)

SCHEMES = {"aorta": WALL_SCHEME, "heart": AREA_SCHEME, "kidney": AREA_SCHEME,
           "wall": WALL_SCHEME, "area": AREA_SCHEME}


def scheme_for(organ: str | Scheme | None) -> Scheme:
    if isinstance(organ, Scheme):
        return organ
    return SCHEMES.get(str(organ or "aorta").lower(), WALL_SCHEME)


def _safe(num: float, den: float) -> float:
    return float(num) / float(den) if den not in (0, None) and np.isfinite(den) and den > 0 else float("nan")


def _weighted_mean(v: pd.Series, w: pd.Series) -> float:
    """Mean of *v* over the rows where both it and its weight are usable."""
    x, q = v.to_numpy(dtype=float), w.to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(q) & (q > 0)
    return float(np.average(x[ok], weights=q[ok])) if ok.any() else float("nan")


def _counted_positions(part: pd.DataFrame) -> pd.DataFrame | None:
    """
    Every counted position of a group's sections, stacked into one table.

    ``None`` unless *all* of them are readable.  Pooling three sections'
    positions with one of them missing quietly changes which group is being
    described, which is a worse failure than not pooling at all -- so the
    whole group falls back together, and the pooled row says it did.
    """
    if PROFILE_COLUMN not in part.columns:
        return None
    paths = part[PROFILE_COLUMN].astype(str).tolist()
    if len(paths) != len(part) or any(not q or q == "nan" for q in paths):
        return None
    want = set(_MEDIAN_OF.values()) | {"counted"}
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, usecols=lambda c: c in want))
        except Exception:
            return None
    prof = pd.concat(frames, ignore_index=True)
    # An along-wall table written before these columns existed reads fine and
    # carries none of them; that is a fallback, not an empty result.
    if not (set(_MEDIAN_OF.values()) & set(prof.columns)):
        return None
    if "counted" in prof.columns:
        prof = prof[prof["counted"].astype(bool)]
    return prof if len(prof) else None


def _exact_medians(part: pd.DataFrame) -> dict[str, float] | None:
    """
    The media-quality block rebuilt from positions instead of from medians.

    This is the same move as summing integrals, one level down: keep what each
    section measured at every position, put a group's positions together, and
    reduce once.  A weighted mean of medians cannot do it -- it lands on a
    value no position in the mouse has whenever one section is bimodal, which
    is the case these measures exist to catch.
    """
    try:
        from wall_analysis import FRAGMENT_FILL, FRAGMENT_REACH_UM
    except Exception:
        # The thresholds are read from the module that applied them rather
        # than copied here, so they cannot drift apart.  Without it there is
        # no exact rebuild, only a guess.
        return None
    # Positions are interchangeable only while they are the same length of
    # wall.  Two batches run at different steps go up the weighted way, where
    # each section carries its own length explicitly.
    if len(_numeric(part, "arc_step_um").dropna().unique()) > 1:
        return None
    prof = _counted_positions(part)
    if prof is None:
        return None

    out: dict[str, float] = {}
    for total, col in _MEDIAN_OF.items():
        if col in prof.columns:
            out[total] = float(_numeric(prof, col).median())
    if "lamellar_disorder" in prof.columns:
        # Coverage is the share of positions where three lamellae were found,
        # so over pooled positions it is a plain mean -- and stays exact.
        out["lamellar_coverage"] = float(np.isfinite(
            _numeric(prof, "lamellar_disorder").to_numpy(dtype=float)).mean())
    if {"muscle_beyond_edge_um", "muscle_beyond_edge_fill"} <= set(prof.columns):
        reach = _numeric(prof, "muscle_beyond_edge_um").to_numpy(dtype=float)
        fill = _numeric(prof, "muscle_beyond_edge_fill").to_numpy(dtype=float)
        ok = np.isfinite(fill)
        out["fragmented_length_fraction"] = float(
            ((reach > FRAGMENT_REACH_UM) & (fill < FRAGMENT_FILL))[ok].mean()
        ) if ok.any() else float("nan")
    return out or None


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """One column as floats, or a column of NaN if the table has not got it."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _extensive_wall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in the extensive columns a single-section result does not carry.

    ``collagen_area_um2_per_mm_length`` is reported per millimetre because that
    is the comparable form; pooling needs it back as a total.
    """
    df = df.copy()
    if "collagen_area_um2_total" not in df.columns:
        per_mm = _numeric(df, "collagen_area_um2_per_mm_length")
        length_mm = _numeric(df, "wall_length_um") / 1000.0
        df["collagen_area_um2_total"] = per_mm * length_mm
    return df


def _extensive_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    The same, for the area analysis: areas are reported in mm^2 because that
    is the readable unit, and pooling wants them back in um^2 alongside the
    stain integrals so everything sums in the same units.
    """
    df = df.copy()
    for total, mm2 in (("tissue_area_um2", "tissue_area_mm2"),
                       ("imaged_area_um2", "imaged_area_mm2"),
                       ("collagen_area_um2_total", "collagen_area_mm2")):
        if total not in df.columns and mm2 in df.columns:
            df[total] = _numeric(df, mm2) * 1e6
    return df


def add_extensive(df: pd.DataFrame, organ: str | Scheme | None = None) -> pd.DataFrame:
    return scheme_for(organ).fill_extensive(df)


def _derive_wall(row: dict[str, float]) -> dict[str, float]:
    """The intensive numbers, rebuilt from summed integrals."""
    blue = row.get("collagen_od_um2_total", float("nan"))
    red = row.get("muscle_od_um2_total", float("nan"))
    length = row.get("wall_length_um", float("nan"))
    area = row.get("media_area_um2", float("nan"))
    blue_area = row.get("collagen_area_um2_total", float("nan"))
    return {
        "collagen_to_muscle_ratio": _safe(blue, red),
        "collagen_fraction_of_stain": _safe(blue, blue + red),
        "mean_media_thickness_um": _safe(area, length),
        "collagen_od_um2_per_mm_length": _safe(blue, length / 1000.0),
        "muscle_od_um2_per_mm_length": _safe(red, length / 1000.0),
        "collagen_area_fraction_media": _safe(blue_area, area),
    }


def _derive_area(row: dict[str, float]) -> dict[str, float]:
    """
    The same rebuild for the area analysis.

    Every denominator here is *tissue* area, never image area.  These sections
    are half empty slide and how much varies with where the field was taken,
    so per image area would largely measure how much blank slide the operator
    included.  ``tissue_fraction_of_image`` is kept as a readout precisely so
    that choice stays visible rather than implicit.
    """
    blue = row.get("collagen_od_um2_total", float("nan"))
    red = row.get("muscle_od_um2_total", float("nan"))
    tissue = row.get("tissue_area_um2", float("nan"))
    imaged = row.get("imaged_area_um2", float("nan"))
    blue_area = row.get("collagen_area_um2_total", float("nan"))
    return {
        "collagen_to_muscle_ratio": _safe(blue, red),
        "collagen_fraction_of_stain": _safe(blue, blue + red),
        "collagen_area_fraction_tissue": _safe(blue_area, tissue),
        "collagen_od_um2_per_mm2_tissue": _safe(blue, tissue / 1e6),
        "muscle_od_um2_per_mm2_tissue": _safe(red, tissue / 1e6),
        "tissue_fraction_of_image": _safe(tissue, imaged),
    }


def derive_readouts(row: dict[str, float],
                    organ: str | Scheme | None = None) -> dict[str, float]:
    return scheme_for(organ).derive(row)


REQUIRED = list(WALL_SCHEME.required)


def check_columns(df: pd.DataFrame, organ: str | Scheme | None = None) -> None:
    """Raise if the table cannot be pooled, naming exactly what is missing."""
    missing = [c for c in scheme_for(organ).required
               if c not in df.columns or not _numeric(df, c).notna().any()]
    if missing:
        raise ValueError(
            "cannot pool: no usable " + ", ".join(missing) +
            ". These are the integrals pooling adds up; a batch table that keeps "
            "only a subset of totals has to keep these too.")


def add_readouts(df: pd.DataFrame, organ: str | Scheme | None = None) -> pd.DataFrame:
    """
    Give a section table the readouts that otherwise only appear after pooling.

    A single section already carries most of them, but not all -- muscle per
    unit length, for one, is only assembled on the way up.  Without this,
    switching a plot to the per-section view silently drops a panel, which is
    the worst way for a table to be missing a column.
    """
    sch = scheme_for(organ)
    df = sch.fill_extensive(df)
    for i, row in enumerate(df.to_dict(orient="records")):
        derived = sch.derive(row)
        for k, v in derived.items():
            if k not in df.columns:
                df[k] = np.nan
            if not np.isfinite(pd.to_numeric(pd.Series([row.get(k)]), errors="coerce").iloc[0]):
                df.iloc[i, df.columns.get_loc(k)] = v
    return df


def pool(df: pd.DataFrame, by: Sequence[str],
         organ: str | Scheme | None = None,
         medians: str = "weighted") -> pd.DataFrame:
    """
    Add up the integrals within each group, then rebuild the readouts.

    Not ``groupby(...).mean()``: averaging ratios across sections of unequal
    length answers a question nobody asked.  A 4 mm ring and a 0.4 mm off-cut
    are one wall measured twice, and the wall's collagen fraction is all its
    collagen over all its muscle.

    *medians* says what to do with the scheme's ``weighted`` columns -- the
    per-section medians and spreads, which cannot be summed:

    ``"weighted"``
        Average them over the group, each section weighted by the wall behind
        it, so the 0.4 mm off-cut counts a tenth of the 4 mm ring rather than
        equally.  Exact for the two columns that are themselves fractions of
        wall length and an approximation for the rest; ``SECTION_MEDIANS``
        says which, and why.  Needs nothing but the table.
    ``"exact"``
        Rebuild them from every counted position of every section in the
        group, one median over the lot -- the summing argument one level
        down.  Needs each row to say where its along-wall table is, in
        ``profile_csv``, and falls back to ``"weighted"`` for any group whose
        sections cannot all be read.
    ``"off"``
        Leave them out of the pooled table entirely, rather than have an
        approximate number read as an exact one.

    Either way the pooled row carries ``medians_from``, so which of the two
    actually happened is in the table and not only in the call.
    """
    sch = scheme_for(organ)
    df = sch.fill_extensive(df)
    keys = [k for k in by if k in df.columns]
    if not keys:
        raise ValueError(f"none of {list(by)} are columns of this table")
    work = df.copy()
    for k in keys:
        work[k] = work[k].fillna("").astype(str).replace("", "unknown")

    rows: list[dict[str, Any]] = []
    for values, part in work.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row: dict[str, Any] = dict(zip(keys, values))
        for col in sch.extensive:
            # A quantity nobody measured stays NaN; summing it as zero would
            # turn a missing column into a confident zero downstream.
            vals = _numeric(part, col)
            row[col] = float(vals.sum()) if vals.notna().any() else float("nan")
        row.update(sch.derive(row))
        if sch.weighted and medians != "off":
            got = _exact_medians(part) if medians == "exact" else None
            if got is not None:
                row.update(got)
                row["medians_from"] = "positions"
            else:
                weight = _numeric(part, sch.weight)
                for col in sch.weighted:
                    w, cov = weight, _COVERED_BY.get(col)
                    # Fall back to plain length when the coverage column is not
                    # in the table: better a whole-length weight than no number.
                    if cov and cov in part.columns:
                        w = weight * _numeric(part, cov)
                    row[col] = _weighted_mean(_numeric(part, col), w)
                row["medians_from"] = "section medians, by wall length"
        if "wall_length_um" in row:
            row["wall_length_mm"] = row["wall_length_um"] / 1000.0
        if "tissue_area_um2" in row:
            row["tissue_area_mm2"] = row["tissue_area_um2"] / 1e6
        # A section can contribute more than one row: an arch section often
        # catches the vessel twice, and both are measured.  Those are two
        # vessels of one section, so the section count has to come from the
        # section names and not from the number of rows, or a mouse looks
        # better replicated than it is.
        row["n_sections"] = int(part["name"].astype(str).nunique()
                                if "name" in part.columns else len(part))
        row["n_vessels"] = int(len(part))
        row["n_fields"] = int(len(part))
        for extra, col in (("n_slides", "slide"), ("n_mice", "mouse"),
                           ("n_segments_covered", "segment")):
            if col in part.columns and col not in keys:
                row[extra] = int(part[col].astype(str).replace("", "unknown").nunique())
        row["sections"] = ", ".join(sorted(part["name"].astype(str))) if "name" in part else ""
        rows.append(row)
    return pd.DataFrame(rows)


def per_mouse_segment(sections: pd.DataFrame,
                      organ: str | Scheme | None = None,
                      medians: str = "weighted") -> pd.DataFrame:
    """One row per mouse per region: the slices of that region pooled."""
    return pool(sections, ["genotype", "mouse", "segment"], organ, medians)


def per_mouse(sections: pd.DataFrame,
              organ: str | Scheme | None = None,
              medians: str = "weighted") -> pd.DataFrame:
    """One row per mouse, every region pooled -- the whole-organ summary."""
    sch = scheme_for(organ)
    out = pool(sections, ["genotype", "mouse"], sch, medians)
    out.insert(2, "segment", sch.pooled_label)
    return out


# =============================================================================
# 3.  Comparison
# =============================================================================


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t and its two-sided p, or NaN when there is nothing to test."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats
    except ImportError:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def per_genotype(mouse_rows: pd.DataFrame, readouts: Sequence[str] = READOUTS) -> pd.DataFrame:
    """
    Mean +/- SD across **mice**, one row per genotype per segment per readout.

    n is the number of animals, and ``n_sections`` says how many sections went
    into them.  The two being very different is the whole point of the table:
    twenty sections from three mice is an n of 3.
    """
    rows: list[dict[str, Any]] = []
    keys = [k for k in ("segment", "genotype") if k in mouse_rows.columns]
    for values, part in mouse_rows.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        base = dict(zip(keys, values))
        for readout in readouts:
            if readout not in part.columns:
                continue
            v = pd.to_numeric(part[readout], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rows.append({
                **base,
                "readout": readout,
                "n_mice": int(len(v)),
                "n_sections": int(_numeric(part, "n_sections").fillna(0).sum()),
                "mean": float(np.mean(v)) if len(v) else float("nan"),
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                "sem": float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"),
                "min": float(np.min(v)) if len(v) else float("nan"),
                "max": float(np.max(v)) if len(v) else float("nan"),
            })
    return pd.DataFrame(rows, columns=list(keys) + [
        "readout", "n_mice", "n_sections", "mean", "sd", "sem", "min", "max"])


def compare_genotypes(mouse_rows: pd.DataFrame,
                      readouts: Sequence[str] = READOUTS) -> pd.DataFrame:
    """
    Every pair of genotypes, per segment, per readout.

    Welch's t-test, which does not assume the two groups have the same
    variance -- with three or four mice a side you cannot check that
    assumption, so not making it is free.  ``p`` is NaN below two mice per
    group, and that is the honest answer: a difference between two single
    animals has no p-value, only a direction.
    """
    rows: list[dict[str, Any]] = []
    seg_col = "segment" if "segment" in mouse_rows.columns else None
    segments = sorted(mouse_rows[seg_col].astype(str).unique()) if seg_col else [""]
    for seg in segments:
        part = mouse_rows[mouse_rows[seg_col].astype(str) == seg] if seg_col else mouse_rows
        genos = sorted(part["genotype"].astype(str).replace("", "unknown").unique())
        for i, g1 in enumerate(genos):
            for g2 in genos[i + 1:]:
                a_rows = part[part["genotype"].astype(str) == g1]
                b_rows = part[part["genotype"].astype(str) == g2]
                for readout in readouts:
                    if readout not in part.columns:
                        continue
                    a = pd.to_numeric(a_rows[readout], errors="coerce").to_numpy(dtype=float)
                    b = pd.to_numeric(b_rows[readout], errors="coerce").to_numpy(dtype=float)
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    t, p = _welch(a, b)
                    mean_a = float(np.mean(a)) if len(a) else float("nan")
                    mean_b = float(np.mean(b)) if len(b) else float("nan")
                    rows.append({
                        **({seg_col: seg} if seg_col else {}),
                        "readout": readout,
                        "genotype_a": g1, "genotype_b": g2,
                        "n_a": int(len(a)), "n_b": int(len(b)),
                        "mean_a": mean_a, "mean_b": mean_b,
                        "difference": mean_b - mean_a,
                        "ratio": _safe(mean_b, mean_a),
                        "t": t, "p": p,
                    })
    # An empty table still needs its header: a nought-byte CSV is not a result
    # anyone can read back, and "one genotype, nothing to compare" is a real
    # and common state.
    columns = ([seg_col] if seg_col else []) + [
        "readout", "genotype_a", "genotype_b", "n_a", "n_b",
        "mean_a", "mean_b", "difference", "ratio", "t", "p"]
    return pd.DataFrame(rows, columns=columns)


def summarise(sections: pd.DataFrame,
              organ: str | Scheme | None = None,
              medians: str = "weighted") -> dict[str, pd.DataFrame]:
    """
    The whole ladder in one call: sections in, four tables out.

    ``by_mouse_segment`` is the one to plot; ``by_mouse`` is the whole-organ
    summary; ``by_genotype`` and ``comparison`` are the group statistics, with
    the mouse as the unit of analysis throughout.  *organ* picks which
    integrals are being pooled -- a wall measured along its length, or tissue
    measured over its area -- and nothing else about the ladder changes.
    """
    sch = scheme_for(organ)
    # Derive the integrals *before* checking for them.  Some are reported in
    # the readable unit and only exist in summable form once converted, and a
    # table rejected for missing a column it was about to be given is a
    # confusing way to fail.
    sections = sch.fill_extensive(sections)
    check_columns(sections, sch)
    sections = add_readouts(sections, sch)
    mouse_segment = per_mouse_segment(sections, sch, medians)
    mouse_all = per_mouse(sections, sch, medians)
    stacked = pd.concat([mouse_segment, mouse_all], ignore_index=True, sort=False)
    # The averaged columns join the group statistics when they got this far.
    # A batch run before they existed simply has not got them, and both tables
    # below skip a readout their table is missing.
    readouts = list(sch.readouts) + [c for c in sch.weighted if c in stacked.columns]
    return {
        "sections": sections,
        "by_mouse_segment": mouse_segment,
        "by_mouse": mouse_all,
        "by_genotype": per_genotype(stacked, readouts),
        "comparison": compare_genotypes(stacked, readouts),
    }


#: What a curve through the wall is held constant on before genotypes are
#: compared.  Age is in it because baseline wall composition moves with age
#: far more than most genotypes move it within one age, so a group that mixes
#: a 12-week and a 32-week animal has a spread that is mostly calendar.
DEPTH_KEYS = ("segment", "age", "genotype")


def average_depth_profile(depth: pd.DataFrame,
                          keys: Sequence[str] | None = None) -> pd.DataFrame:
    """
    The stain curves through the wall, averaged up the same ladder as
    everything else: sections into a mouse, mice into a genotype -- held
    apart by region and by age.

    Averaging the sections directly would make n the number of sections, and a
    mouse that happened to yield six of them would count six times over -- the
    same mistake ``per_genotype`` exists to avoid, only harder to see because
    the output is a curve rather than a number.  So each mouse's sections are
    averaged first, at every depth, and the spread reported across genotypes is
    the spread **between animals**: ``muscle_sd`` here is biological
    variability, not the variation along one wall that the per-section file
    carries under the same name.

    That first average is weighted by ``n_rows`` -- the arc positions of the
    section that reached this relative depth -- so a 0.4 mm off-cut does not
    pull a mouse's curve as hard as a 4 mm ring.  It is the same argument as
    pooling by sums, applied at every depth separately, because how much wall
    a section contributes is itself a function of depth: a section whose outer
    edge is ragged has fewer positions reaching 1.0 than reaching 0.5.

    The depth grid is a fixed ``linspace`` from ``WallParams``, so sections
    share it exactly and the rows line up without interpolation; grouping on
    the value rather than the row index means a batch run with different
    parameters degrades into a sparser grid instead of silently mismatching.
    """
    if depth is None or not len(depth):
        return pd.DataFrame()
    keys = [k for k in (list(keys) if keys else list(DEPTH_KEYS))
            if k in depth.columns]
    if "mouse" not in depth.columns or not keys:
        return pd.DataFrame()
    depth = depth.copy()
    for k in keys:
        depth[k] = depth[k].fillna("").astype(str).replace("", "unknown")

    out: list[pd.DataFrame] = []
    for measure in ("muscle_mean", "collagen_mean"):
        if measure not in depth.columns:
            continue
        # Sections (and both vessels of one) into the mouse that produced them.
        v = _numeric(depth, measure)
        w = (_numeric(depth, "n_rows") if "n_rows" in depth.columns
             else pd.Series(1.0, index=depth.index))
        # A depth a section never reached contributes neither a value nor a
        # weight; letting its weight stand would dilute the mouse's curve
        # towards nothing exactly where the section ran out of wall.
        w = w.where(np.isfinite(v) & np.isfinite(w) & (w > 0), 0.0)
        work = depth[keys + ["mouse", "depth_relative"]].copy()
        work["_v"], work["_w"] = v * w, w
        summed = work.groupby(keys + ["mouse", "depth_relative"],
                              dropna=False, observed=True)[["_v", "_w"]].sum()
        per_mouse_curve = (summed["_v"] / summed["_w"].where(summed["_w"] > 0)
                           ).rename(measure).reset_index()
        g = per_mouse_curve.groupby(keys + ["depth_relative"],
                                    dropna=False, observed=True)[measure]
        stat = g.agg(["mean", "std", "count"]).reset_index()
        stem = measure.replace("_mean", "")
        stat = stat.rename(columns={"mean": f"{stem}_mean",
                                    "std": f"{stem}_sd",
                                    "count": "n_mice"})
        out.append(stat.set_index(keys + ["depth_relative"]))
    if not out:
        return pd.DataFrame()

    merged = pd.concat(out, axis=1)
    # n_mice is the same count for both measures; keep one copy.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    cols = [c for c in ("muscle_mean", "muscle_sd", "collagen_mean", "collagen_sd",
                        "n_mice") if c in merged.columns]
    return merged[cols].reset_index().sort_values(keys + ["depth_relative"],
                                                  ignore_index=True)


def write_summary(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, str]:
    """Write each table beside the batch output and hand back the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        p = out_dir / f"cohort_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)
    return paths


# =============================================================================
# 4.  Plots
# =============================================================================

# What each readout is called on an axis, and how to print it.
# Two short lines beat one long one: a y-label wide enough to reach the panel
# beside it is how a grid of plots turns into a puzzle.
LABELS: dict[str, tuple[str, float]] = {
    "collagen_to_muscle_ratio":     ("collagen ÷ muscle", 1.0),
    "collagen_fraction_of_stain":   ("collagen fraction\nof stain (%)", 100.0),
    "mean_media_thickness_um":      ("media thickness\n(µm)", 1.0),
    "collagen_od_um2_per_mm_length": ("collagen per mm\n(OD·µm²/mm)", 1.0),
    "muscle_od_um2_per_mm_length":  ("muscle per mm\n(OD·µm²/mm)", 1.0),
    "collagen_area_fraction_media": ("collagen > muscle\n(% of media)", 100.0),
    "wall_length_mm":               ("wall measured\n(mm)", 1.0),
    # -- the media-quality block, averaged up rather than summed.
    "muscle_variation":             ("muscle unevenness\n(CV across wall)", 1.0),
    "lamellar_contrast":            ("lamellar contrast\n(CV)", 1.0),
    "lamellar_spacing_um":          ("lamellar spacing\n(µm)", 1.0),
    "lamellar_disorder":            ("lamellar disorder\n(CV of gaps)", 1.0),
    "lamellar_coverage":            ("wall with\nlamellae (%)", 100.0),
    "section_edge_width_um":        ("edge width\n(µm)", 1.0),
    "median_muscle_beyond_edge_um": ("muscle past the\nmedia edge (µm)", 1.0),
    "median_muscle_beyond_edge_fill": ("fill past the\nmedia edge (%)", 100.0),
    "fragmented_length_fraction":   ("fragmented outer\nmedia (% of wall)", 100.0),
    # -- the area analysis: every denominator is tissue area, and the labels
    #    say so, because "% of tissue" and "% of image" are different numbers
    #    and a figure that does not distinguish them is misread.
    "collagen_area_fraction_tissue": ("collagen > muscle\n(% of tissue)", 100.0),
    "collagen_od_um2_per_mm2_tissue": ("collagen per mm²\n(OD·µm²/mm²)", 1.0),
    "muscle_od_um2_per_mm2_tissue": ("muscle per mm²\n(OD·µm²/mm²)", 1.0),
    "tissue_fraction_of_image":     ("tissue\n(% of imaged)", 100.0),
    "tissue_area_mm2":              ("tissue measured\n(mm²)", 1.0),
}

# The colours, the style, the point layout and the significance brackets are
# all labkit's now, so a cohort figure and a vesicle figure are the same
# figure with different numbers in it.
from labkit import plots as LKP
from labkit import stats as LKS
from labkit.plots import PlotSpec

#: Kept as a name because callers outside this module import it.
GENOTYPE_COLOURS = LKP.PALETTES["red-grey"]

#: Which pooled table a "a dot is ..." choice draws from.
UNIT_TABLE = {"mouse_segment": "by_mouse_segment",
              "mouse": "by_mouse",
              "section": "sections"}


def genotypes_in(tables: dict[str, pd.DataFrame], unit: str = "mouse_segment"
                 ) -> list[str]:
    """The genotype labels a figure of this unit will draw, in drawing order.

    The web panel needs them before the figure exists, to offer a colour box
    per genotype; the figure needs them to hand the same genotype the same
    colour in every panel.  One function, so the boxes cannot end up naming
    something the figure does not draw.
    """
    df = tables.get(UNIT_TABLE.get(unit, "by_mouse_segment"))
    if df is None or "genotype" not in getattr(df, "columns", ()):
        return []
    labels = df["genotype"].fillna("").astype(str).replace("", "unknown")
    return sorted(set(labels))


def _panel_comparisons(comp: pd.DataFrame, segment: str, readout: str,
                       spec: PlotSpec) -> list[dict]:
    """The brackets for one panel, from the batch's own comparison table.

    Read out of the table rather than recomputed so the figure and the CSV
    beside it cannot disagree.  ``compare_genotypes`` already ran a Welch test
    per genotype pair per segment per readout; the correction across the panel
    happens here, because the family is what the panel shows.
    """
    if comp is None or not len(comp) or "readout" not in comp.columns:
        return []
    rows = comp[comp["readout"] == readout]
    if "segment" in rows.columns and segment:
        rows = rows[rows["segment"].astype(str) == str(segment)]
    rows = rows[np.isfinite(pd.to_numeric(rows["p"], errors="coerce"))]
    if not len(rows):
        return []

    p_raw = pd.to_numeric(rows["p"], errors="coerce").to_numpy(float)
    if spec.p_correction == "none" or len(p_raw) == 1:
        p_use = p_raw
    else:
        from statsmodels.stats.multitest import multipletests
        p_use = multipletests(p_raw, alpha=spec.p_alpha, method=spec.p_correction)[1]

    out = []
    for (_, row), p in zip(rows.iterrows(), p_use):
        significant = bool(p < spec.p_alpha)
        if spec.p_only_significant and not significant:
            continue
        out.append({
            "a": str(row["genotype_a"]), "b": str(row["genotype_b"]),
            "p": float(p), "significant": significant,
            "text": LKS.format_p(float(p), style=spec.p_style,
                                 show_ns=not spec.p_only_significant),
        })
    return out


def _fit_group_labels(fig) -> None:
    """
    Turn the group labels on their side when they would otherwise collide.

    Genotype names are as long as the breeding scheme makes them -- "L8D-E-KO"
    against "L8D-KO" -- and a panel narrow enough to sit two-up on a page is
    not wide enough to put both under the axis flat.  Left alone they overlap
    into something unreadable, which is the wrong way for a size control to
    fail: asking for a smaller panel should make the panel smaller, not make
    the figure wrong.

    The overlap is *measured* rather than predicted.  A rule of thumb about
    characters and point sizes has to know the font's metrics, the width the
    y-label and tick labels took off the panel, and what the layout engine did
    with the rest -- guess any of them and the labels either collide anyway or
    tilt when they had no need to.  Laying the figure out once and asking the
    labels where they actually are costs one silent draw and is simply right.

    One tilt is not always enough: at a 25 mm panel "L8A-Sox10-Cre" and
    "L8D-E-KO" still sit on top of each other at 30 degrees.  So the
    measurement is the loop condition rather than a one-off -- each rung below
    is applied, drawn, and measured again, and the first one that clears is
    the one kept.  Upright is what always clears it, because a turned label is
    only as wide as it is tall however long the name; the shrinking font after
    that is for a panel too narrow for even one upright line per group.

    The rung is the figure's, not each panel's: the same genotypes are under
    every panel of a cohort figure, and one of them tilted further than its
    neighbours reads as if it meant something.
    """
    # Cost in readability, cheapest first.
    LADDER = ((30, 1.0), (45, 1.0), (60, 1.0), (90, 1.0), (90, 0.8), (90, 0.65))

    def drawn() -> bool:
        try:
            fig.draw_without_rendering()
        except Exception:                 # older matplotlib, or no renderer
            try:
                fig.canvas.draw()
            except Exception:
                return False
        return True

    def collides(ax) -> bool:
        ticks = [t for t in ax.get_xticklabels() if t.get_text()]
        if len(ticks) < 2:
            return False
        try:
            boxes = sorted((t.get_window_extent() for t in ticks), key=lambda b: b.x0)
        except Exception:
            return False
        # One pixel of air between neighbours; touching is already too close.
        return any(a.x1 > b.x0 - 1.0 for a, b in zip(boxes, boxes[1:]))

    if not drawn():
        return
    panels = {}
    for ax in fig.axes:
        # Every panel with labels, not only the ones that collide: a panel
        # whose segment has a single genotype cannot collide, and left behind
        # flat while its neighbours turned it reads as a statement about that
        # segment.  ``collides`` still ignores it, so it follows and never
        # drives.
        if ax.axison and any(t.get_text() for t in ax.get_xticklabels()):
            panels[ax] = ([t.get_text() for t in ax.get_xticklabels()],
                          ax.get_xticklabels()[0].get_fontsize())
    for rotation, shrink in LADDER:
        if not any(collides(ax) for ax in panels):
            return
        for ax, (texts, size) in panels.items():
            if rotation > 45:
                # Past 45 degrees the "(n=6)" second line turns with the label
                # and runs onto the next group's, so it goes back up on the
                # first -- the trade labkit's own style_axes makes.
                try:
                    ax.set_xticklabels([t.replace("\n(n=", " (n=") for t in texts])
                except Exception:
                    pass
            for tick in ax.get_xticklabels():
                tick.set(rotation=rotation, ha="right", rotation_mode="anchor",
                         fontsize=size * shrink)
        if not drawn():
            return


def plot_cohort(
    tables: dict[str, pd.DataFrame],
    readouts: Sequence[str] | None = None,
    unit: str = "mouse_segment",
    spec: PlotSpec | dict | None = None,
    title: str = "",
    panel_mm: float = 0.0,
    panel_h_mm: float = 0.0,
    ncol: int = 0,
    organ: str = "",
    **overrides,
):
    """
    One panel per readout: the group summary, with a dot on every animal.

    The dots are the point of the figure.  With three or four mice a side, a
    bar and an error bar hide whether the difference is four animals agreeing
    or one animal carrying the group, and that is exactly what a reader needs
    to see.  Each dot is one *mouse* (or one section, if you ask for that unit
    -- in which case the figure says so, because sections within an animal are
    repeated measures and a bar over them is not an n).

    ``unit`` is ``'mouse_segment'`` (a dot per mouse, panels split by segment),
    ``'mouse'`` (a dot per mouse, all segments pooled) or ``'section'``.
    Everything else -- the shape, the error bars, the point layout, the
    colours, and whether and how p-values are printed -- comes from ``spec``,
    which is the same :class:`labkit.plots.PlotSpec` the other tools take, so
    the controls in the side panel are the controls in the other side panels.

    The grid is the exception, and has its own three arguments because this
    figure is the one that grows: a cohort figure is one panel per readout per
    region, so it can be three panels or thirty, and a single figure width
    then means a completely different panel size from one run to the next.
    ``panel_mm`` and ``panel_h_mm`` size *one panel's plotting area* and let
    the figure be whatever that makes it, which is what you want when the
    panels have to match the ones beside them on the page; ``ncol`` sets how
    many go across.  The axis label, the ticks and the panel title are added
    outside that number, so a panel is the same size whether or not the
    readout above it has a long name (see :func:`labkit.plots.fit_axes`).
    Either falls back to ``spec.axes_width_mm``/``axes_height_mm`` -- the same
    control, coming from the shared plot panel -- and then to the old
    behaviour: the width from ``spec``, split between up to three columns,
    with panels slightly wider than they are tall.
    """
    import matplotlib.pyplot as plt

    if isinstance(spec, PlotSpec):
        pass
    elif spec:
        spec = PlotSpec.from_dict(spec)
    else:
        # A cohort figure is one panel per readout per segment, so it is wide.
        # labkit's 85 mm default is a single journal column, which is the right
        # default for one panel and far too small for six.
        spec = PlotSpec(width_mm=180.0, font_pt=9.0, error="sd")
    for key, value in overrides.items():          # legacy keyword call sites
        if hasattr(spec, key) and value is not None:
            setattr(spec, key, value)

    source = UNIT_TABLE.get(unit, "by_mouse_segment")
    df = tables.get(source)
    if df is None or not len(df):
        raise ValueError(f"no rows in the {source} table to plot")
    df = df.copy()
    # A table read back from CSV has not been through summarise(), so fill in
    # any readout that is only assembled on the way up.  Harmless for the
    # pooled tables, which already carry all of them.
    try:
        df = add_readouts(df)
    except Exception:
        pass
    for col in ("genotype", "segment"):
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("").astype(str).replace("", "unknown")

    wanted = list(readouts or READOUTS)
    readouts = [r for r in wanted if r in df.columns]
    dropped = [r for r in wanted if r not in readouts]
    if not readouts:
        raise ValueError("none of the requested readouts are in the table: "
                         + ", ".join(wanted))

    segments = sorted(df["segment"].unique())
    genotypes = sorted(df["genotype"].unique())
    from fibrosis import counterstain_name
    counterstain = counterstain_name(organ) if organ else "muscle"
    # Fixed to the genotype, not to its position in the panel: a panel that is
    # missing a genotype used to shift every colour after it, so the same
    # animal group was grey in one panel of a figure and red in the next.
    palette = dict(zip(genotypes, LKP.colours_for(spec, genotypes)))
    comp = tables.get("comparison")

    # One panel per readout *per segment*: a shared x axis of segments with
    # grouped bars cannot carry a significance bracket per segment without the
    # brackets overlapping, and the brackets are the reason the figure exists.
    panels = [(r, s) for r in readouts for s in segments] if len(segments) > 1 \
        else [(r, segments[0] if segments else "") for r in readouts]
    ncol = max(1, int(ncol)) if ncol else min(3, len(panels))
    ncol = min(ncol, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    # One panel's size, then the figure, rather than the other way round.
    # This is only the starting canvas; fit_axes below settles it on the real
    # plotting area once the labels that eat into it are on the figure.
    want_w = float(panel_mm or spec.axes_width_mm or 0.0)
    want_h = float(panel_h_mm or spec.axes_height_mm or 0.0)
    panel_w = max(spec.width_mm, 40.0) * LKP.MM / ncol
    panel_h = panel_w * 0.92

    with LKP.publication_style(font_pt=spec.font_pt):
        fig, axes = plt.subplots(
            nrow, ncol,
            figsize=(panel_w * ncol, panel_h * nrow),
            squeeze=False, layout="constrained")

        for k, (readout, segment) in enumerate(panels):
            ax = axes[k // ncol][k % ncol]
            label, scale = LABELS.get(readout, (readout.replace("_", " "), 1.0))
            # The column names are the aorta's, where the red channel really is
            # muscle.  In a kidney it is tubular epithelium, and an axis
            # labelled "muscle" over a kidney is a wrong statement printed on a
            # figure -- worse than a clumsy one, because it will be believed.
            if counterstain and counterstain != "muscle":
                label = label.replace("muscle", counterstain)
            here = df[df.segment == segment] if segment else df
            groups = {}
            for genotype in genotypes:
                values = pd.to_numeric(
                    here[here.genotype == genotype][readout],
                    errors="coerce").to_numpy(dtype=float) * scale
                values = values[np.isfinite(values)]
                if values.size:
                    groups[genotype] = values
            if not groups:
                ax.axis("off")
                continue

            _, positions = LKP.group_plot(
                groups, kind=spec.kind, centre=spec.centre, error=spec.error,
                points=spec.points or False,
                palette=[palette[g] for g in groups], ax=ax,
                ylabel=label, show_n=spec.show_n, log_y=spec.log_y,
                title=segment if len(segments) > 1 else "",
            )
            if spec.show_p:
                LKP.annotate_p(
                    ax, _panel_comparisons(comp, segment, readout, spec), positions,
                    style=spec.p_style, alpha=spec.p_alpha, align=spec.p_align,
                    only_significant=spec.p_only_significant)


        for k in range(len(panels), nrow * ncol):     # unused cells in the last row
            axes[k // ncol][k % ncol].axis("off")

        # Say what a dot is, every time, on the figure itself.
        dot = {"mouse_segment": "one dot = one mouse",
               "mouse": "one dot = one mouse, all segments pooled",
               "section": "one dot = one SECTION \u2014 sections within a mouse are "
                          "repeated measures, not independent samples"}[unit]
        centre = {"mean": "mean", "median": "median"}[spec.centre]
        err = {"sd": f"{centre} \u00b1 SD", "sem": f"{centre} \u00b1 SEM",
               "ci": f"{centre}, 95% CI", "none": centre}[spec.error]
        fig.suptitle(title or spec.title or "Cohort summary")
        foot = f"{dot} \u00b7 {err}"
        if spec.show_p and comp is not None and len(comp):
            correction = {"fdr_bh": "FDR-corrected", "holm": "Holm-corrected",
                          "bonferroni": "Bonferroni-corrected",
                          "none": "uncorrected"}[spec.p_correction]
            foot += f" \u00b7 Welch t, {correction} within each panel"
        if dropped:
            foot += "  \u00b7  not in this table: " + ", ".join(dropped)
        fig.supxlabel(
            foot,
            color=LKP.theme.PRINT["warn"] if (unit == "section" or dropped)
            else LKP.theme.PRINT["muted"])
    # Last, once everything that affects the layout is on the figure, and
    # outside the style block on purpose.  Size, rotate, size: the labels are
    # only asked whether they collide once the panel is the width it is going
    # to be, and a label that then turns on its side takes height the canvas
    # has to give back.  All three out here because constrained layout settles
    # the axes box at draw time under the rcParams in force *then* -- so a
    # figure fitted inside the block and saved outside it is fitted to a
    # layout that is not the one being saved, and the collision the middle
    # step measures is not the one the reader gets either.
    LKP.fit_axes(fig, want_w, want_h)
    _fit_group_labels(fig)
    LKP.fit_axes(fig, want_w, want_h)
    return fig


#: Muscle red and collagen blue, the colours the section figure draws them in
#: (``wall_analysis.plot_wall_report``), so a cohort curve and the section it
#: came from are read the same way round.
STAIN_COLOURS = {"muscle_mean": "#d1382f", "collagen_mean": "#2c7fb8"}


def plot_depth_profiles(avg: pd.DataFrame, *, spec: PlotSpec | None = None,
                        title: str = "", counterstain: str = "muscle",
                        panel_mm: float = 0.0, panel_h_mm: float = 0.0,
                        stains_together: bool = False):
    """
    The stain curves through the wall: one row per region and age, genotypes
    overlaid, the counterstain and the collagen side by side.

    Two panels rather than one, because a single axis carrying both stains for
    every genotype is four curves in two different meanings of colour and
    nobody reads it.  x is relative depth -- 0 the lumen, 1 the
    media/adventitia junction -- so "a third of the way through the media" is
    the same place in a thick wall and a thin one, which is the whole reason
    this view exists.  The shaded band is +/- SD **between animals** at each
    depth, so it is biological spread and not the variation along one wall.

    ``stains_together`` turns the grid the other way up: a panel per condition
    with both stains in it, coloured as the section figure colours them.  That
    is the figure for "where in this wall is the collagen relative to the
    muscle", which is a question inside one genotype -- colour still means one
    thing per panel, which was the objection to overlaying all four curves.
    The default grid stays the one for comparing genotypes.

    *avg* is what ``average_depth_profile`` returns.
    """
    import matplotlib.pyplot as plt

    if avg is None or not len(avg):
        raise ValueError("no averaged depth profile to plot -- run a batch "
                         "with the wall analysis switched on")
    avg = avg.copy()
    for k in DEPTH_KEYS:
        if k in avg.columns:
            avg[k] = avg[k].fillna("").astype(str).replace("", "unknown")
    spec = spec or PlotSpec()
    facets = [k for k in ("segment", "age") if k in avg.columns]
    measures = [(m, m.replace("_mean", "_sd"), lbl) for m, lbl in
                (("muscle_mean", counterstain or "muscle"),
                 ("collagen_mean", "collagen")) if m in avg.columns]
    if not measures:
        raise ValueError("the depth table carries neither muscle_mean nor "
                         "collagen_mean -- it is not an averaged depth profile")
    groups = (list(avg.groupby(facets, dropna=False, sort=True)) if facets
              else [((), avg)])
    genotypes = sorted(avg["genotype"].unique()) if "genotype" in avg.columns else [""]
    # By name, so a panel a genotype is missing from does not recolour the rest.
    palette = dict(zip(genotypes, LKP.colours_for(spec, genotypes)))

    def _where(values) -> str:
        values = values if isinstance(values, tuple) else (values,)
        return " · ".join(str(v) for v in values if str(v) != "unknown")

    def _curves(part, pairs) -> tuple[list, list]:
        """The series for one panel, and the colour of each."""
        series, colours = [], []
        for (mean_col, sd_col, label), genotype in pairs:
            here = (part[part.genotype == genotype] if "genotype" in part.columns
                    else part).sort_values("depth_relative")
            if not len(here):
                continue
            entry = {"label": label if stains_together else (str(genotype) or "all"),
                     "x": here["depth_relative"].to_numpy(dtype=float),
                     "mean": _numeric(here, mean_col).to_numpy(dtype=float)}
            if sd_col in here.columns:
                entry["sem"] = _numeric(here, sd_col).to_numpy(dtype=float)
            if "n_mice" in here.columns:
                entry["n"] = _numeric(here, "n_mice").to_numpy(dtype=float)
            series.append(entry)
            colours.append(STAIN_COLOURS[mean_col] if stains_together
                           else palette[genotype])
        return series, colours

    # One flat list of panels, laid out row-major.  Comparing genotypes, a row
    # is a region and age and the columns are the stains; comparing the stains,
    # a panel *is* a condition and they wrap -- the cross product of region,
    # age and genotype is mostly empty on a real cohort, and a grid of six
    # curves in eighteen slots is holes with a figure round them.
    panels: list[tuple[str, list, list]] = []
    if stains_together:
        keys = facets + (["genotype"] if "genotype" in avg.columns else [])
        for values, part in (avg.groupby(keys, dropna=False, sort=True) if keys
                             else [((), avg)]):
            genotype = (values if isinstance(values, tuple) else (values,))[-1] if keys else ""
            series, colours = _curves(part, [(m, genotype) for m in measures])
            if series:
                panels.append((_where(values) or "all", series, colours))
        ncol = min(3, len(panels)) or 1
    else:
        for values, part in groups:
            for measure in measures:
                series, colours = _curves(part, [(measure, g) for g in genotypes])
                panels.append((f"{_where(values)} · {measure[2]}".lstrip(" ·"),
                               series, colours))
        ncol = len(measures)
    nrow = -(-len(panels) // ncol)

    want_w = float(panel_mm or spec.axes_width_mm or 0.0)
    want_h = float(panel_h_mm or spec.axes_height_mm or 0.0)
    panel_w = max(spec.width_mm, 40.0) * LKP.MM / ncol
    panel_h = panel_w * 0.78

    with LKP.publication_style(font_pt=spec.font_pt):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                                 figsize=(panel_w * ncol, panel_h * nrow),
                                 layout="constrained")
        for i, (head, series, colours) in enumerate(panels):
            ax = axes[i // ncol][i % ncol]
            # The media itself, so which part of the curve is wall and which is
            # the lumen or the adventitia beside it can be seen rather than
            # counted off the axis.
            ax.axvspan(0.0, 1.0, color=LKP.theme.PRINT["grid"], alpha=0.16,
                       lw=0, zorder=0)
            LKP.xy_plot(series, ax=ax, palette=colours, markers=False,
                        zero_lines=False, legend=(i == 0),
                        ylabel="stain (OD)" if stains_together
                        else f"{measures[i % ncol][2]} (OD)",
                        title=head)
        # The last row can be short.  x lives on the bottom row (sharex), so
        # whichever panel is lowest in each column has to take the labels back.
        for c in range(ncol):
            lowest = max((r for r in range(nrow) if r * ncol + c < len(panels)),
                         default=None)
            for r in range(nrow):
                if r * ncol + c >= len(panels):
                    axes[r][c].set_axis_off()
            if lowest is not None:
                axes[lowest][c].tick_params(labelbottom=True)

        # One label under the whole figure rather than one per panel: the
        # sentence is wider than a panel and two of them collide.
        fig.supxlabel("relative depth through the wall  (0 = lumen, 1 = outer media)")
        if title:
            fig.suptitle(title)
    # Outside the style block on purpose.  Constrained layout settles the axes
    # box at draw time, and this figure settles on a slightly different one
    # under the publication rcParams than under the caller's -- so a panel
    # fitted inside the block and saved outside it came out 1.97 in wide when
    # 2.00 was asked for.  Fitting where the saving happens lands the number.
    LKP.fit_axes(fig, want_w, want_h)
    return fig


def save_depth_plot(avg: pd.DataFrame, out_dir: str | Path,
                    name: str = "cohort_depth", **kwargs) -> dict[str, str]:
    """The depth figure as a raster to look at and a vector to edit."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_depth_profiles(avg, **kwargs)
    png, svg = out_dir / f"{name}_plot.png", out_dir / f"{name}_plot.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def save_cohort_plot(tables: dict[str, pd.DataFrame], out_dir: str | Path,
                     name: str = "cohort", **kwargs) -> dict[str, str]:
    """The figure as a raster to look at and a vector to edit."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_cohort(tables, **kwargs)
    png, svg = out_dir / f"{name}_plot.png", out_dir / f"{name}_plot.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    # Real <text>, not matplotlib's default of every glyph traced out as its
    # own curve -- the labels can then be edited as text in a vector editor.
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}
