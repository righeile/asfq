"""
The fast paths have to give the old answers.

Pieces of the pipeline were rewritten for speed -- a bilinear sampler, square
morphology, the distance transform, the Gaussian smooth, hole filling and the
small-component filters -- and each one is only allowed to exist because it
agrees with the scipy or skimage call it replaced.  This checks that, on
shapes and masks with the awkward cases in them: even-sided structuring
elements, coordinates exactly on the last row, coordinates off the image
entirely, holes that touch the border.

Run it directly:  python test_fast_paths.py
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import wall_analysis as WA


def _mask(shape=(180, 210), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.random(shape) > 0.86
    return ndi.binary_dilation(m, np.ones((3, 3)))


def test_square_morph_matches_scipy() -> None:
    m = _mask()
    for size in (3, 4, 5, 12, 13):
        for op, ref in (("close", ndi.binary_closing), ("open", ndi.binary_opening)):
            got = WA.square_morph(m, size, op)
            want = ref(m, np.ones((size, size), bool))
            assert np.array_equal(got, want), f"{op} {size}x{size}"
        assert np.array_equal(WA.square_morph(m, size, "dilate"),
                              ndi.binary_dilation(m, np.ones((size, size), bool))), size


def test_fill_holes_matches_scipy() -> None:
    for seed in range(4):
        m = _mask(seed=seed)
        assert np.array_equal(WA.fill_holes(m), ndi.binary_fill_holes(m)), seed


def test_small_component_filters_match_skimage() -> None:
    from skimage.morphology import remove_small_objects, remove_small_holes
    big = WA._SK_MAX_SIZE
    for seed in range(4):
        m = _mask(seed=seed)
        for n in (2, 17, 500):
            want = (remove_small_objects(m, max_size=n - 1) if big
                    else remove_small_objects(m, n))
            assert np.array_equal(WA.drop_small_objects(m, n), want), ("objects", seed, n)
            want = (remove_small_holes(m, max_size=n - 1) if big
                    else remove_small_holes(m, n))
            assert np.array_equal(WA.fill_small_holes(m, n), want), ("holes", seed, n)


def test_edt_matches_scipy() -> None:
    m = _mask(seed=1)
    want = ndi.distance_transform_edt(m)
    assert np.array_equal(WA.edt(m), want.astype(np.float32))


def test_bilinear_matches_map_coordinates() -> None:
    rng = np.random.default_rng(2)
    img = rng.random((40, 53)).astype(np.float32)
    h, w = img.shape
    # Inside, outside, and sitting exactly on the last row and column.
    ys = np.array([[0.0, -1e-9, h - 1, h - 1 + 1e-6, 0.5, h / 2, -2.0, h + 3.0]])
    xs = np.array([[0.0, 0.0, w - 1, w - 1, 0.5, w / 2, 1.0, 2.0]])
    def same(got, want):
        ok = np.isfinite(want)
        assert np.array_equal(np.isfinite(got), ok), "different pixels fell off the image"
        # One float32 step apart at worst: same arithmetic, rounded once more.
        ulp = np.spacing(np.float32(1.0)) * np.maximum(np.abs(want[ok]), 1.0)
        assert np.all(np.abs(got[ok] - want[ok]) <= ulp)

    same(WA._bilinear(img, ys, xs),
         ndi.map_coordinates(img, [ys, xs], order=1, mode="constant", cval=np.nan))

    yr = rng.uniform(-3, h + 3, (300, 40))
    xr = rng.uniform(-3, w + 3, (300, 40))
    same(WA._bilinear(img, yr, xr),
         ndi.map_coordinates(img, [yr, xr], order=1, mode="constant", cval=np.nan))

    # Band coordinates arrive as float32.  The same points written as float64
    # are the same points, so the answer may not depend on which was handed
    # over -- if it does, the interpolation is being carried in single.
    y32, x32 = yr.astype(np.float32), xr.astype(np.float32)
    assert np.array_equal(WA._bilinear(img, y32, x32),
                          WA._bilinear(img, y32.astype(np.float64),
                                       x32.astype(np.float64)), equal_nan=True)

    # RGB is gathered in one pass; it must still equal channel by channel.
    rgb = rng.random((h, w, 3)).astype(np.float32)
    got = WA._bilinear(rgb, y32, x32)
    for ch in range(3):
        same(got[..., ch], ndi.map_coordinates(
            rgb[..., ch], [y32, x32], order=1, mode="constant", cval=np.nan))


def test_gaussian_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    a = (rng.random((150, 190)) * 3).astype(np.float32)
    for sigma in (1.0, 7.5, 20.6):
        got = WA.gaussian(a, sigma)
        want = ndi.gaussian_filter(a, sigma)
        # Not bit-for-bit -- OpenCV accumulates the separable pass in float32
        # -- but far below anything measured off these maps.
        assert np.abs(got - want).max() < 1e-4 * max(1.0, float(np.abs(want).max()))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
