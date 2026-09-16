import numpy as np
import pytest

import pixelperfect.sampling as sampling_module
from pixelperfect.sampling import recover_cells, improve_structure, sampling_gradient
from pixelperfect.palette import quantize_cells


def test_robust_beats_center_with_center_outlier():
    image = np.ones((8, 8, 4), np.float32)
    image[..., :3] = [.7, .15, .3]
    image[4, 4, :3] = [0, 1, 0]
    robust = recover_cells(image, [0, 8], [0, 8])
    center = recover_cells(image, [0, 8], [0, 8], "center")
    truth = image[0, 0, :3]
    assert np.linalg.norm(robust.rgba[0, 0, :3] - truth) < .01
    assert np.linalg.norm(center.rgba[0, 0, :3] - truth) > .5


@pytest.mark.parametrize("method", ["robust", "center", "median"])
def test_hidden_rgb_does_not_change_result(method):
    image = np.zeros((16, 16, 4), np.float32)
    image[3:13, 3:13] = [.3, .7, .2, 1]
    other = image.copy()
    other[other[..., 3] == 0, :3] = np.random.default_rng(4).random((156, 3))
    cuts = [0, 4, 8, 12, 16]
    a = recover_cells(image, cuts, cuts, method)
    b = recover_cells(other, cuts, cuts, method)
    np.testing.assert_array_equal(a.rgba, b.rgba)
    np.testing.assert_array_equal(a.confidence, b.confidence)
    assert np.all(a.rgba[a.rgba[..., 3] == 0, :3] == 0)


def test_fractional_cells_preserve_area_alpha():
    image = np.ones((2, 3, 4), np.float32)
    image[:, 0, 3] = 0
    cells = recover_cells(image, [0, 1.5, 3], [0, 2])
    assert cells.rgba[0, 0, 3] == pytest.approx(1 / 3)
    assert cells.rgba[0, 1, 3] == 1
    np.testing.assert_allclose(cells.rgba[0, 0, :3], 1)


def test_line_hole_and_isolated_highlight_are_preserved():
    truth = np.ones((7, 7, 4), np.float32)
    truth[..., :3] = [.2, .3, .5]
    truth[:, 2, :3] = [.9, .1, .1]
    truth[3, 4] = [0, 0, 0, 0]
    truth[1, 5, :3] = [1, 1, .7]
    image = truth.repeat(6, 0).repeat(6, 1)
    cuts = np.arange(8) * 6
    cells = improve_structure(recover_cells(image, cuts, cuts), image, cuts, cuts)
    np.testing.assert_allclose(cells.rgba, truth, atol=1e-6)
    assert cells.structure["supported_holes"] == 1
    assert cells.structure["retained_isolated_details"] >= 1
    assert cells.structure["connectivity"] == 4


def test_input_supported_minority_line_can_be_promoted():
    image = np.ones((24, 24, 4), np.float32)
    image[..., :3] = .8
    image[:, 11:13, :3] = .1
    cuts = [0, 8, 16, 24]
    cells = recover_cells(image, cuts, cuts)
    assert cells.rgba[1, 1, 0] > .7
    improved = improve_structure(cells, image, cuts, cuts)
    assert improved.rgba[1, 1, 0] < .2
    assert improved.structure["structure_changes"] >= 1


def test_coherent_minority_highlight_survives_but_single_hot_pixel_does_not():
    cuts = [0, 8, 16, 24]
    image = np.ones((24, 24, 4), np.float32)
    image[..., :3] = .15
    image[11:13, 11:13, :3] = 1
    cells = recover_cells(image, cuts, cuts)
    assert cells.rgba[1, 1, 0] < .2
    improved = improve_structure(cells, image, cuts, cuts)
    assert improved.rgba[1, 1, 0] > .9
    assert improved.structure["promoted_highlights"] == 1
    image[11:13, 11:13, :3] = .15
    image[12, 12, :3] = 1
    noisy = improve_structure(recover_cells(image, cuts, cuts), image, cuts, cuts)
    assert noisy.rgba[1, 1, 0] < .2
    assert noisy.structure["promoted_highlights"] == 0


def test_palette_is_deterministic_bounded_and_alpha_preserving():
    rng = np.random.default_rng(23)
    image = rng.random((8, 8, 4), dtype=np.float32)
    image[..., 3] = 1
    image[0, :, 3] = 0
    cuts = np.arange(9)
    cells = recover_cells(image, cuts, cuts)
    a = quantize_cells(cells, 5)
    b = quantize_cells(cells, 5)
    np.testing.assert_array_equal(a.rgba, b.rgba)
    np.testing.assert_array_equal(a.rgba[..., 3], image[..., 3])
    assert len(np.unique(a.rgba[a.rgba[..., 3] > 0, :3], axis=0)) <= 5
    assert np.all(a.rgba[0, :, :3] == 0)


def test_palette_reserves_rare_highlight_and_keeps_line_hole():
    truth = np.ones((9, 9, 4), np.float32)
    truth[..., :3] = [.1, .25, .35]
    truth[..., 0] += np.arange(9)[None, :] * .025
    truth[:, 2, :3] = [.9, .12, .1]
    truth[2, 6, :3] = [1, 1, 1]
    truth[6, 6, 3] = 0
    cuts = np.arange(10) * 4
    image = truth.repeat(4, 0).repeat(4, 1)
    cells = improve_structure(recover_cells(image, cuts, cuts), image, cuts, cuts)
    reduced = quantize_cells(cells, 4)
    np.testing.assert_array_equal(reduced.rgba[2, 6, :3], [1, 1, 1])
    assert np.all(np.linalg.norm(reduced.rgba[:, 2, :3] - reduced.rgba[:, 1, :3], axis=1) > .4)
    assert reduced.rgba[6, 6, 3] == 0
    assert len(np.unique(reduced.rgba[reduced.rgba[..., 3] > 0, :3], axis=0)) <= 4
    assert reduced.structure["palette_reserved_details"] == 1


def test_hidden_rgb_does_not_consume_palette_budget():
    image = np.zeros((8, 8, 4), np.float32)
    image[..., :3] = np.random.default_rng(93).random((8, 8, 3))
    image[3, 3] = [.8, .1, .2, 1]
    image[4, 4] = [.2, .7, .1, .5]
    cuts = np.arange(9)
    cells = recover_cells(image, cuts, cuts)
    reduced = quantize_cells(cells, 2)
    np.testing.assert_array_equal(reduced.rgba[3:5, 3:5], cells.rgba[3:5, 3:5])
    assert reduced.structure["palette_size"] == 2


def test_precomputed_gradient_is_identical():
    image = np.random.default_rng(15).random((16, 16, 4), dtype=np.float32)
    cuts = np.arange(5) * 4
    cached = recover_cells(image, cuts, cuts, gradient=sampling_gradient(image))
    direct = recover_cells(image, cuts, cuts)
    np.testing.assert_array_equal(cached.rgba, direct.rgba)
    np.testing.assert_array_equal(cached.alternatives, direct.alternatives)


@pytest.mark.parametrize("fractional", [False, True])
@pytest.mark.parametrize("transparent", [False, True])
def test_batched_simple_sampler_matches_scalar_reference(monkeypatch, fractional, transparent):
    rng = np.random.default_rng(126)
    truth = np.ones((4, 4, 4), np.float32)
    truth[..., :3] = rng.integers(0, 6, (4, 4, 3)) / 5
    if transparent:
        truth[0, :, 3] = 0
        truth[1, :, 3] = .5
    image = truth.repeat(8, 0).repeat(8, 1)
    image[3:6, 3:6, :3] = [.8, .7, .2]
    image[image[..., 3] == 0, :3] = rng.random((int(np.sum(image[..., 3] == 0)), 3))
    cuts = np.r_[0, np.arange(4, 32, 4) + (.3 if fractional else 0), 32]
    cached_gradient = sampling_gradient(image)
    batched = recover_cells(image, cuts, cuts, gradient=cached_gradient)
    monkeypatch.setattr(sampling_module, "_batched_simple_cells",
                        lambda rgba, x, y, *args: np.zeros((len(y) - 1, len(x) - 1), bool))
    scalar = recover_cells(image, cuts, cuts, gradient=cached_gradient)
    np.testing.assert_allclose(batched.rgba, scalar.rgba, atol=1e-7)
    np.testing.assert_allclose(batched.confidence, scalar.confidence, atol=1e-7)
    np.testing.assert_array_equal(np.rint(batched.rgba * 255), np.rint(scalar.rgba * 255))
    np.testing.assert_allclose(batched.alternatives, scalar.alternatives, atol=1e-7)


@pytest.mark.parametrize("colors", [0, -1, 257, 2.5, True])
def test_invalid_palette(colors):
    cells = recover_cells(np.ones((2, 2, 4), np.float32), [0, 2], [0, 2])
    with pytest.raises(ValueError):
        quantize_cells(cells, colors)


def test_invalid_lines_and_sampling():
    image = np.ones((2, 2, 4), np.float32)
    with pytest.raises(ValueError, match="cut lines"):
        recover_cells(image, [0, 1, 1, 2], [0, 2])
    with pytest.raises(ValueError, match="sampling"):
        recover_cells(image, [0, 2], [0, 2], "unknown")


def test_stratification_missed_visible_island_uses_visible_rgb():
    image = np.zeros((40, 40, 4), np.float32)
    image[..., :3] = [.8, .1, .9]
    image[0, 0] = [.1, .8, .2, 1]
    cells = recover_cells(image, [0, 40], [0, 40], max_samples=16)
    np.testing.assert_allclose(cells.rgba[0, 0, :3], [.1, .8, .2])
    assert cells.rgba[0, 0, 3] == pytest.approx(1 / 1600)
