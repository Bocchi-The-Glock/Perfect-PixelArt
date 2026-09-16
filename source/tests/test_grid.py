"""Synthetic grids with known geometry; no downloaded test assets."""
from dataclasses import replace

import numpy as np

from pixelperfect.config import Config
from pixelperfect.features import extract_features
from pixelperfect.grid import generate_candidates, make_lines, refine_candidate


def random_grid(width=12, height=10, spacing=8, seed=7):
    rng = np.random.default_rng(seed)
    truth = np.ones((height, width, 4), dtype=np.float32)
    truth[..., :3] = rng.uniform(.05, .95, (height, width, 3))
    return np.repeat(np.repeat(truth, spacing, axis=0), spacing, axis=1), truth


def test_clean_grid_and_harmonics():
    image, _ = random_grid()
    f = extract_features(image)
    grids = generate_candidates(f, image.shape, Config(local_warp="off"))
    first = grids[0]
    assert (first.sx, first.sy, first.phase_x, first.phase_y) == (8., 8., 0., 0.)
    assert (len(first.x_lines) - 1, len(first.y_lines) - 1) == (12, 10)
    assert any(c.sx == 4 and c.sy == 4 for c in grids)
    assert any(c.sx == 16 and c.sy == 16 for c in grids)
    axis = first.metadata["axis_candidates"]["x"]
    assert len(axis["selected"]) <= 6
    assert any("half" in c["families"] for c in axis["considered"])
    assert any("double" in c["families"] for c in axis["considered"])


def test_shifted_crop_recovers_phase_and_covers_input():
    image, _ = random_grid(width=16, height=14)
    image = image[5:-2, 3:-1]
    grid = generate_candidates(extract_features(image), image.shape, Config())[0]
    assert abs(grid.sx - 8) < .05 and abs(grid.sy - 8) < .05
    assert abs(grid.phase_x - 5) < .15 and abs(grid.phase_y - 3) < .15
    assert grid.x_lines[0] == grid.y_lines[0] == 0
    assert grid.x_lines[-1] == image.shape[1] and grid.y_lines[-1] == image.shape[0]
    assert np.all(np.diff(grid.x_lines) > 0) and np.all(np.diff(grid.y_lines) > 0)


def test_partial_fragments_have_consistent_rule():
    # 1-pixel edge remnants at spacing8 merge; 2-pixel remnants remain.
    assert np.array_equal(make_lines(18, 8, 1), [0, 9, 18])
    assert np.array_equal(make_lines(18, 8, 2), [0, 2, 10, 18])
    assert np.array_equal(make_lines(17, 8, 0), [0, 8, 17])


def test_target_count_remains_exact_under_phase_shift():
    image, _ = random_grid()
    image = np.roll(image, 2, axis=1)
    config = Config(target_size=(12, 10))
    grid = generate_candidates(extract_features(image), image.shape, config)[0]
    assert len(grid.x_lines) == 13 and len(grid.y_lines) == 11
    assert grid.x_lines[0] == 0 and grid.x_lines[-1] == image.shape[1]
    assert np.all(np.diff(grid.x_lines) > 0)


def test_alpha_hidden_rgb_cannot_create_edges():
    image, _ = random_grid()
    image[:, :32, 3] = 0
    other = image.copy()
    other[:, :32, :3] = np.random.default_rng(31).random(other[:, :32, :3].shape)
    a, b = extract_features(image), extract_features(other)
    assert np.array_equal(a.gradient_x, b.gradient_x)
    assert np.array_equal(a.gradient_y, b.gradient_y)
    assert np.array_equal(a.profile_x, b.profile_x)


def test_equal_luminance_colour_edges_remain_visible():
    image = np.ones((32, 64, 4), np.float32)
    image[..., :3] = [1, 0, 0]
    image[:, 32:, :3] = [0, .299 / .587, 0]
    f = extract_features(image)
    assert f.raw_gradient_x[:, 32].mean() > .3


def test_flat_and_transparent_images_have_no_automatic_grid():
    for image in (np.ones((64, 64, 4), np.float32), np.zeros((64, 64, 4), np.float32)):
        f = extract_features(image)
        assert generate_candidates(f, image.shape, Config()) == []
        assert not f.active_x.any() and not f.active_y.any()
        assert len(generate_candidates(f, image.shape, Config(pixel_size=8))) == 1


def test_missing_same_colour_boundaries_still_find_base_scale():
    image, truth = random_grid(width=18, height=16)
    truth[2:7, 2:12, :3] = truth[2, 2, :3]
    image = np.repeat(np.repeat(truth, 8, axis=0), 8, axis=1)
    grids = generate_candidates(extract_features(image), image.shape, Config())
    assert grids[0].sx == 8 and grids[0].sy == 8


def test_local_warp_requires_multiple_strips_and_stays_bounded():
    image, _ = random_grid(width=16, height=14)
    f = extract_features(image)
    config = Config(pixel_size=8)
    grid = generate_candidates(f, image.shape, config)[0]
    # Deliberately offset the fixed-spacing initialization from coherent edges.
    shifted = replace(grid, x_lines=grid.x_lines.copy())
    shifted.x_lines[1:-1] += 1
    result = refine_candidate(shifted, f, config)
    assert result.warped
    assert np.all(np.diff(result.x_lines) > 0)
    assert np.max(np.abs(result.x_lines - shifted.x_lines)) <= .18 * 8 + 1e-8
    assert np.mean(abs(result.x_lines[1:-1] - grid.x_lines[1:-1])) < .1
    single = replace(f, profiles_x=f.profiles_x[:1], active_x=f.active_x[:1])
    result_single = refine_candidate(shifted, single, config)
    assert np.array_equal(result_single.x_lines, shifted.x_lines)


def test_noninteger_manual_grid_and_repeatability():
    image, _ = random_grid()
    f = extract_features(image)
    config = Config(pixel_size=(7.5, 8.25), local_warp="off")
    a = generate_candidates(f, image.shape, config)[0]
    b = generate_candidates(f, image.shape, config)[0]
    assert a.sx == 7.5 and a.sy == 8.25
    assert np.array_equal(a.x_lines, b.x_lines)
    assert a.metadata == b.metadata


def test_noninteger_automatic_spacing_with_raster_rounding():
    _, truth = random_grid(width=22, height=18)
    sx, sy = 7.3, 7.8
    # Continuous nearest-neighbour image rasterization produces unequal
    # integer block widths. It retains mean-period evidence, but the recovered
    # phase is only identifiable to a fraction of the source pixel width.
    xi = np.minimum((np.arange(round(22 * sx)) / sx).astype(int), 21)
    yi = np.minimum((np.arange(round(18 * sy)) / sy).astype(int), 17)
    image = truth[yi[:, None], xi[None, :]]
    grid = generate_candidates(extract_features(image), image.shape, Config())[0]
    assert abs(grid.sx - sx) < .06 and abs(grid.sy - sy) < .06
    assert len(grid.x_lines) == 23 and len(grid.y_lines) == 19


def test_unstructured_noise_and_smooth_ramp_do_not_establish_grid():
    rng = np.random.default_rng(919)
    noise = np.ones((96, 96, 4), np.float32)
    noise[..., :3] = rng.uniform(0, 1, (96, 96, 3))
    flat_noise = np.ones_like(noise)
    flat_noise[..., :3] = np.clip(rng.normal(.5, .025, (96, 96, 3)), 0, 1)
    ramp = np.ones_like(noise)
    ramp[..., :3] = ((np.arange(96)[:, None] + np.arange(96)[None, :]) / 190)[..., None]
    for image in (noise, flat_noise, ramp):
        assert generate_candidates(extract_features(image), image.shape, Config()) == []
