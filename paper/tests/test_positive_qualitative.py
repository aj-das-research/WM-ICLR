"""Selection, timing and display-mask guards; synthetic fixtures are not results."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_positive_qualitative.py"
SPEC = importlib.util.spec_from_file_location("positive_qualitative_test", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def example(environment="pusht", trajectory="episode0", category="ours_only", observation=1):
    return {"environment": environment, "trajectory_id": trajectory,
            "observation_id": observation, "category": category}


def timed_example(ours, baseline, ours_stop=None):
    return {"methods": {mode: {"frame_steps": steps,
            "frames": np.stack([np.full((3, 3, 3), step, dtype=np.uint8) for step in steps]),
            "record": {"native_steps": (ours_stop if mode == "factorized" and ours_stop is not None else steps[-1])}}
            for mode, steps in (("factorized", ours), ("framewise", baseline))}}


def test_positive_selection_is_exhaustive_not_limited_to_display_page_size():
    cases = [example(env, f"episode{index:02}", category)
             for env in ("pusht", "reacher")
             for index, category in enumerate(["ours_only"] * 7 + ["baseline_only", "neither", "both", "support_success"])]
    selected = report.positive_cases(list(reversed(cases)))
    expected = [case for case in cases if case["category"] == "ours_only"]
    assert selected == expected
    assert len(selected) == 14  # Deliberately exceeds the current four-case figure.
    assert report.positive_cases(cases) == selected
    assert all(case["category"] == "ours_only" for case in selected)
    assert report.positive_cases([example(category="support_success")]) == []


def test_positive_sort_uses_environment_trajectory_and_observation_without_mutation():
    cases = [example("reacher", "z", observation=2), example("pusht", "b"),
             example("pusht", "a", observation=2), example("pusht", "a", observation=1)]
    original = deepcopy(cases)
    assert [(case["environment"], case["trajectory_id"], case["observation_id"])
            for case in report.positive_cases(cases)] == [
                ("pusht", "a", 1), ("pusht", "a", 2), ("pusht", "b", 1), ("reacher", "z", 2)]
    assert cases == original


def test_partial_terminal_time_uses_recorded_shared_frame_not_interpolation():
    case = timed_example([10, 15, 20, 23], [10, 15, 20, 25, 30, 35, 40, 45, 50])
    before = {mode: item["frames"].copy() for mode, item in case["methods"].items()}
    time, indices = report.matched_frame(case)
    assert time == 20 and indices == {"factorized": 2, "framewise": 2}
    for mode, item in case["methods"].items():
        assert item["frame_steps"][indices[mode]] == time
        np.testing.assert_array_equal(item["frames"][indices[mode]], np.full((3, 3, 3), 20))
        np.testing.assert_array_equal(item["frames"], before[mode])
    assert case["methods"]["factorized"]["record"]["native_steps"] == 23


@pytest.mark.parametrize("ours,baseline,expected", [
    ([10, 15, 20], [10, 15, 20, 25, 30], 20),
    ([10, 11], [10, 15, 20], 10),
    ([10, 15, 17], [10, 15, 16], 15),
    ([10, 15, 20, 25], [10, 20, 25, 30], 25),
])
def test_matching_uses_native_times_instead_of_equal_frame_indices(ours, baseline, expected):
    case = timed_example(ours, baseline)
    time, indices = report.matched_frame(case)
    assert time == expected
    assert indices == {"factorized": ours.index(expected), "framewise": baseline.index(expected)}


def test_matching_never_uses_a_frame_after_recorded_stop_and_rejects_no_shared_time():
    case = timed_example([10, 15, 20, 25], [10, 15, 20, 25, 30], ours_stop=23)
    assert report.matched_frame(case)[0] == 20
    with pytest.raises(ValueError, match="actually recorded shared"):
        report.matched_frame(timed_example([10, 15], [11, 16]))


def pusht_goal():
    # Palette values are observed in the actual warm development goal image.
    pixels = np.full((224, 224, 3), [247, 206, 178], dtype=np.uint8)
    pixels[20:32, 30:42] = [144, 134, 130]  # Warm gray task block.
    pixels[80:86, 100:106] = [84, 105, 178]  # Blue task pusher.
    pixels[145:170, 150:175] = [145, 193, 103]  # Persistent green renderer marker.
    pixels[180:190, 20:40] = [20, 20, 20]  # Dark border/background.
    return pixels


def test_pusht_goal_mask_includes_gray_and_blue_but_excludes_green_and_background():
    goal = pusht_goal()
    original = goal.copy()
    mask = report.goal_mask(goal, "pusht")
    expected = np.zeros((224, 224), dtype=bool)
    expected[20:32, 30:42] = True
    expected[80:86, 100:106] = True
    np.testing.assert_array_equal(mask, expected)
    assert not mask[145:170, 150:175].any()
    np.testing.assert_array_equal(goal, original)


@pytest.mark.parametrize("missing", ["block", "pusher"])
def test_pusht_annotation_requires_both_task_objects(missing):
    goal = pusht_goal()
    if missing == "block":
        goal[20:32, 30:42] = [250, 240, 235]
    else:
        goal[80:86, 100:106] = [250, 240, 235]
    with pytest.raises(ValueError, match="block/pusher"):
        report.goal_mask(goal, "pusht")


def test_reacher_mask_is_orange_only_and_leaves_original_goal_unchanged():
    goal = np.full((224, 224, 3), [22, 24, 26], dtype=np.uint8)
    goal[30:40, 150:165] = [225, 125, 35]
    for index, color in enumerate(([190, 190, 190], [255, 255, 255], [230, 20, 20],
                                    [20, 220, 20], [20, 70, 230], [240, 240, 20])):
        goal[100:110, index * 20:index * 20 + 10] = color
    original = goal.copy()
    expected = np.zeros((224, 224), dtype=bool)
    expected[30:40, 150:165] = True
    np.testing.assert_array_equal(report.goal_mask(goal, "reacher"), expected)
    np.testing.assert_array_equal(goal, original)


@pytest.mark.parametrize("environment", ["pusht", "reacher"])
@pytest.mark.parametrize("damage", ["dtype", "shape"])
def test_masks_reject_nonoriginal_rgb_shapes_or_dtype(environment, damage):
    goal = pusht_goal()
    goal = goal.astype(np.float32) if damage == "dtype" else goal[:200]
    with pytest.raises(ValueError, match="original 224x224 uint8"):
        report.goal_mask(goal, environment)


def test_masks_reject_unknown_environment_or_implausible_silhouette():
    with pytest.raises(ValueError, match="Unknown environment"):
        report.goal_mask(pusht_goal(), "other")
    for goal in (np.zeros((224, 224, 3), dtype=np.uint8),
                 np.full((224, 224, 3), [225, 125, 35], dtype=np.uint8)):
        with pytest.raises(ValueError, match="empty or implausible"):
            report.goal_mask(goal, "reacher")


def test_contour_geometry_matches_asymmetric_pixel_rows_and_columns():
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[12:27, 75:88] = [230, 120, 20]
    mask = np.zeros((224, 224), dtype=bool)
    mask[12:27, 75:88] = True
    fig = plt.figure(figsize=(3, 3))
    try:
        axes = report.image_panel(fig, [.1, .1, .8, .8], pixels, "fixture", mask=mask)
        np.testing.assert_array_equal(axes.images[0].get_array(), pixels)
        vertices = np.concatenate([path.vertices for collection in axes.collections
                                   for path in collection.get_paths()])
        # Pixel centers use x=column and y=row; interpolation at .5 outlines
        # the boundary half a pixel outside the first/last occupied center.
        np.testing.assert_allclose(vertices.min(axis=0), [74.5, 11.5])
        np.testing.assert_allclose(vertices.max(axis=0), [87.5, 26.5])
        assert axes.get_xlim() == (-.5, 223.5)
        assert axes.get_ylim() == (223.5, -.5)
        # Small y values must appear near the image top, not vertically reflected.
        top = axes.transData.transform((80, 12))[1]
        bottom = axes.transData.transform((80, 211))[1]
        assert top > bottom
    finally:
        plt.close(fig)


def orange_pose(left, top, width=10):
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[top:top + width, left:left + width] = [225, 125, 35]
    return pixels


def displayed_images(case, indices):
    images = [case["methods"]["factorized"]["goal"]]
    for mode in ("factorized", "framewise"):
        item = case["methods"][mode]
        images.extend((item["frames"][indices[mode]], item["frames"][-1]))
    return images


def assert_crop_contains_images(case, indices, crop, require_full_padding=True):
    left, top, right, bottom = crop
    assert all(isinstance(value, int) for value in crop)
    assert 0 <= left < right <= 224 and 0 <= top < bottom <= 224
    assert right - left == bottom - top
    for pixels in displayed_images(case, indices):
        foreground = report.goal_mask(pixels, "reacher")
        ys, xs = np.where(foreground)
        assert left <= xs.min() <= xs.max() < right
        assert top <= ys.min() <= ys.max() < bottom
        margins = (int(xs.min()) - left, int(ys.min()) - top,
                   right - 1 - int(xs.max()), bottom - 1 - int(ys.max()))
        if require_full_padding:
            assert min(margins) >= 12
        else:
            # Padding can end at the original image boundary; never invent
            # pixels outside it merely to enforce a twelve-pixel margin.
            assert margins[0] >= min(12, int(xs.min()))
            assert margins[1] >= min(12, int(ys.min()))
            assert margins[2] >= min(12, 223 - int(xs.max()))
            assert margins[3] >= min(12, 223 - int(ys.max()))


def test_detail_crop_includes_goal_and_both_methods_matched_and_terminal_poses():
    case = {"environment": "reacher", "methods": {
        "factorized": {"goal": orange_pose(45, 40),
                       "frames": np.stack((orange_pose(90, 70), orange_pose(60, 100)))},
        "framewise": {"frames": np.stack((orange_pose(130, 60), orange_pose(80, 130)))}}}
    indices = {"factorized": 0, "framewise": 0}
    crop = report.detail_crop(case, indices)
    assert_crop_contains_images(case, indices, crop)
    # Removing a distant baseline endpoint would alter the required shared crop.
    changed = deepcopy(case)
    changed["methods"]["framewise"]["frames"][-1] = orange_pose(60, 60)
    assert report.detail_crop(changed, indices) != crop
    assert report.detail_crop({"environment": "pusht"}, indices) is None


@pytest.mark.parametrize("positions", [[(0, 0), (30, 0)], [(214, 214), (180, 214)],
                                        [(0, 0), (214, 214)]])
def test_square_crop_stays_in_image_and_retains_boundary_foreground(positions):
    first, second = [orange_pose(*position) for position in positions]
    case = {"environment": "reacher", "methods": {
        "factorized": {"goal": first, "frames": np.stack((first, second))},
        "framewise": {"frames": np.stack((second, first))}}}
    indices = {"factorized": 0, "framewise": 0}
    crop = report.detail_crop(case, indices)
    assert_crop_contains_images(case, indices, crop, require_full_padding=False)
    if positions == [(0, 0), (214, 214)]:
        assert crop == (0, 0, 224, 224)


def test_cropped_contour_stays_in_original_pixel_coordinates():
    pixels = orange_pose(75, 12)
    mask = report.goal_mask(pixels, "reacher")
    fig = plt.figure(figsize=(3, 3))
    try:
        crop = (60, 0, 110, 50)
        axes = report.image_panel(fig, [.1, .1, .8, .8], pixels, "fixture", mask=mask, crop=crop)
        np.testing.assert_array_equal(axes.images[0].get_array(), pixels)
        vertices = np.concatenate([path.vertices for collection in axes.collections
                                   for path in collection.get_paths()])
        np.testing.assert_allclose(vertices.min(axis=0), [74.5, 11.5])
        np.testing.assert_allclose(vertices.max(axis=0), [84.5, 21.5])
        assert axes.get_xlim() == (59.5, 109.5)
        assert axes.get_ylim() == (49.5, -.5)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("seed", [2031004, 2031008, 2031011])
def test_actual_three_reacher_cases_share_one_crop_with_all_poses_and_padding(seed):
    root = SCRIPT.parents[2]
    trajectory = f"development-s{seed}-d1"
    case = {"environment": "reacher", "trajectory_id": trajectory, "seed": seed,
            "category": "ours_only", "observation_id": 1, "methods": {}}
    for mode in report.source.MODES:
        directory = root / "results/development_official_budget" / f"reacher_{mode}_s0"
        video = directory / "videos" / f"{trajectory}-o1.npz"
        if not video.exists():
            pytest.skip("Recorded development archives are unavailable in this checkout")
        record = json.loads((directory / "planning_development.json").read_text())
        row = next(row for row in record["planning"]["records"]
                   if row["trajectory_id"] == trajectory and row["observation_id"] == 1)
        with np.load(video, allow_pickle=False) as stored:
            frames, goal = stored["frames"].copy(), stored["goal_image"].copy()
        case["methods"][mode] = {"record": row, "frames": frames, "goal": goal,
                                  "frame_steps": report.source.frame_steps(row, frames)}
    _, indices = report.matched_frame(case)
    crop = report.detail_crop(case, indices)
    assert_crop_contains_images(case, indices, crop)
    fig = plt.figure(figsize=(5.5, 6.4))
    try:
        metadata = report.draw_example(fig, case, 0, 1, "A")
        assert metadata["display_crop_xyxy"] == list(crop)
        assert metadata["detail_magnification"] == 224 / (crop[2] - crop[0])
        views = [axes for axes in fig.axes if axes.images and axes.get_xlim() == (crop[0] - .5, crop[2] - .5)]
        assert len(views) == 6  # Goal, shared-time and endpoint in both method rows.
        assert len({axes.get_ylim() for axes in views}) == 1
        sizes = np.array([axes.get_position().bounds[2:] for axes in views])
        np.testing.assert_allclose(sizes, np.repeat(sizes[:1], len(views), axis=0), rtol=0, atol=1e-12)
        overview = [axes for axes in fig.axes if axes.images and axes not in views]
        assert len(overview) == 1 and overview[0].get_xlim() == (-.5, 223.5)
        assert overview[0].get_ylim() == (223.5, -.5)
    finally:
        plt.close(fig)
