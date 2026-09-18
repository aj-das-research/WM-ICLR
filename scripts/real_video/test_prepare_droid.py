"""Meaningful leakage/action-alignment checks for the real-video extractor."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("prepare_droid", Path(__file__).with_name("prepare_droid.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_distinct_episodes_same_site_day_are_one_split_group():
    a = "/nfs/data/RAIL/success/2023-04-17/Mon_Apr_17_14:48:05_2023/"
    b = "/nfs/data/RAIL/success/2023-04-17/Mon_Apr_17_18:22:15_2023/"
    sa = m.session_from_metadata(a+"trajectory.h5", a+"recordings/MP4")
    sb = m.session_from_metadata(b+"trajectory.h5", b+"recordings/MP4")
    assert sa == sb == "RAIL/2023-04-17"
    assert m.split_for(sa) == m.split_for(sb)


def test_unknown_or_disagreeing_session_is_not_replaced_by_episode_key():
    good = "/nfs/RAIL/success/2023-04-17/episode/trajectory.h5"
    with pytest.raises(ValueError):
        m.session_from_metadata(good, good.replace("2023-04-17", "2023-04-18"))
    with pytest.raises(ValueError):
        m.session_from_metadata("/unknown/episode/file", "/unknown/episode/video")


def test_group_actions_cannot_cross_last_observation_or_reorder_native_controls():
    native = np.arange(13*7, dtype=np.float64).reshape(13,7)
    actions, indices = m.group_actions(native)
    assert indices.tolist() == [0,5,10]
    assert actions.shape == (2,35)
    np.testing.assert_array_equal(actions[0], native[:5].reshape(35))
    np.testing.assert_array_equal(actions[1], native[5:10].reshape(35))
    assert not np.isin(native[10:].ravel(), actions).any()


def test_invalid_native_actions_are_rejected():
    for native in (np.ones((8,8)), np.empty((0,7)), np.full((9,7), np.nan)):
        with pytest.raises(ValueError):
            m.group_actions(native)
