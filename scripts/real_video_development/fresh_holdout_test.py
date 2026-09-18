"""Meaningful acquisition/identity guard checks; no data or image access."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("fresh_holdout", Path(__file__).with_name("fresh_holdout.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FreshHoldoutTests(unittest.TestCase):
    def test_deterministic_shards_and_exclusion(self):
        items = [{"name": f"robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-{i:05d}-of-02048"} for i in range(2048)]
        excluded = {row["name"] for row in items[:24]}
        actual = module.select_shards(items, excluded)
        self.assertEqual(actual, module.select_shards(list(reversed(items)), excluded))
        self.assertEqual(len(actual), 12)
        self.assertFalse(excluded & {row["name"] for row in actual})

    def test_incomplete_listing_is_rejected(self):
        with self.assertRaises(ValueError):
            module.select_shards([], [])

    def test_status_does_not_change_session(self):
        self.assertEqual(module.session_from_metadata("/site/success/2023-07-03/episode", "/site/failure/2023-07-03/recording"), "site/2023-07-03")

    def test_session_disagreement_fails(self):
        with self.assertRaises(ValueError):
            module.session_from_metadata("/site/success/2023-07-03/episode", "/site/success/2023-07-04/recording")

    def test_no_episode_fallback_without_date(self):
        with self.assertRaises(ValueError):
            module.session_from_metadata("/site/episode", "/site/recording")

    def test_exclusion_union_including_original_val_and_test(self):
        rows = [dict(episode_id="new", session_id="original-test", serialized_example_sha256="new-hash"),
                dict(episode_id="original-id", session_id="new-session", serialized_example_sha256="different-hash"),
                dict(episode_id="another", session_id="new-session", serialized_example_sha256="original-hash"),
                dict(episode_id="keep", session_id="fresh-session", serialized_example_sha256="fresh-hash")]
        old = dict(session_ids=["original-train", "original-val", "original-test"], episode_ids=["original-id"], serialized_example_sha256=["original-hash"])
        result = module.mark_exclusions(rows, old)
        self.assertEqual([row["episode_id"] for row in result if row["retained"]], ["keep"])
        self.assertEqual(result[0]["excluded_reasons"], ["original_session"])

    def test_duplicate_identity_fails(self):
        rows = [dict(episode_id="same", session_id="a", serialized_example_sha256="1"),
                dict(episode_id="same", session_id="b", serialized_example_sha256="2")]
        with self.assertRaises(ValueError):
            module.mark_exclusions(rows, dict(session_ids=[], episode_ids=[], serialized_example_sha256=[]))


if __name__ == "__main__":
    unittest.main()
