"""Exact horizon/baseline math and session-level matched uncertainty guards."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_video_evaluation", ROOT / "scripts/real_video/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.episodes = [{"episode_id": "recording-a", "session_id": "session-a"},
                         {"episode_id": "recording-b", "session_id": "session-b"}]
        self.features = torch.arange(3+horizon, dtype=torch.float32)[None,:,None].repeat(3,1,2)
        self.actions = torch.arange(2+horizon, dtype=torch.float32)[None,:,None].repeat(3,1,10)
        self.episode_indices = [0,0,1]
    def __len__(self):
        return 3
    def __getitem__(self,index):
        return {"features": self.features[index], "actions": self.actions[index],
                "episode_index": self.episode_indices[index], "window_start": index}


class RecordingPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_std", torch.tensor([2.,2.]))
        self.observed = []
    def predict(self,support,past,future):
        assert support.shape[1] == 3 and past.shape[1] == 2
        assert not self.training and torch.is_inference_mode_enabled()
        self.observed.append(tuple(t.detach().clone() for t in (support,past,future)))
        return evaluation.baseline_predictions(support,len(future[0]))["constant_velocity"]


@pytest.mark.parametrize("horizon", [5,10])
def test_exact_horizon_math_and_episode_weighting(horizon):
    model = RecordingPredictor()
    result = evaluation.evaluate_dataset(model,LinearDataset(horizon),batch_size=2)
    assert result["episode_count"] == result["session_count"] == 2
    assert result["window_count"] == 3
    summary = result["summary"]
    assert summary["model"]["mean_raw_mse"] == 0
    assert summary["constant_velocity"]["mean_raw_mse"] == 0
    expected = np.mean(np.arange(1,horizon+1)**2)
    assert summary["persistence"]["mean_raw_mse"] == expected
    assert summary["persistence"]["mean_standardized_mse"] == expected/4
    for h in ((1,3,5) if horizon==5 else (10,)):
        assert summary["persistence"][f"h{h}_raw_mse"] == h*h
        assert summary["persistence"][f"h{h}_standardized_mse"] == h*h/4
        assert summary["model"][f"h{h}_cosine_error"] < 1e-6
    assert set(key for key in summary["model"] if key.startswith("h")) == {
        f"h{h}_{metric}" for h in ((1,3,5) if horizon==5 else (10,))
        for metric in ("raw_mse","standardized_mse","cosine_error")}


def test_future_images_never_enter_prediction_inputs():
    dataset = LinearDataset()
    first = RecordingPredictor()
    evaluation.evaluate_dataset(first,dataset)
    dataset.features[:,3:] += 10000
    second = RecordingPredictor()
    evaluation.evaluate_dataset(second,dataset)
    for inputs_a, inputs_b in zip(first.observed,second.observed):
        for a,b in zip(inputs_a,inputs_b):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_action_reversal_preserves_support_and_within_block_order():
    actions = torch.arange(2*7*35).reshape(2,7,35)
    reversed_actions = evaluation.reverse_future_actions(actions)
    assert torch.equal(reversed_actions[:,:2],actions[:,:2])
    assert torch.equal(reversed_actions[:,2],actions[:,-1])
    assert torch.equal(reversed_actions[:,-1],actions[:,2])
    assert torch.equal(evaluation.reverse_future_actions(reversed_actions),actions)


def test_equal_episode_summary_does_not_overweight_overlapping_windows():
    rows = [{"episode_id":"a","windows":100,"errors":{"model":{"mean_standardized_mse":1.}}},
            {"episode_id":"b","windows":1,"errors":{"model":{"mean_standardized_mse":9.}}}]
    assert evaluation.summarize_episode_errors(rows)["model"]["mean_standardized_mse"] == 5.


def test_session_bootstrap_preserves_cluster_episodes_and_seed_variation():
    # Two episodes in the first session. A session-mean point estimate would
    # equal 5; the registered equal-episode estimand is 10/3.
    values = np.array([[0,0,10],[1,1,11],[2,2,12]],dtype=float)
    result = evaluation.crossed_session_bootstrap(values,["a","a","b"],draws=1000,seed=19)
    assert result["mean_difference"] == pytest.approx(13/3)
    assert result == evaluation.crossed_session_bootstrap(values,["a","a","b"],draws=1000,seed=19)
    assert result["ci95"][0] < result["mean_difference"] < result["ci95"][1]
    tie = evaluation.crossed_session_bootstrap(np.zeros((3,3)),["a","a","b"],draws=17)
    assert tie["ci95"] == [0.,0.] and tie["mean_difference"] == 0
    with pytest.raises(ValueError,match="three-seed"):
        evaluation.crossed_session_bootstrap(values[:2],["a","a","b"])
    with pytest.raises(ValueError,match="two independent"):
        evaluation.crossed_session_bootstrap(values,["a","a","a"])


def result_fixture(mode,seed):
    value = {"status":"completed","mode":mode,"seed":seed,"source_dependencies":{},"test_payloads":{},
             "population":"primary","camera":evaluation.CAMERAS[0],"horizon":5,
             "cache_manifest_sha256":"cache","dataset_manifest_sha256":"real",
             "source_sha256":"source","aggregation":"equal episodes","precision":"fp32"}
    rows = []
    for i in range(3):
        errors = {name:{"h5_standardized_mse":float(i+1+(seed+(mode=="framewise") if name in ("model","reversed_future_actions") else 0))}
                  for name in ("model","persistence","constant_velocity","reversed_future_actions")}
        rows.append({"episode_id":str(i),"session_id":"a" if i<2 else "b",
                     "windows":1,"window_starts":[0],"errors":errors})
    value["episodes"] = rows
    value["summary"] = evaluation.summarize_episode_errors(rows)
    return value


def test_aggregation_requires_all_three_seeds_and_exact_population(tmp_path,monkeypatch):
    # Package integrity is tested separately by the full-epoch lifecycle suite;
    # this fixture isolates matched-population and bootstrap arithmetic.
    monkeypatch.setattr(evaluation,"validate_result_checkpoint",lambda record: None)
    paths = []
    for mode in evaluation.training.RealVideoWorldModel.MODES:
        for seed in (0,1,2):
            path = tmp_path/f"{mode}-{seed}.json"
            evaluation.training.atomic_json(result_fixture(mode,seed),path)
            paths.append(path)
    with pytest.raises(ValueError,match="all four"):
        evaluation.aggregate(paths[:-1],tmp_path/"out.json",draws=100)
    complete = evaluation.aggregate(paths,tmp_path/"out.json",draws=100)
    framewise = next(row for row in complete["paired_comparisons"] if row["reference"]=="framewise")
    assert framewise["mean_difference"] == -1 and framewise["ci95"] == [-1.,-1.]
    value = result_fixture("action_free",2)
    value["camera"] = evaluation.CAMERAS[1]
    evaluation.training.atomic_json(value,paths[-1])
    with pytest.raises(ValueError,match="Unmatched"):
        evaluation.aggregate(paths,tmp_path/"out.json",draws=100)
