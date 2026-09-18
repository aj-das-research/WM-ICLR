"""Scientific correctness checks for metrics and bounded upstream CEM costs."""
from types import SimpleNamespace
import json

import numpy as np
import torch
from torch import nn

from shiftwm.evaluate import LatentGoalCost, clustered_interval, evaluate_planning


class ActionIntegrator(nn.Module):
    """Known transition function solely for numerical planner contract tests."""
    def __init__(self, action_mean=0.0, action_std=1.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("action_mean", torch.full((10,), action_mean))
        self.register_buffer("action_std", torch.full((10,), action_std))

    def rollout_features(self, history, past, future, contexts=None):
        return future.sum(-1, keepdim=True).cumsum(1)


def test_bootstrap_clusters_do_not_count_windows_as_independent():
    rows = [{"trajectory": "a", "score": 0.0}] * 20 + [{"trajectory": "b", "score": 1.0}] * 20
    result = clustered_interval(rows, "score", "trajectory", repetitions=500)
    assert result["clusters"] == 2
    assert result["observations"] == 40
    assert result["mean"] == .5
    assert result["ci95"] == [0.0, 1.0]


def test_cem_prediction_and_execution_use_identical_bounded_actions():
    cost = LatentGoalCost(ActionIntegrator(action_mean=256., action_std=64.), "pusht", action_interface="absolute")
    normalized = torch.tensor([-10., 0., 1., 6., -1.] * 2).reshape(1, 1, 1, 10)
    raw = cost.to_native(normalized)
    assert raw.min() == 0
    assert raw.max() == 512
    torch.testing.assert_close(raw.flatten(), torch.tensor([0., 256., 320., 512., 192.] * 2))
    info = {"history_features": torch.zeros(1, 1, 3, 1), "past_actions": torch.zeros(1, 1, 2, 10),
            "observation_context": torch.zeros(1, 1, 1), "dynamics_context": torch.zeros(1, 1, 1),
            "goal_features": raw.sum(-1).reshape(1, 1, 1)}
    torch.testing.assert_close(cost.get_cost(info, normalized), torch.zeros(1, 1))


def test_relative_search_scores_are_not_clipped_before_denormalization():
    cost = LatentGoalCost(ActionIntegrator(action_mean=0., action_std=.2), "pusht")
    z_scores = torch.tensor([3., -3., 7., -7., 0.] * 2)
    torch.testing.assert_close(cost.to_native(z_scores), torch.tensor([.6, -.6, 1., -1., 0.] * 2))


def test_upstream_cem_cost_adapter_handles_real_candidate_axes():
    from gymnasium.spaces import Box
    from stable_worldmodel.planning.solver.cem import CEMSolver
    cost = LatentGoalCost(ActionIntegrator(), "reacher")
    solver = CEMSolver(cost, num_samples=32, n_steps=3, topk=8, seed=19)
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32), n_envs=1,
                     config=SimpleNamespace(horizon=2, action_block=5))
    info = {"history_features": torch.zeros(1, 3, 1), "past_actions": torch.zeros(1, 2, 10),
            "observation_context": torch.zeros(1, 1), "dynamics_context": torch.zeros(1, 1),
            "goal_features": torch.zeros(1, 1)}
    result = solver.solve(info)
    assert result["actions"].shape == (1, 2, 10)
    assert torch.isfinite(result["actions"]).all()


def test_planning_resumes_completed_real_episodes_without_duplicates(tmp_path, monkeypatch):
    """Interrupted reachability audit resumes by exact episode/condition identity."""
    from shiftwm.generate import collect_episode
    from shiftwm.data import TRAIN_COMBINATIONS
    import shiftwm.evaluate as evaluation

    root = tmp_path / "data"
    episode = collect_episode({"env": "pusht", "output": str(root), "split": "test", "seed": 31005,
                               "dynamics_id": 0, "image_size": 32, "steps": 9, "action_block": 5,
                               "action_interface": "relative"})
    manifest = {"environment": "pusht", "image_size": 32, "action_block": 5,
                "action_interface": "relative", "episodes": [episode],
                "train_combinations": TRAIN_COMBINATIONS}
    (root / "manifest.json").write_text(json.dumps(manifest))
    model = ActionIntegrator(action_std=.2)
    model.config = SimpleNamespace(history_length=3)
    model.action_dim = 10
    model.encode_images = lambda images: images.mean(dim=(-3, -2, -1))[:, None]
    progress = tmp_path / "progress.json"
    # Advance the deadline only after one complete real episode was persisted.
    with monkeypatch.context() as patch:
        patch.setattr(evaluation.time, "time", lambda: 2.0 if progress.exists() else 0.0)
        partial = evaluate_planning(model, root, episodes_per_dynamics=1, samples=8, iterations=1,
                                    elites=2, policy="replay_oracle", progress_path=progress,
                                    max_runtime_seconds=1, run_identity={"test": "resume"})
    assert partial["status"] == "interrupted"
    assert len(partial["records"]) == 1
    result = evaluate_planning(model, root, episodes_per_dynamics=1, samples=8, iterations=1,
                               elites=2, policy="replay_oracle", progress_path=progress,
                               run_identity={"test": "resume"})
    assert result["status"] == "complete"
    assert len(result["records"]) == 3
    assert len({(row["trajectory_id"], row["observation_id"]) for row in result["records"]}) == 3
    assert all(row["success"] for row in result["records"])
