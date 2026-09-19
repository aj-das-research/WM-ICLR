# IWS Box/Rope cache completion

Both registered native feature caches are complete and were independently revalidated on CPU.

| Task | Trajectories | Internal train / development | Native frames and command rows | Command width | Full packages verified |
|---|---:|---:|---:|---:|---:|
| Box | 602 | 481 / 121 | 120,274 | 14 | 602 |
| Rope | 602 | 481 / 121 | 120,312 | 8 | 602 |

Every one of the 1,204 package identities, payload checksums, finite feature/command arrays, native row indices, and registered source identities passed the existing fail-closed validator. Both original job logs contain all 602 sequential episode-verification events and their final all-packages-verified receipt. Normalization uses only each task's 481 internal-training recordings. The ten official-validation IDs per task are absent from package indexes/directories, build logs, and statistics; the frozen loader rejects them before video/HDF5 access. No raw videos or official-validation payloads were opened by this completion audit.

No jobs were listed by `squeue` at the receipt timestamp. Jobs 200465/200466 are no longer retained by `scontrol`, and `sacct` cannot reach its accounting database. Their final Slurm exit codes are therefore **unavailable**, not assumed. The fresh CPU validation processes both exited 0; that is separate from historical scheduler exit codes.

This completes feature preparation only. IWS predictor training, checkpoints and benchmark results remain pending. PushT's previously completed 600-recording / 119,887-frame cache is unchanged; all three prepared caches together contain 1,804 recordings and 360,473 native frames.

The exact manifests, source hashes, log hashes, current scheduler outputs and commands are in `reports/evidence/iws_box_rope_cache_completion.json`.
