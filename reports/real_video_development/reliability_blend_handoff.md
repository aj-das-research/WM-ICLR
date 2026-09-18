# Reliability diagnostic handoff

- Job **200201**: COMPLETED, exit 0; CPU only, 8 allocated CPUs, 16 GiB request; 103 s Slurm elapsed / 95.57 s measured study time, 0.97 GiB peak RSS.
- All three seeds, eleven arms. Training: 758 eligible episodes / 295 sessions / 6,078 windows; validation: 132 episodes / 57 sessions / 1,351 windows. The 93 train and 11 validation short recordings remain audited.
- **Do not promote**: shrunk gate h5 0.15175676 versus 0.15176811 for the prior blend; +0.00747%, paired MSE interval [-0.00002378,+0.00000256]. One seed does not improve; the 1% and interval criteria fail.
- All training fits selected prior=1, kappa=100. Equal averaging is a stronger mean control: h5 0.15151322, h10 0.21178491. These are exploratory validation results, not new test confirmation or algorithmic novelty.
- Ten engineering tests passed. Independent verification recomputed 8,712 per-episode endpoint values (maximum discrepancy 2.22e-16) and exactly reproduced the primary 10,000-draw session/seed bootstrap interval. All 45 dependency hashes, 32 exported artifact-file hashes and six cache hashes match.
- Portable gate packages and six existing donor copies: `artifacts/development/reliability_blend_v1/`; exact offline reload parity for all seeds, no newly trained neural weights. FP64 table scoring and FP32 exported inference precision are explicitly distinguished.
- Paper integration input: `paper/sections/reliability_development.tex`; all-eleven-arm table and source ledger: `paper/generated/real_video/reliability_development_*`. Standalone ICLR-width proof inspected, no overflow warnings. Main paper source and full paper build were not touched by this task.
