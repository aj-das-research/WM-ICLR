#!/usr/bin/env python3
"""Pin source and evidence hashes for the current manuscript artifact."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
relative = ["paper/main.tex", "paper/references.bib", "paper/figures/world_method.tex",
            "paper/figures/factor_split.tex", "paper/world_model_draft.pdf",
            "src/shiftwm/model.py", "src/shiftwm/evaluate.py", "src/shiftwm/data.py",
            "reports/model_parameter_counts.json", "paper/generated/training_ledger.json",
            "paper/generated/result_ledger.json", "paper/generated/primary_results.json",
            "paper/evidence/development_budget.json", "paper/generated/official_development.json",
            "paper/generated/goal_calibration.json", "paper/generated/goal_calibration.pdf",
            "paper/scripts/render_goal_calibration.py", "paper/generated/goal_intervention.tex",
            "reports/evidence/goal_intervention_results.json",
            "reports/evidence/upstream_planning_results.json"]
relative += ["paper/generated/forecast_comparison.json", "paper/generated/forecast_comparison.pdf",
             "paper/scripts/render_forecast.py", "paper/generated/training_plot.json",
             "paper/generated/training_curves.pdf", "paper/scripts/render_training.py",
             "paper/evidence/figure_skill_provenance.json"]
relative += ["paper/generated/world_teaser.json", "paper/generated/world_teaser.pdf",
             "paper/scripts/render_teaser.py", "paper/figures/geometry_repair_review.md",
             "paper/figures/arrow_regeneration_review.md",
             "paper/figures/numeric_spacing_review.md",
             "paper/evidence/figure_geometry_checks.json"]
relative += ["scripts/summarize_paired_planning.py", "reports/paired_planning_protocol.md",
             "reports/evidence/paired_planning_results.json", "paper/generated/paired_planning.tex",
             "src/shiftwm/dynamics_revision.py", "scripts/train_dynamics_revision.py",
             "scripts/evaluate_dynamics_revision.py", "scripts/run_dynamics_revision_campaign.py",
             "paper/scripts/render_dynamics_revision.py", "paper/generated/dynamics_revision.json",
             "paper/generated/dynamics_revision.tex", "reports/dynamics_revision_protocol.md",
             "reports/evidence/dynamics_revision_allocations.json",
             "reports/evidence/dynamics_revision_validation.json"]
relative += [str(path.relative_to(root)) for path in sorted((root / 'configs/dynamics_revision').glob('*.json'))]
relative += ["src/shiftwm/rollout_revision.py", "scripts/train_rollout_revision.py",
             "scripts/evaluate_rollout_revision.py", "scripts/run_rollout_revision_campaign.py",
             "paper/scripts/render_rollout_revision.py", "paper/generated/rollout_revision.json",
             "paper/generated/rollout_revision.tex", "reports/rollout_revision_protocol.md",
             "reports/evidence/rollout_revision_allocations.json",
             "reports/evidence/rollout_revision_validation.json",
             "paper/scripts/render_qualitative.py", "paper/generated/qualitative/evidence.json",
             "paper/figures/qualitative_brief.md", "paper/figures/qualitative_review.md",
             "reports/evidence/qualitative_data_validation.json",
             "reports/qualitative_examples_inventory.md"]
relative += ["paper/scripts/render_positive_qualitative.py",
             "paper/generated/qualitative/positive_evidence.json",
             "paper/figures/positive_qualitative_brief.md",
             "paper/figures/positive_qualitative_review.md",
             "reports/qualitative_positive_cases.md",
             "reports/evidence/qualitative_positive_cases.json"]
relative += ["paper/scripts/render_technical_contract.py",
             "paper/scripts/render_technical_qualitative.py",
             "paper/generated/qualitative/technical_contract.json",
             "paper/generated/qualitative/technical_evidence.json",
             "paper/generated/qualitative/technical_summary.tex",
             "paper/figures/technical_qualitative_brief.md",
             "paper/figures/technical_qualitative_review.md",
             "scripts/diagnose_qualitative_replay.py",
             "scripts/diagnose_qualitative_prediction.py",
             "results/qualitative_diagnostics/replay_diagnostics.json",
             "results/qualitative_diagnostics/prediction_diagnostics.json"]
relative += [str(path.relative_to(root)) for path in sorted((root / 'configs/rollout_revision').glob('*.json'))]
relative += [str(path.relative_to(root)) for path in sorted((root / 'paper/generated/qualitative').glob('*.pdf'))]
relative += ["scripts/aggregate_results.py", "paper/generated/primary_results.tex",
             "paper/generated/primary_results_appendix.tex", "paper/generated/method_comparison.tex",
             "paper/generated/primary_comparison_status.tex",
             "paper/scripts/render_gain_summary.py", "paper/generated/primary_gain_summary.json",
             "paper/generated/planning_gain_rows.tex", "paper/generated/forecast_gain_summary.tex",
             "paper/scripts/render_planning_comparison.py", "paper/generated/planning_comparison.json",
             "paper/generated/planning_comparison.pdf"]
relative += ["paper/figures/method.pdf", "paper/figures/method_editorial_redesign_brief.md",
             "paper/figures/method_editorial_redesign_review.md",
             "paper/figures/planning_flow_redesign_brief.md",
             "paper/figures/planning_flow_redesign_review.md",
             "paper/figures/editorial_plot_review.md", "paper/figures/planning_comparison_brief.md",
             "paper/scripts/refresh_training.py", "paper/scripts/record_build.py"]
relative += ["paper/scripts/render_planning_controls.py", "paper/generated/planning_controls.json",
             "paper/generated/planning_controls.tex", "reports/results_diagnosis_2026-09-18.md",
             "reports/evidence/results_diagnosis_2026-09-18.json"]
relative += ["paper/evidence/domain_extension_setup.json", "reports/domain_extension_protocol.md",
             "reports/evidence/expansion_status_2026-09-19.json"]
relative += ["paper/scripts/record_extension_diagnostics.py", "paper/evidence/extension_diagnostics.json",
             "paper/generated/extension_drone_development.tex", "paper/figures/goal_geometry.pdf",
             "paper/figures/goal_geometry_review.json", "paper/figures/geometry_diagnostic_brief.md",
             "artifacts/diagnostics/goal_geometry.json", "reports/extensions_mechanism_diagnosis.md",
             "reports/evidence/extensions_mechanism_probe.json", "reports/evidence/drone_action_ranking.json",
             "reports/evidence/drone_action_ranking_validation.json", "reports/geometry_revision_protocol.md"]
relative += ["paper/evidence/extension_diagnostic_review.json", "paper/figures/extension_diagnostic_review.md"]
relative += ["reports/real_droid_protocol.md", "reports/real_video_benchmark_plan.md",
             "reports/world_model_novelty_update_2026-09-19.md",
             "reports/evidence/real_video/droid_selected_inventory.json",
             "data/real_video/droid_selected/processed/data_audit.json",
             "paper/generated/real_video/recorded_droid_views.pdf",
             "paper/generated/real_video/recorded_droid_views_ledger.json"]
relative += ["scripts/real_video/finalize_campaign.py", "reports/real_droid_results.json",
             "paper/generated/real_video/recorded_droid_views_review.json",
             "paper/figures/real_video_data_review.md", "paper/scripts/render_real_video_data.py"]
relative += ["reports/completed_extension_results.json", "reports/real_droid_interpretation.md",
             "reports/evidence/real_video/execution_receipt.json",
             "scripts/real_video/finalize_droid.slurm",
             "paper/scripts/render_real_video_comparison.py",
             "reports/real_video_comparison_protocol.md",
             "paper/scripts/generate_completed_extensions.py"]
relative += ["reports/real_droid_development_diagnosis.json",
             "reports/real_droid_development_diagnosis_protocol.md",
             "reports/real_droid_residual_calibration_results.json",
             "reports/real_droid_residual_calibration_protocol.md",
             "reports/evidence/real_droid_calibration_exports.json",
             "configs/real_video_development/residual_calibration_v1.json",
             "src/shiftwm/real_video_development.py",
             "scripts/real_video_development/calibrate_residual.py",
             "scripts/real_video_development/summarize_calibration.py",
             "paper/scripts/render_real_video_calibration.py",
             "paper/generated/real_video/residual_calibration.tex",
             "paper/generated/real_video/residual_calibration.sources.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/extensions_completed").glob("*"))
             if path.is_file() and path.suffix in (".tex", ".json")]
relative += ["reports/real_droid_fresh_evaluation_protocol.md",
             "reports/real_droid_fresh_holdout_protocol.md",
             "reports/real_droid_fresh_evaluation_results.json",
             "reports/real_droid_fresh_evaluation_verification.json",
             "reports/real_droid_fresh_evaluation_execution.json",
             "configs/real_video_development/fresh_evaluation_v1.json",
             "data/real_video/droid_fresh_v1/metadata_audit.json",
             "data/real_video/droid_fresh_v1/decoded_confirmatory_v1/data_audit.json",
             "data/features/droid_fresh_confirmatory_v1/fresh_cache_audit.json",
             "paper/scripts/render_fresh_real_video.py"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "scripts/real_video_development").glob("fresh_evaluation*")) if path.is_file()]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video").glob("fresh_confirmatory*")) if path.is_file()]
relative += ["paper/scripts/render_fresh_mechanism.py"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video").glob("fresh_mechanism*")) if path.is_file()]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video/fresh_mechanism_assets").glob("*")) if path.is_file()]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video").glob("comparison*"))
             if path.is_file() and path.suffix in (".pdf", ".json", ".py", ".tex")]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/tables").glob("*.sources.json"))]
relative += ["reports/real_droid_generalization_protocol.md",
             "configs/real_video_development/generalization_v1/registration.json",
             "scripts/real_video_development/generalization_campaign.py",
             "scripts/real_video_development/finalize_generalization.py",
             "reports/real_droid_generalization_results.json",
             "reports/real_droid_generalization_finalization.json"]
relative += ["paper/scripts/render_reliability_development.py",
             "reports/real_video_development/reliability_blend_protocol.md",
             "reports/real_video_development/reliability_blend_results.json",
             "scripts/real_video_development/finalize_horizon10_paper.py",
             "reports/real_droid_horizon10_protocol.md",
             "reports/horizon10_paper_tables_protocol.md",
             "reports/real_droid_horizon10_results.json",
             "configs/real_video_development/horizon10_v1/registration.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video").glob("reliability_development*")) if path.is_file()]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video/horizon10").glob("*"))
             if path.is_file() and path.suffix in (".tex", ".json", ".pdf")]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video/generalization").glob("*"))
             if path.is_file() and path.suffix in (".tex", ".json", ".pdf")]
relative += ["paper/scripts/render_generalization_summary.py",
             "reports/generalization_summary_figure_protocol.md"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video").glob("generalization_summary*"))
             if path.is_file() and path.suffix in (".tex", ".json", ".pdf")]
relative += [str(path.relative_to(root)) for directory in ('paper/tables', 'paper/sections')
             for path in sorted((root / directory).glob('*.tex'))]
relative += [str(path.relative_to(root)) for path in sorted((root / 'paper/figures/assets').glob('*'))
             if path.is_file()]
relative += [str(path.relative_to(root)) for path in sorted((root / 'paper/figures/split_assets').glob('*'))
             if path.is_file()]
sources = {name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
           for name in relative if (root / name).is_file()}
record = {"built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "status": "development_manuscript", "sources": sources,
          "notice": "Training records are not planning results. Sources pin this build; no experimental conclusion is inferred."}
output = root / "paper/evidence/manuscript_sources.json"
output.write_text(json.dumps(record, indent=2) + "\n")
