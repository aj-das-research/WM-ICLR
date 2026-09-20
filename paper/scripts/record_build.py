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
relative += ["paper/scripts/render_horizon10_summary.py",
             "scripts/real_video_spatial_reporting/finalize_paper.py",
             "reports/spatial_paper_protocol.md",
             "configs/real_video_spatial/v1/registration.json",
             "reports/real_video_spatial/finalization.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video/spatial").glob("*"))
             if path.is_file() and path.suffix in (".tex", ".json", ".pdf")]
relative += [str(path.relative_to(root)) for path in
             sorted((root / "paper/generated/real_video").glob("horizon10_summary*"))
             if path.is_file() and path.suffix in (".tex", ".json", ".pdf")]
relative += [str(path.relative_to(root)) for directory in ('paper/tables', 'paper/sections')
             for path in sorted((root / directory).rglob('*.tex'))]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/generated/real_video').glob('spatial_*'))
             if path.is_file() and path.suffix in ('.tex', '.json', '.pdf')]
relative += ["paper/scripts/render_spatial_versions.py", "paper/scripts/render_spatial_task.py", "paper/scripts/render_spatial_method.py",
             "scripts/real_video_spatial_qualitative/render.py",
             "paper/figure_sources/spatial_qualitative/manifest.json",
             "paper/figure_sources/spatial_qualitative/replay.json",
             "paper/figure_sources/spatial_qualitative/replay_arrays.npz"]
relative += ["paper/abstract.txt",
             "reports/real_video_spatial_components/finalization.json",
             "configs/real_video_spatial_components/v1/registration.json",
             "reports/real_video_development/spatial_components_protocol.md"]
relative += ["paper/scripts/render_teaser_story.py",
             "paper/scripts/teaser_story_glyphs.py",
             "paper/design/teaser_story/population.json",
             "paper/design/teaser_story/brief.md",
             "paper/design/story_style.json",
             "paper/scripts/render_spatial_architecture_story.py",
             "paper/design/spatial_architecture_story/brief.md",
             "paper/design/spatial_architecture_story/canonical_graph.json",
             "paper/scripts/render_spatial_architecture_refined.py",
             "paper/scripts/render_teaser_refined.py",
             "paper/scripts/render_results_refined.py",
             "paper/scripts/render_qualitative_compact.py",
             "paper/scripts/render_task_refined.py",
             "paper/scripts/render_mixing_diagnostics.py",
             "paper/scripts/render_task_story.py",
             "paper/scripts/render_mixing_story.py",
             "paper/design/task_story/brief.md",
             "paper/design/task_story/canonical_graph.json",
             "paper/design/mixing_story/brief.md",
             "paper/design/task_mixing_story/before_snapshot.json",
             "paper/scripts/render_spatial_main_architecture.py",
             "paper/scripts/render_anchoring_teaser.py",
             "paper/scripts/render_parallel_qualitative.py",
             "paper/scripts/render_editorial_tables.py",
             "paper/scripts/render_spatial_editorial.py",
             "paper/scripts/render_method_family.py",
             "paper/scripts/render_editorial_qualitative.py",
             "paper/scripts/render_editorial_task.py",
             "paper/design/editorial_style.json",
             "paper/design/refined_style.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/generated/editorial').glob('*'))
             if path.is_file() and path.suffix in ('.tex', '.json', '.pdf', '.txt')]
relative += ["paper/scripts/render_teaser_gallery.py",
             "paper/scripts/render_spatial_architecture_math.py",
             "paper/design/teaser_gallery/brief.md",
             "paper/design/spatial_architecture_math/brief.md",
             "paper/design/spatial_architecture_math/math_review.json",
             "reports/evidence/teaser_gallery_independent_review.json",
             "reports/evidence/spatial_architecture_math_independent_review.json",
             "reports/evidence/multitask_figures_public_reproduction.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/figure_sources/teaser_gallery').rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.png')]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/figure_sources/spatial_method').glob('*'))
             if path.is_file() and path.suffix in ('.drawio', '.json', '.md', '.py')]
relative += ["paper/scripts/render_teaser_cinematic.py",
             "paper/scripts/render_teaser_benchmarks.py",
             "paper/scripts/render_teaser_sources.py",
             "paper/scripts/render_teaser_visual_design.py",
             "paper/scripts/render_teaser_camera_story.py",
             "paper/scripts/render_architecture_visual_design.py",
             "paper/scripts/render_spatial_architecture_exploded.py",
             "paper/scripts/export_drawio_browser.py",
             "paper/generated/editorial/teaser_cinematic.drawio",
             "paper/generated/editorial/teaser_benchmarks.drawio",
             "paper/generated/editorial/teaser_sources.drawio",
             "paper/generated/editorial/teaser_visual_design.drawio",
             "paper/generated/editorial/teaser_camera_story.drawio",
             "reports/evidence/teaser_camera_story_independent_review.json",
             "reports/evidence/teaser_camera_story_public_reproduction.json",
             "reports/evidence/teaser_camera_story_native_review.json",
             "reports/evidence/teaser_camera_story_integrated_review.json",
             "reports/evidence/task_story_removal_review.json",
             "paper/figure_sources/architecture_visual_design/architecture-visual-design.drawio",
             "paper/evidence/paper_visual_design_application.json",
             "reports/evidence/architecture_visual_design_author_review.json",
             "reports/evidence/paper_visual_design_independent_review.json",
             "reports/evidence/paper_visual_design_public_reproduction.json",
             "reports/evidence/paper_visual_design_native_review.json",
             "reports/evidence/paper_visual_design_integrated_review.json",
             "reports/evidence/teaser_cinematic_independent_review.json",
             "reports/evidence/spatial_architecture_exploded_independent_review.json",
             "reports/evidence/cinematic_exploded_native_review.json",
             "reports/evidence/spatial_architecture_exploded_native_portability_review.json",
             "reports/evidence/teaser_cinematic_public_reproduction.json",
             "reports/evidence/teaser_benchmarks_independent_review.json",
             "reports/evidence/teaser_benchmarks_public_reproduction.json",
             "reports/evidence/benchmark_gallery_inventory_review.json",
             "reports/evidence/benchmark_teaser_native_review.json",
             "reports/evidence/benchmark_teaser_integrated_review.json",
             "reports/evidence/teaser_sources_independent_review.json",
             "reports/evidence/teaser_sources_public_reproduction.json",
             "reports/evidence/teaser_sources_native_review.json",
             "reports/evidence/teaser_sources_integrated_review.json",
             "reports/evidence/cinematic_exploded_integrated_review.json",
             "reports/evidence/iws_plot_portability_maintenance_20260919.json",
             "paper/scripts/render_iws_results.py",
             "paper/figure_sources/iws_results/brief.md"]
relative += [str(path.relative_to(root)) for directory in
             ('teaser_cinematic', 'teaser_benchmarks', 'teaser_sources', 'teaser_visual_design', 'teaser_camera_story', 'architecture_visual_design', 'benchmark_gallery', 'spatial_architecture_exploded', 'visual_story_references_v1', 'forecast_comparison_reference_v2')
             for path in sorted((root / 'paper/figure_sources' / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.drawio', '.json', '.md', '.png', '.svg', '.py', '.txt')]
relative += [str(path.relative_to(root)) for path in sorted((root / 'paper/figures/assets').glob('*'))
             if path.is_file()]
relative += ["paper/scripts/render_mixing_multibenchmark.py",
             "paper/scripts/render_decoder_evidence.py",
             "paper/scripts/render_composition_landscape.py",
             "paper/evidence/figure6_12_visual_design_application.json",
             "reports/evidence/decoder_evidence_independent_review.json",
             "reports/evidence/composition_landscape_independent_review.json",
             "reports/evidence/composition_landscape_native_review.json",
             "reports/evidence/figure6_12_public_reproduction.json",
             "reports/evidence/figure6_12_integrated_review.json",
             "paper/evidence/mixing_visual_design_application.json",
             "reports/evidence/mixing_multibenchmark_independent_review.json",
             "reports/evidence/mixing_multibenchmark_public_reproduction.json",
             "reports/evidence/mixing_multibenchmark_integrated_review.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/figure_sources/mixing_multibenchmark').rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.py', '.png', '.txt')]
relative += [str(path.relative_to(root)) for directory in ('decoder_evidence', 'composition_landscape')
             for path in sorted((root / 'paper/figure_sources' / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.py', '.png', '.txt', '.drawio')]
relative += [str(path.relative_to(root)) for path in sorted((root / 'paper/figures/split_assets').glob('*'))
             if path.is_file()]
relative += ["paper/scripts/render_recorded_forecast_compact.py",
             "paper/scripts/render_validation_controls_compact.py",
             "paper/scripts/render_composition_compact.py",
             "paper/scripts/render_reacher_outcomes_compact.py",
             "paper/evidence/compact_appendix_visual_design_application.json",
             "reports/evidence/recorded_forecast_compact_independent_review.json",
             "reports/evidence/validation_controls_compact_independent_review.json",
             "reports/evidence/composition_compact_independent_review.json",
             "reports/evidence/composition_compact_native_review.json",
             "reports/evidence/reacher_outcomes_compact_independent_review.json",
             "reports/evidence/compact_appendix_public_reproduction.json",
             "reports/evidence/validation_controls_compact_author_review.json",
             "reports/evidence/reacher_outcomes_compact_public_reproduction.json",
             "reports/evidence/compact_appendix_secondary_integrated_review.json",
             "reports/evidence/compact_appendix_integrated_review.json"]
relative += [str(path.relative_to(root)) for directory in
             ('recorded_forecast_compact', 'validation_controls_compact',
              'composition_compact', 'reacher_outcomes_compact')
             for path in sorted((root / 'paper/figure_sources' / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.py', '.png', '.txt', '.drawio', '.tex')]
relative += ["paper/evidence/remaining_compaction_visual_design_application.json",
             "reports/evidence/remaining_compaction_public_reproduction.json",
             "reports/evidence/remaining_compaction_integrated_review.json",
             "reports/evidence/remaining_compaction_secondary_integrated_review.json"]
for compact_name in ('spatial_ablation_compact', 'positive_cases_compact',
                     'technical_story_compact', 'pusht_outcomes_compact'):
    relative += [f"paper/scripts/render_{compact_name}.py",
                 f"reports/evidence/{compact_name}_independent_review.json"]
    relative += [str(path.relative_to(root)) for path in
                 sorted((root / 'paper/figure_sources' / compact_name).rglob('*'))
                 if path.is_file() and path.suffix in ('.json', '.md', '.py', '.png', '.txt', '.tex')]
relative += ["paper/scripts/render_droid_forecast_setup.py",
             "paper/evidence/droid_forecast_setup_visual_design_application.json",
             "reports/evidence/droid_forecast_setup_independent_review.json",
             "reports/evidence/droid_forecast_setup_native_review.json",
             "reports/evidence/droid_forecast_setup_public_reproduction.json",
             "reports/evidence/droid_forecast_setup_integrated_review.json"]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/figure_sources/droid_forecast_setup').rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.png', '.txt', '.tex', '.drawio')]
relative += ["paper/scripts/render_iws_main_summary.py",
             "paper/scripts/render_iws_results_v2.py",
             "reports/evidence/iws_results_reporting_review.json",
             "reports/evidence/iws_results_display_v2_review.json",
             "reports/evidence/iws_completed_manuscript_review.json",
             "paper/scripts/render_iws_unbounded_results.py",
             "paper/tests/test_iws_unbounded_reporting.py",
             "reports/evidence/iws_unbounded_reporting_author_review.json",
             "reports/evidence/iws_unbounded_reporting_independent_review.json",
             "reports/evidence/iws_completed_integrated_review.json",
             "reports/evidence/iws_interim_editorial_update.json",
             "reports/evidence/iws_completion_resume_audit.json",
             "reports/real_video_iws/next_evaluation_gate_audit.md",
             "reports/real_video_iws/development_diagnosis_plan.md",
             "reports/real_video_iws/diagnostics/training_envelope_v1.json",
             "reports/real_video_iws/release/README.md",
             "reports/real_video_iws/release/local_inference_export.json",
             "reports/real_video_iws/release/relocated_cpu_parity.json",
             "reports/real_video_iws/release/readiness_checks.json"]
relative += [str(path.relative_to(root)) for folder in ('iws_results', 'iws_results_v2') for path in
             sorted((root / 'paper/figure_sources' / folder).rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.py')]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/generated/experiment_alignment').glob('*'))
             if path.is_file() and path.suffix in ('.json', '.tex', '.pdf')]
relative += ["paper/sections/comparison_inventory.tex",
             "paper/scripts/render_comparison_inventory.py",
             "paper/scripts/render_current_real_scorecards.py",
             "paper/scripts/render_simulator_tables.py",
             "paper/scripts/render_iws_variant_forecasts.py",
             "scripts/publishing/refresh_comparison_presentations.py",
             "reports/benchmark_metrics_and_comparators_2026-09-20.md",
             "reports/evidence/comparison_scorecards_integrated_review.json",
             "reports/evidence/comparison_scorecards_independent_review.json",
             "reports/evidence/comparison_presentation_refresh.json",
             "reports/evidence/iws_variant_plot_automation_independent_review.json",
             "paper/tests/test_current_real_scorecards.py",
             "paper/tests/test_iws_variant_forecasts.py",
             "tests/test_simulator_presentation_tables.py",
             "tests/test_comparison_presentation_refresh.py"]
relative += ["paper/build.sh", "paper/scripts/check_table_completeness.py",
             "tests/test_table_completeness_guard.py",
             "reports/evidence/empty_table_repair_2026-09-20.json",
             "reports/evidence/iws_unbounded_completed_independent_review.json"]
relative += [str(path.relative_to(root))
             for directory in ('scripts/real_video_iws_unbounded_release',
                               'reports/real_video_iws_unbounded/release')
             for path in sorted((root / directory).glob('*'))
             if path.is_file() and path.suffix in ('.py', '.json', '.md')]
relative += [str(path.relative_to(root)) for path in
             sorted((root / 'paper/generated/benchmark_scorecards').glob('*'))
             if path.is_file() and path.suffix in ('.json', '.tex', '.csv')]
relative += [str(path.relative_to(root)) for folder in
             ('paper/generated/simulator_tables', 'paper/table_sources/simulator_tables',
              'paper/generated/iws_variant_forecasts')
             for path in sorted((root / folder).rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.tex', '.md', '.csv', '.pdf', '.svg', '.png')]
relative += [str(path.relative_to(root)) for folder in ('current_real_scorecards', 'simulation_scorecards')
             for path in sorted((root / 'paper/figure_sources' / folder).rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.md', '.py', '.csv')]
relative += ["paper/scripts/render_iws_compact_evidence.py",
             "paper/scripts/render_iws_predictor_resources.py",
             "tests/test_iws_predictor_resources_table.py",
             "paper/scripts/render_landscape_evidence_v1.py",
             "paper/scripts/render_validation_controls_landscape_v2.py",
             "paper/tests/test_iws_compact_evidence.py",
             "reports/evidence/iws_compact_evidence_independent_review.json",
             "reports/evidence/landscape_figures_independent_review_2026-09-20.json",
             "reports/evidence/table_gap_audit_2026-09-20.json",
             "reports/pending_results_execution_plan_2026-09-20.md",
             "reports/evidence/compact_evidence_integrated_review_2026-09-20.json",
             "reports/evidence/compact_evidence_independent_layout_review_2026-09-20.json",
             "reports/evidence/iws_predictor_resources_independent_review_2026-09-20.json"]
relative += [str(path.relative_to(root)) for directory in
             ('paper/figure_sources/iws_compact_evidence',
              'paper/figure_sources/landscape_evidence_v1',
              'paper/figure_sources/validation_controls_landscape_v2',
              'paper/generated/iws_compact_evidence',
              'paper/design/iws_compact_evidence',
              'paper/design/validation_controls_landscape_v2')
             for path in sorted((root / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.json', '.tex', '.md', '.pdf', '.svg', '.png')]
relative += [str(path.relative_to(root)) for directory in
             ('paper/table_sources/iws_predictor_resources_v1', 'paper/generated/iws_resources',
              'scripts/real_video_iws_resources_v1', 'configs/real_video_iws_resources_v1',
              'reports/real_video_iws_resources_v1', 'paper/design/iws_resources_v1')
             for path in sorted((root / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.py', '.json', '.md', '.tex', '.slurm', '.png')]
relative += ["paper/scripts/render_iws_reserved_evidence.py", "docs/IWS_RESERVED_EVALUATION.md",
             "reports/evidence/iws_reserved_presentation_independent_review_2026-09-20.json",
             "reports/evidence/iws_reserved_integrated_review_2026-09-20.json"]
relative += [str(path.relative_to(root)) for directory in
             ('scripts/extensions_diagnostics_20260920', 'reports/extensions_diagnostics_20260920')
             for path in sorted((root / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.py', '.json', '.md', '.slurm')]
relative += [str(path.relative_to(root)) for directory in
             ('paper/figure_sources/iws_reserved_evidence', 'paper/generated/iws_reserved_evidence',
              'paper/design/iws_reserved_evidence', 'configs/real_video_iws_reserved_v1',
              'reports/real_video_iws_reserved_v1', 'scripts/real_video_iws_reserved_v1',
              'src/shiftwm/real_video_iws_reserved',
              'configs/real_video_iws_reserved_recovery_v2', 'reports/real_video_iws_reserved_recovery_v2',
              'scripts/real_video_iws_reserved_recovery_v2', 'src/shiftwm/real_video_iws_reserved_recovery',
              'scripts/real_video_iws_reserved_diagnostics_20260920',
              'reports/real_video_iws_reserved_diagnostics_20260920')
             for path in sorted((root / directory).rglob('*'))
             if path.is_file() and path.suffix in ('.py', '.json', '.md', '.tex', '.slurm', '.pdf', '.svg', '.png')]
sources = {name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
           for name in relative if (root / name).is_file()}
record = {"built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "status": "development_manuscript", "sources": sources,
          "notice": "Training records are not planning results. Sources pin this build; no experimental conclusion is inferred."}
output = root / "paper/evidence/manuscript_sources.json"
output.write_text(json.dumps(record, indent=2) + "\n")
