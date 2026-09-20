"""Prevent blank result grids without rejecting plans or partial comparisons."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "paper/scripts/check_table_completeness.py"
SPEC = importlib.util.spec_from_file_location("table_completeness_guard", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


EMPTY = r"""\begin{table}
\caption{Planned mechanism and resource analysis.}\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Variant & Success $\uparrow$ & Error $\downarrow$ & ms/replan & Trainable M\\
\midrule
ShiftWM (ours) &\missing&\missing&\missing&\missing\\
Without observation context &\missing&\missing&\missing&\missing\\
Without dynamics context &\missing&\missing&\missing&\missing\\
Unpaired contexts &\missing&\missing&\missing&\missing\\
Shuffled dynamics support &\missing&\missing&\missing&\missing\\
Shared context, matched size &\missing&\missing&\missing&\missing\\
Joint gradient adaptation &\missing&\missing&\missing&\missing\\
\bottomrule\end{tabular}\end{table}
"""
GOOD = r"""\begin{table}\label{tab:good}
\begin{tabular}{lrr}\toprule Method & Success & MSE\\\midrule
Baseline & 0.0 & 0.2\\ Ours & 0.0 & 0.1\\\bottomrule\end{tabular}\end{table}
"""


def fixture(root, active, auxiliary=""):
    (root / "main.tex").write_text(active)
    fls = root / "main.fls"
    fls.write_text(f"PWD {root}\nINPUT main.tex\nINPUT ./main.tex\nOUTPUT main.pdf\n")
    aux = root / "main.aux"
    aux.write_text(auxiliary)
    return fls, aux


def inspect(source):
    block = next(guard.table_blocks(source))[1]
    return guard.inspect_table(block)


def test_seven_by_four_legacy_grid_is_rejected_even_when_caption_says_planned(tmp_path):
    fls, aux = fixture(tmp_path, EMPTY, r"\newlabel{tab:ablation}{{36}{49}}")
    result = guard.check(tmp_path, fls, aux, tmp_path / "main.tex")
    assert result["status"] == "failed"
    assert result["findings"][0]["rows"] == 7
    assert result["findings"][0]["empty_cells"] == 28
    assert result["active_tex_files"] == 1  # recorder aliases must not duplicate tables


def test_numeric_table_with_real_zero_success_is_allowed():
    assert inspect(GOOD) is None


def test_task_and_method_identifiers_do_not_hide_missing_measurements():
    source = r"""\begin{tabular}{llrr}\toprule Task & Method & MSE & MAE\\\midrule
    PushT & Baseline & \missing & \missing\\
    PushT & Ours & --- & ---\\\bottomrule\end{tabular}"""
    result = inspect(source)
    assert result["rows"] == 2 and result["empty_cells"] == 4
    assert result["metric_columns_zero_based"] == [2, 3]


def test_task_and_method_table_with_one_measured_metric_is_not_empty():
    source = r"""\begin{tabular}{llrr}\toprule Task & Method & MSE & MAE\\\midrule
    PushT & Baseline & 0.1 & \missing\\
    PushT & Ours & --- & ---\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_spanning_header_does_not_guess_metric_column_positions():
    source = r"""\begin{tabular}{lrr}\toprule & \multicolumn{2}{c}{Success}\\
    Method & Canonical & Held-out\\\midrule
    Ours & 0.1 & ---\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_partial_ci_cells_do_not_make_an_empty_table():
    source = r"""\begin{tabular}{lrrl}\toprule Method & MSE & Gain & CI\\\midrule
    Baseline & 0.2 & --- & ---\\ Ours & 0.1 & +50 & [-.12,-.08]\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_populated_external_planning_table_is_allowed():
    source = r"""\begin{tabular}{llll}\toprule
    Benchmark & External method & Target metrics & Execution status\\\midrule
    Drone & PPO & Success; physical error & Not run\\
    Tissue & LapGym PPO & Success; goal distance & Pending\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_status_table_with_pending_final_tests_is_allowed():
    source = r"""\begin{tabular}{lll}\toprule Stage & Drone & Tissue\\\midrule
    Training & Complete & Complete\\ Final test & Pending & Pending\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_header_only_empirical_table_is_rejected():
    source = r"\begin{tabular}{lr}\toprule Method & MSE\\\midrule\bottomrule\end{tabular}"
    assert inspect(source)["reason"] == "measurement header with no data rows"


def test_no_midrule_header_only_table_is_rejected():
    source = r"\begin{tabular}{lr}\hline Method & Latency\\\hline\end{tabular}"
    assert inspect(source)["rows"] == 0


@pytest.mark.parametrize("marker", [r"\missing", "---", "--", "—", "N/A", "{}", r"\textcolor{pending}{---}"])
def test_explicit_missing_metric_markers_are_recognized(marker):
    source = r"\begin{tabular}{lr}\toprule Method & MSE\\\midrule Ours & " + marker + r"\\\bottomrule\end{tabular}"
    assert inspect(source)["empty_cells"] == 1


def test_hyphens_in_meanings_are_not_global_missing_detection():
    source = r"""\begin{tabular}{ll}\toprule Method & Error definition\\\midrule
    Signed-error model & target-minus-prediction\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_inactive_archived_inputs_are_never_scanned(tmp_path):
    fls, aux = fixture(tmp_path, GOOD, r"\newlabel{tab:good}{{1}{1}}")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "unused.tex").write_text(EMPTY)
    result = guard.check(tmp_path, fls, aux, tmp_path / "main.tex")
    assert result["status"] == "passed" and result["active_tex_files"] == 1
    assert "archive/unused.tex" not in result["active_sources_sha256"]


def test_inactive_labeled_alternative_in_active_file_is_ignored(tmp_path):
    fls, aux = fixture(tmp_path, GOOD + r"\iffalse" + EMPTY + r"\fi", r"\newlabel{tab:good}{{1}{1}}")
    result = guard.check(tmp_path, fls, aux, tmp_path / "main.tex")
    assert result["status"] == "passed"
    assert len(result["ignored_inactive_alternatives"]) == 1


def test_commented_old_tables_are_ignored(tmp_path):
    commented = "\n".join("% " + line for line in EMPTY.splitlines())
    fls, aux = fixture(tmp_path, GOOD + commented, r"\newlabel{tab:good}{{1}{1}}")
    assert guard.check(tmp_path, fls, aux, tmp_path / "main.tex")["status"] == "passed"


def test_nested_linebreaks_and_escaped_ampersands_do_not_create_rows():
    source = r"""\begin{tabular}{lr}\toprule Method & MSE\\\midrule
    \shortstack{Method\\with\&parts} & 0.1\\\bottomrule\end{tabular}"""
    assert inspect(source) is None


def test_missing_main_in_recorder_is_a_hard_error(tmp_path):
    fls, aux = fixture(tmp_path, GOOD)
    fls.write_text(f"PWD {tmp_path}\n")
    with pytest.raises(ValueError, match="does not contain"):
        guard.check(tmp_path, fls, aux, tmp_path / "main.tex")


def test_guard_cli_blocks_publication_with_useful_diagnostics(tmp_path):
    fls, aux = fixture(tmp_path, EMPTY, r"\newlabel{tab:ablation}{{36}{49}}")
    output = tmp_path / "guard.json"
    run = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path), "--fls", str(fls),
                          "--aux", str(aux), "--output", str(output)], cwd=tmp_path, text=True, capture_output=True)
    assert run.returncode == 1
    assert "28 cells" in run.stderr
    assert json.loads(output.read_text())["status"] == "failed"


def test_build_runs_guard_before_public_pdf_copies_with_fresh_recorder():
    script = (PROJECT / "paper/build.sh").read_text()
    guard_position = script.index("python scripts/check_table_completeness.py")
    assert guard_position < script.index("cp build/main.pdf proposal.pdf")
    assert guard_position < script.index("cp build/main.pdf world_model_draft.pdf")
    assert script.count("pdflatex -recorder") == 3
    assert script.index("rm -f build/main.fls") < script.index("pdflatex -recorder")
