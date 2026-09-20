#!/usr/bin/env python3
r"""Reject wholly unfilled empirical tables in the recorder's active TeX inputs.

This is a presentation guard, not a numerical/scientific validator. It recognizes
explicit empty cells (\missing, dashes and N/A) under measurement headings. It
does not scan archived files or treat arbitrary hyphens in prose as missing data.
Compiled auxiliary labels disambiguate labeled inactive alternatives in an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ENVIRONMENT = re.compile(r"\\begin\{(tabular\*?|longtable)\}")
METRIC = re.compile(
    r"\b(success|error|mse|mae|rmse|accuracy|latency|memory|parameters?|trainable|"
    r"psnr|ssim|lpips|fid|throughput|gpu[ -]?hours?|ms/replan|score|gain|ci|interval)\b", re.I)
RULE = re.compile(r"\\(?:toprule|midrule|bottomrule|hline|addlinespace)(?:\[[^\]]*\])?"
                  r"|\\(?:cmidrule|cline)(?:\([^)]*\))?\{[^}]*\}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomment(text: str) -> str:
    # Preserve newlines for source locations; an escaped percent is visible text.
    return re.sub(r"(?<!\\)(?:\\\\)*%[^\n]*", lambda m: " " * len(m[0]), text)


def brace_end(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected braced TeX table argument")
    depth = 0
    for i in range(start, len(text)):
        if i and text[i - 1] == "\\":
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("Unbalanced TeX table argument")


def table_blocks(text: str):
    for match in ENVIRONMENT.finditer(text):
        env = match[1]
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] == "[":
            pos = text.index("]", pos) + 1
        for _ in range(2 if env == "tabular*" else 1):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            pos = brace_end(text, pos)
        end = text.find(r"\end{" + env + "}", pos)
        if end < 0:
            raise ValueError(f"Unclosed {env} environment")
        yield match.start(), text[pos:end]


def split_top_level(text: str, separator: str) -> list[str]:
    r"""Do not split \shortstack line breaks or escaped \& inside a cell."""
    parts, start, depth, i = [], 0, 0, 0
    while i < len(text):
        if depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i += len(separator)
            if separator == r"\\":
                suffix = re.match(r"(?:\*)?(?:\[[^\]]*\])?", text[i:])
                i += suffix.end()
            start = i
            continue
        if text[i] == "\\":
            # A control symbol consumes the escaped character, including \&.
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth = max(0, depth - 1)
        i += 1
    parts.append(text[start:])
    return parts


def visible(cell: str) -> str:
    cell = RULE.sub("", cell)
    cell = re.sub(r"\\(?:textcolor|colorbox)\s*\{[^}]*\}", "", cell)
    cell = re.sub(r"\\(?:color|rowcolor|cellcolor)\s*\{[^}]*\}", "", cell)
    cell = re.sub(r"\\missing\b", "MISSINGCELL", cell)
    cell = re.sub(r"\\(?:textemdash|textendash|emdash|endash)\b", "---", cell)
    cell = re.sub(r"\\(?:textbf|textit|textrm|textsf|texttt|emph|ensuremath)\b", "", cell)
    cell = re.sub(r"\\[A-Za-z@]+\*?", " ", cell)
    return re.sub(r"[{}$\s]", "", cell).strip()


def empty_cell(cell: str) -> bool:
    token = visible(cell)
    return token in ("", "MISSINGCELL", "--", "---", "—", "–", "N/A", "NA")


def labeled_float(text: str, position: int) -> list[str]:
    opens = list(re.finditer(r"\\begin\{table\*?\}", text[:position]))
    if not opens:
        return []
    start = opens[-1].start()
    close = re.search(r"\\end\{table\*?\}", text[start:])
    if not close or start + close.start() < position:
        return []
    return re.findall(r"\\label\{([^}]+)\}", text[start:start + close.end()])


def inspect_table(body: str) -> dict | None:
    boundary = re.search(r"\\midrule\b", body)
    if boundary:
        header, data = body[:boundary.start()], body[boundary.end():]
    else:
        rows = split_top_level(body, r"\\")
        header_rows = next((i for i, row in enumerate(rows) if len(split_top_level(row, "&")) >= 2), None)
        if header_rows is None:
            return None
        header = "\n".join(rows[:header_rows + 1])
        data = r"\\".join(rows[header_rows + 1:])
    # A table of plan/status descriptions has no numerical measurement headings.
    if not METRIC.search(header):
        return None
    # Task/backbone/method identifiers are not measurements. For an ordinary
    # one-line header, inspect the actual metric columns, not everything after
    # the first identifier. Spanning/multiline headings retain the conservative
    # all-nonlabel-cell check rather than guessing column positions.
    header_rows = [split_top_level(row, "&") for row in split_top_level(header, r"\\")]
    header_rows = [row for row in header_rows if len(row) >= 2]
    metric_columns = None
    if len(header_rows) == 1 and not re.search(r"\\(?:multicolumn|multirow)\b", header):
        metric_columns = [i for i, cell in enumerate(header_rows[0]) if METRIC.search(cell)]
    # Longtable continuation headings/footers are not experimental rows.
    boundaries = list(re.finditer(r"\\end(?:firsthead|head|foot|lastfoot)\b", data))
    if boundaries:
        data = data[boundaries[-1].end():]
    data_rows = []
    for row in split_top_level(data, r"\\"):
        cells = split_top_level(row, "&")
        if len(cells) >= 2:
            data_rows.append(cells)
    if not data_rows:
        # A table delegating rows to an active generated input is inspected in
        # that file; do not call its wrapper a header-only table.
        if re.search(r"\\(?:input|include|InputIfFileExists)\b", data):
            return None
        return {"reason": "measurement header with no data rows", "rows": 0, "empty_cells": 0}
    measurements = [[cells[i] if i < len(cells) else "" for i in metric_columns]
                    if metric_columns is not None else cells[1:] for cells in data_rows]
    if all(all(empty_cell(cell) for cell in cells) for cells in measurements):
        return {"reason": "every empirical data cell is an explicit placeholder",
                "rows": len(data_rows), "empty_cells": sum(len(cells) for cells in measurements),
                "metric_columns_zero_based": metric_columns,
                "row_labels": [visible(cells[0]) for cells in data_rows]}
    return None


def active_inputs(root: Path, recorder: Path, main: Path) -> list[Path]:
    lines = recorder.read_text().splitlines()
    pwd = next((Path(line[4:]).resolve() for line in lines if line.startswith("PWD ")), None)
    if pwd != root:
        raise ValueError("Recorder PWD does not match the manuscript root")
    paths = set()
    for line in lines:
        if not line.startswith("INPUT ") or not line.endswith(".tex"):
            continue
        path = (pwd / line[6:]).resolve()
        if path.is_relative_to(root):
            if not path.is_file():
                raise ValueError(f"Recorded TeX input is missing: {path}")
            paths.add(path)
    if main.resolve() not in paths:
        raise ValueError("Recorder does not contain the requested main manuscript")
    return sorted(paths)


def check(root: Path, recorder: Path, auxiliary: Path, main: Path) -> dict:
    root = root.resolve()
    files = active_inputs(root, recorder, main)
    aux_text = auxiliary.read_text()
    compiled_labels = set(re.findall(r"\\newlabel\{([^}]+)\}", aux_text))
    findings, ignored, examined = [], [], 0
    for path in files:
        source = uncomment(path.read_text())
        for position, body in table_blocks(source):
            examined += 1
            finding = inspect_table(body)
            if finding is None:
                continue
            labels = labeled_float(source, position)
            entry = {"source": str(path.relative_to(root)), "line": source[:position].count("\n") + 1,
                     "labels": labels, **finding}
            if labels and not any(label in compiled_labels for label in labels):
                ignored.append({**entry, "reason": "labeled alternative is absent from compiled auxiliary output"})
                continue
            findings.append(entry)
    return {"schema_version": 1, "status": "failed" if findings else "passed",
            "scope": "Active local TeX inputs only; explicit empty empirical tables, not scientific metric validation.",
            "active_tex_files": len(files), "table_bodies_examined": examined,
            "findings": findings, "ignored_inactive_alternatives": ignored,
            "recorder_sha256": sha(recorder), "auxiliary_sha256": sha(auxiliary),
            "guard_sha256": sha(Path(__file__)),
            "active_sources_sha256": {str(path.relative_to(root)): sha(path) for path in files}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fls", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--main", type=Path, default=Path("main.tex"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.root, args.fls, args.aux, args.root / args.main)
    except (OSError, ValueError) as error:
        print(f"Table completeness guard could not validate this build: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["findings"]:
        for item in report["findings"]:
            print(f"Unfilled empirical table: {item['source']}:{item['line']}: {item['reason']} "
                  f"({item['rows']} rows, {item['empty_cells']} cells)", file=sys.stderr)
        return 1
    print(f"Table completeness: passed ({report['table_bodies_examined']} active table bodies; "
          f"{len(report['ignored_inactive_alternatives'])} inactive alternatives ignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
