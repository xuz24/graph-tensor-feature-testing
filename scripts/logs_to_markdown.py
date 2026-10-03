#!/usr/bin/env python3
"""Convert complete 0.out–11.out experiment directories to Markdown tables."""

import argparse
from decimal import Decimal
from pathlib import Path
import re
import sys


COLUMNS = ("auroc", "ap", "wq_auroc", "wq_mrr", "wq_mrr_norm", "n_queries")
METADATA = re.compile(r"setup=(\w+)\s+fold=(\d+)\s+seed=(\d+)")
SPLIT_METADATA = re.compile(r"^(\w+) fold (\d+) seed (\d+):", re.MULTILINE)


def parse_log(path):
    text = path.read_text()
    labels = METADATA.findall(text) + SPLIT_METADATA.findall(text)
    if not labels or len(set(labels)) != 1:
        raise ValueError(f"{path}: missing or conflicting setup/fold/seed labels")
    setup, fold, seed = labels[0]
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines)
               if line.split()[:2] == ["arm", "auroc"]]
    if len(headers) != 1:
        raise ValueError(f"{path}: expected exactly one final results table")
    start = headers[0]
    header = lines[start].split()
    if not set(COLUMNS).issubset(header):
        raise ValueError(f"{path}: results table is missing required columns")
    rows = {}
    for line in lines[start + 1:]:
        if not line.strip():
            break
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError(f"{path}: malformed results row: {line}")
        arm = fields[0]
        if arm in rows:
            raise ValueError(f"{path}: duplicate arm {arm}")
        try:
            values = tuple(Decimal(fields[header.index(col)].replace(",", ""))
                           for col in COLUMNS)
        except ArithmeticError as exc:
            raise ValueError(f"{path}: invalid numeric results: {line}") from exc
        count = values[-1]
        if not count.is_finite() or count < 0 or count != count.to_integral_value():
            raise ValueError(f"{path}: invalid n_queries: {count}")
        rows[arm] = values
    if not rows:
        raise ValueError(f"{path}: empty results table")
    return setup, int(fold), int(seed), rows


def load_run(directory):
    expected = {f"{i}.out" for i in range(12)}
    actual = {p.name for p in directory.glob("*.out") if p.stem.isdigit()}
    if actual != expected:
        raise ValueError(f"{directory}: expected 0.out through 11.out; "
                         f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    runs = {}
    setup = None
    arms = None
    for i in range(12):
        path = directory / f"{i}.out"
        label, fold, seed, rows = parse_log(path)
        if setup is not None and label != setup:
            raise ValueError(f"{path}: setup {label} differs from {setup}")
        if arms is not None and set(rows) != set(arms):
            raise ValueError(f"{path}: arm set differs from other logs")
        if (fold, seed) in runs:
            raise ValueError(f"{path}: duplicate fold {fold}, seed {seed}")
        setup = label
        arms = arms or list(rows)
        runs[fold, seed] = rows
    if set(runs) != {(f, s) for f in range(4) for s in range(3)}:
        raise ValueError(f"{directory}: expected every combination of folds 0–3 and seeds 0–2")
    return setup, arms, runs


def average(tables, arms):
    return {arm: tuple(sum(table[arm][i] for table in tables) / len(tables)
                       for i in range(len(COLUMNS))) for arm in arms}


def table(rows, arms, averaged=False):
    lines = ["| arm | " + " | ".join(COLUMNS) + " |",
             "| --- | " + " | ".join(["---:"] * len(COLUMNS)) + " |"]
    for arm in arms:
        values = [f"{value:.{1 if averaged else 0}f}" if col == "n_queries"
                  else f"{value:.3f}" for col, value in zip(COLUMNS, rows[arm])]
        lines.append("| " + " | ".join([arm, *values]) + " |")
    return "\n".join(lines)


def render_run(directory):
    setup, arms, runs = load_run(directory)
    parts = [f"# {setup}"]
    for fold in range(4):
        parts.append(f"## Fold {fold}")
        for seed in range(3):
            parts.extend([f"### Seed {seed}:", table(runs[fold, seed], arms)])
        parts.extend([f"### Fold {fold} averages (avg over seeds 0,1,2):",
                      table(average([runs[fold, s] for s in range(3)], arms), arms, True)])
    parts.extend(["## All fold averages:", "By seed (averaged across all 4 folds):"])
    for seed in range(3):
        parts.extend([f"### Seed {seed}:",
                      table(average([runs[f, seed] for f in range(4)], arms), arms, True)])
    parts.extend(["### Grand average (all 4 folds x all 3 seeds, 12 runs):",
                  table(average(list(runs.values()), arms), arms, True)])
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("logs"),
                        help="a run directory or a parent containing run directories (default: logs)")
    parser.add_argument("--output-dir", type=Path,
                        help="write <run-name>.md here (default: <input>/markdown)")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"not a directory: {args.input}")
    directories = ([args.input] if any(args.input.glob("*.out")) else
                   sorted({p.parent for p in args.input.rglob("*.out") if p.stem.isdigit()}))
    if not directories:
        parser.error(f"no run directories found in {args.input}")
    names = [p.name for p in directories]
    if len(names) != len(set(names)):
        parser.error("run directory names must be unique; process duplicate names separately")
    output = args.output_dir or args.input / "markdown"
    failed = False
    for directory in directories:
        try:
            report = render_run(directory)
            output.mkdir(parents=True, exist_ok=True)
            destination = output / f"{directory.name}.md"
            destination.write_text(report)
            print(destination)
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
