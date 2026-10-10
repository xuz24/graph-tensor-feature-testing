#!/usr/bin/env python3
"""Convert 0.out-11.out experiment directories to Markdown tables.

Fault tolerant: a log that is missing, crashed, or badly formatted is skipped instead of aborting
the whole directory. The report is still written from the remaining logs, with a warning block at
the top naming each problem file (and its fold / seed when the log reveals them). Averages are
computed over the runs that were usable. A directory is only reported as failed if NO log in it
could be parsed.
"""

import argparse
from collections import Counter
from decimal import Decimal
from pathlib import Path
import re
import sys


COLUMNS = ("auroc", "ap", "wq_auroc", "wq_mrr", "wq_mrr_norm", "n_queries")
METADATA = re.compile(r"setup=(\w+)\s+fold=(\d+)\s+seed=(\d+)")
SPLIT_METADATA = re.compile(r"^(\w+) fold (\d+) seed (\d+):", re.MULTILINE)
N_FILES, N_FOLDS, N_SEEDS = 12, 4, 3


def find_labels(text):
    """(setup, fold, seed) if the log carries exactly one consistent label, else None."""
    labels = set(METADATA.findall(text) + SPLIT_METADATA.findall(text))
    return next(iter(labels)) if len(labels) == 1 else None


def parse_text(text):
    labels = find_labels(text)
    if labels is None:
        raise ValueError("missing or conflicting setup/fold/seed labels")
    setup, fold, seed = labels
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines)
               if line.split()[:2] == ["arm", "auroc"]]
    if len(headers) != 1:
        raise ValueError(f"expected exactly one final results table, found {len(headers)}")
    start = headers[0]
    header = lines[start].split()
    if not set(COLUMNS).issubset(header):
        raise ValueError("results table is missing required columns")
    rows = {}
    for line in lines[start + 1:]:
        if not line.strip():
            break
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError(f"malformed results row: {line.strip()}")
        arm = fields[0]
        if arm in rows:
            raise ValueError(f"duplicate arm {arm}")
        try:
            values = tuple(Decimal(fields[header.index(col)].replace(",", ""))
                           for col in COLUMNS)
        except ArithmeticError as exc:
            raise ValueError(f"invalid numeric results: {line.strip()}") from exc
        if not all(v.is_finite() for v in values):
            raise ValueError(f"non-finite value in results row: {line.strip()}")
        count = values[-1]
        if count < 0 or count != count.to_integral_value():
            raise ValueError(f"invalid n_queries: {count}")
        rows[arm] = values
    if not rows:
        raise ValueError("empty results table")
    return setup, int(fold), int(seed), rows


def where(labels):
    return f"fold {labels[1]}, seed {labels[2]}" if labels else "fold/seed unknown"


def load_run(directory):
    """Return (setup, arms, runs, issues).

    runs   : {(fold, seed): {arm: values}} for every usable log
    issues : [(filename, "fold F, seed S" | "fold/seed unknown", reason)]
    """
    issues = []
    good = []  # (index, setup, fold, seed, rows)

    extra = sorted(p.name for p in directory.glob("*.out")
                   if p.stem.isdigit() and int(p.stem) >= N_FILES)
    for name in extra:
        issues.append((name, "fold/seed unknown", "unexpected extra file (ignored)"))

    for i in range(N_FILES):
        name = f"{i}.out"
        path = directory / name
        if not path.is_file():
            issues.append((name, "fold/seed unknown", "file is missing"))
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            issues.append((name, "fold/seed unknown", f"could not be read: {exc}"))
            continue
        try:
            setup, fold, seed, rows = parse_text(text)
        except ValueError as exc:
            reason = str(exc)
            if "Traceback" in text:
                reason += " (log contains a Python traceback; the run likely crashed)"
            issues.append((name, where(find_labels(text)), reason))
            continue
        good.append((i, setup, fold, seed, rows))

    if not good:
        details = "; ".join(f"{n}: {r}" for n, _, r in issues)
        raise ValueError(f"{directory}: no usable logs. {details}")

    # Reference setup and arm set = the most common among usable logs, so one odd file
    # is the one that gets flagged rather than the whole directory.
    setup = Counter(g[1] for g in good).most_common(1)[0][0]
    arm_sets = Counter(frozenset(g[4]) for g in good)
    ref_set = arm_sets.most_common(1)[0][0]
    arms = next(list(g[4]) for g in good if frozenset(g[4]) == ref_set)

    runs = {}
    for i, label, fold, seed, rows in good:
        name, loc = f"{i}.out", f"fold {fold}, seed {seed}"
        if label != setup:
            issues.append((name, loc, f"setup {label} differs from {setup}"))
        elif frozenset(rows) != ref_set:
            issues.append((name, loc, "arm set differs from other logs"))
        elif not (0 <= fold < N_FOLDS and 0 <= seed < N_SEEDS):
            issues.append((name, loc, "fold/seed outside the expected range"))
        elif (fold, seed) in runs:
            issues.append((name, loc, "duplicate fold/seed (an earlier file was kept)"))
        else:
            runs[fold, seed] = rows

    issues.sort(key=lambda t: (len(t[0]), t[0]))
    return setup, arms, runs, issues


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


def disclaimer(issues, runs):
    lines = [f"> **WARNING: {len(issues)} problem(s) found in the log files. "
             f"Averages below use only the {len(runs)} usable run(s).**",
             ">"]
    for name, loc, reason in issues:
        lines.append(f"> - `{name}` ({loc}): {reason.replace(chr(10), ' ')}")
    missing = sorted({(f, s) for f in range(N_FOLDS) for s in range(N_SEEDS)} - set(runs))
    if missing:
        lines += [">", "> Missing from the tables: "
                  + ", ".join(f"fold {f} seed {s}" for f, s in missing) + "."]
    return "\n".join(lines)


def render_run(directory):
    setup, arms, runs, issues = load_run(directory)
    parts = [f"# {setup}"]
    if issues:
        parts.append(disclaimer(issues, runs))

    for fold in sorted({f for f, _ in runs}):
        seeds = sorted(s for f, s in runs if f == fold)
        parts.append(f"## Fold {fold}")
        for seed in seeds:
            parts.extend([f"### Seed {seed}:", table(runs[fold, seed], arms)])
        parts.extend([f"### Fold {fold} averages (avg over seeds {','.join(map(str, seeds))}):",
                      table(average([runs[fold, s] for s in seeds], arms), arms, True)])

    parts.extend(["## All fold averages:", "By seed (averaged across the available folds):"])
    for seed in sorted({s for _, s in runs}):
        folds = sorted(f for f, s in runs if s == seed)
        parts.extend([f"### Seed {seed} (folds {','.join(map(str, folds))}):",
                      table(average([runs[f, seed] for f in folds], arms), arms, True)])
    if len(runs) == N_FOLDS * N_SEEDS:
        title = "### Grand average (all 4 folds x all 3 seeds, 12 runs):"
    else:
        title = f"### Grand average (only {len(runs)} of {N_FOLDS * N_SEEDS} runs):"
    parts.extend([title, table(average(list(runs.values()), arms), arms, True)])
    return "\n\n".join(parts) + "\n", issues


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
            report, issues = render_run(directory)
            output.mkdir(parents=True, exist_ok=True)
            destination = output / f"{directory.name}.md"
            destination.write_text(report)
            print(destination)
            for name, loc, reason in issues:
                print(f"Warning: {directory.name}/{name} ({loc}): {reason}", file=sys.stderr)
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())