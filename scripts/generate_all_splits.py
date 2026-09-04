"""Generate every evaluation split after overlaying this update on the earlier data bundle.

`splits.py` beside this file is a VERBATIM copy of `dpsyn/splits.py` — not an adaptation — so the
splits produced here are the pipeline's splits, not lookalikes. `verify_update.py` checks the full
four-fold x three-seed split inventory against the expected current sizes.

    python generate_all_splits.py                       # summary table for every setting
    python generate_all_splits.py --out /tmp/splits     # + one .npz per (setup, fold, seed)
    python generate_all_splits.py --setup B3A --fold 0 --seed 0
    python generate_all_splits.py --data-root /path/to/old_bundle/data
    python generate_all_splits.py --data-root /path/to/old_bundle/data \
        --fold-table ../data/external/jaaks2022/canonical_folds_extmap_plus_jaaks_full.csv
                                                        # EXT (needs fold = -2 rows)

🚨 TWO SILENT INDEXING TRAPS — neither raises, both change the folds
1. DRUG indices come from `graphs/dp_drugs/drug_id_map.csv`, NOT from factorising the fold table.
   B3 / B3A / B4 / LCO / LCOW fold over drug indices.
2. CELL indices are FIRST-APPEARANCE order over the normalised name (`dpsyn/data.py:305`), NOT
   alphabetical. B2 / B2P / B4 / LCOW fold over cell indices. Sorting them was measured to move
   B2 train 20,421 -> 22,710 and the B2 tie floor 0.6548 -> 0.6652 with nothing raised.
`load()` below does both correctly. Copy it rather than rolling your own.

⚠️ `wq_MRR` HAS NO FIXED ZERO. An uninformative ranker scores the "tie floor" column, which is
split-determined and moves 0.5963 -> 0.6920 across settings — wider than most effects in the paper.
Quote it beside any wq_MRR and normalise as (score - floor) / (1 - floor) across settings.
"""
from __future__ import annotations

import argparse, csv, importlib.util, sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


S, M = _mod("splits"), _mod("metrics")

# Fold-based settings, in ladder order. LCO is DEPRECATED (config.DEPRECATED_SETUPS) but runnable,
# and every LCO result already produced stays valid.
FOLDED = ["B1", "LCOW", "LCO", "B2", "B2P", "B3", "B3A", "B4"]
LOO = ["C2", "C3", "C4", "LOSO"]        # need --loo-index; enumerated as usable-index counts
EXTERNAL = ["EXT"]                       # needs a fold table carrying fold = -2 rows


def norm_cell(name):
    """`dpsyn/data.py:norm_cell` — upper-case, strip, drop literal backslashes.

    The fold tables contain names like `NCI\\/ADR-RES`, a backslash written in by an ingest step;
    stripping it recovers ~200 triplets that would otherwise fail every join.
    """
    return str(name).replace("\\", "").strip().upper()


def load(fold_table=None, data_root=None):
    """Triplet table + the pipeline's own drug and cell indexing. See the two traps above."""
    data_root = Path(data_root) if data_root else DATA
    fold_table = Path(fold_table) if fold_table else data_root / "canonical_folds_extmap.csv"
    with open(data_root / "graphs/dp_drugs/drug_id_map.csv", newline="") as fh:
        idx = {r["drugbank_id"].strip().upper(): int(r["drug_idx"]) for r in csv.DictReader(fh)}
    with open(fold_table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    miss = {r["drug1_dbid"].strip().upper() for r in rows} | {r["drug2_dbid"].strip().upper() for r in rows}
    miss -= set(idx)
    if miss:
        raise SystemExit(f"{len(miss)} drug ids in {fold_table.name} are absent from drug_id_map.csv "
                         f"(e.g. {sorted(miss)[:3]}). Wrong graph for this fold table?")
    d1 = np.array([idx[r["drug1_dbid"].strip().upper()] for r in rows], dtype=np.int64)
    d2 = np.array([idx[r["drug2_dbid"].strip().upper()] for r in rows], dtype=np.int64)
    cmap = {}
    cell = np.array([cmap.setdefault(norm_cell(r["cell_name"]), len(cmap)) for r in rows], dtype=np.int64)
    y = np.array([int(float(r["label"])) for r in rows], dtype=np.int64)
    fold_col = (np.array([int(float(r.get("fold", -1) or -1)) for r in rows], dtype=np.int64)
                if "fold" in rows[0] else None)
    return d1, d2, cell, y, fold_col, fold_table


def tie_floor(d1, d2, cell, y, te, min_candidates=3):
    """Analytic within-query MRR of an UNINFORMATIVE ranker: no model, split-determined.

    A query fixes (anchor drug, cell) and ranks candidate partners. All-tied scores give the exact
    expectation E[1/rank] under uniform tie-breaking, which `metrics.within_query` also uses.
    """
    out = []
    for rows in M.build_queries(d1, d2, cell, te, min_candidates):
        yy = y[rows]; t, p = len(rows), int(yy.sum())
        if p and p != t:
            out.append(sum(comb(t - k, p - 1) / comb(t, p) / k for k in range(1, t - p + 2)))
    return float(np.mean(out)) if out else float("nan")


def one(setup, d, num_folds, fold, seed):
    d1, d2, cell, y, fold_col, _ = d
    return S.make_split(setup, cid1=d1, cid2=d2, cell=cell, y=y, num_folds=num_folds,
                        fold_index=fold, seed=seed, fold_col=fold_col)


def describe_split(setup, d, tr, va, te):
    d1, d2, cell, y = d[0], d[1], d[2], d[3]
    lk = S.verify_no_leak(setup, tr, va, te, cid1=d1, cid2=d2, cell=cell)
    flags = [f"{k}={v}" for k, v in lk.items()
             if k.startswith("test_") and isinstance(v, int) and v
             and k != "test_triplets_seen_cell"]
    return dict(n_train=len(tr), n_val=len(va), n_test=len(te), train_frac=len(tr) / len(y),
                prev=float(y[te].mean()) if len(te) else float("nan"),
                tie=tie_floor(d1, d2, cell, y, te), flags=";".join(flags) or "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", help="data/ directory from the earlier bundle")
    ap.add_argument("--fold-table"); ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--folds", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--setup"); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--loo-index", type=int)
    ap.add_argument("--out", help="directory: write one .npz per (setup, fold, seed)")
    a = ap.parse_args()

    if a.folds > a.num_folds:
        ap.error("--folds cannot exceed --num-folds")
    d = load(a.fold_table, a.data_root)
    d1, d2, cell, y, fold_col, table = d
    print(f"{table.name}: {len(y):,} triplets · {len(set(d1.tolist()) | set(d2.tolist())):,} drugs "
          f"· {len(set(cell.tolist()))} cells · prevalence {y.mean():.4f}")
    has_ext = fold_col is not None and (fold_col == -2).any()
    print(f"external rows (fold = -2): {int((fold_col == -2).sum()) if fold_col is not None else 0:,}"
          f"{'' if has_ext else '  → EXT unavailable from this table'}\n")

    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    if a.setup:
        tr, va, te = (S.make_split(a.setup, cid1=d1, cid2=d2, cell=cell, y=y,
                                   num_folds=a.num_folds, fold_index=a.fold, seed=a.seed,
                                   loo_index=a.loo_index, fold_col=fold_col))
        print(S.describe(a.setup))
        r = describe_split(a.setup, d, tr, va, te)
        print(f"  train {r['n_train']:,}  val {r['n_val']:,}  test {r['n_test']:,}  "
              f"frac {r['train_frac']:.3f}  prev {r['prev']:.3f}  tie floor {r['tie']:.4f}  "
              f"leak {r['flags']}")
        if out:
            p = out / f"{a.setup}_f{a.fold}_s{a.seed}.npz"
            np.savez(p, train=tr, val=va, test=te); print(f"  wrote {p}")
        return 0

    hdr = (f"{'setup':6s} {'train':>9s} {'val':>9s} {'test':>9s} {'frac':>6s} {'prev':>6s} "
           f"{'tie floor':>10s}  leak")
    # A fold=-2 table is specifically an EXT table. Running ordinary K-fold settings over the
    # source union would mix the external rows into training and would not reproduce an experiment
    # in this project. An explicit --setup still permits targeted diagnostics.
    todo = EXTERNAL if has_ext else FOLDED
    print(hdr); print("-" * len(hdr))
    for st in todo:
        rs, n = [], 0
        for f in range(a.folds):
            for s in range(a.seeds):
                tr, va, te = one(st, d, a.num_folds, f, s)
                rs.append(describe_split(st, d, tr, va, te)); n += 1
                if out:
                    np.savez(out / f"{st}_f{f}_s{s}.npz", train=tr, val=va, test=te)
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("n_train", "n_val", "n_test",
                                                             "train_frac", "prev", "tie")}
        fl = sorted({r["flags"] for r in rs} - {"-"})
        tag = "  ← DEPRECATED" if st == "LCO" else ""
        print(f"{st:6s} {m['n_train']:9,.0f} {m['n_val']:9,.0f} {m['n_test']:9,.0f} "
              f"{m['train_frac']:6.3f} {m['prev']:6.3f} {m['tie']:10.4f}  "
              f"{(fl[0] if fl else '-'):s}{tag}")
    print(f"\nMeans over folds 0-{a.folds-1} x seeds 0-{a.seeds-1} at --num-folds {a.num_folds}. "
          f"Every (fold, seed) is a DIFFERENT split, so\nthe spread across them is setting "
          f"difficulty, not measurement noise.")

    print("\nLeave-one-out settings need --loo-index; usable indices (two-class test set):")
    for st in LOO:
        if st == "LOSO":
            print(f"  {st:5s} needs `study` (from triplet_targets_v15.csv); 8 usable studies of 11")
            continue
        n_units = (len(set(cell.tolist())) if st == "C2"
                   else len(set(d1.tolist()) | set(d2.tolist())) if st == "C3" else len(y))
        print(f"  {st:5s} {n_units:,} candidate indices — "
              f"{'C4 test sets are never two-class; sweep refuses it' if st == 'C4' else 'run with --setup ' + st + ' --loo-index N'}")
    if not has_ext:
        print("\nEXT (independent Jaaks validation) needs a fold table with rows marked fold = -2:")
        print("  --fold-table data/external/jaaks2022/canonical_folds_extmap_plus_jaaks_full.csv")
    if out:
        print(f"\nwrote {len(todo) * a.folds * a.seeds} .npz files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
