#!/usr/bin/env python3
"""Build the CURRENT drug-synergy graph, fold by fold, with the leak controls made explicit.

    python build_rebuilt_graph.py --data-root /path/to/canonical_data --setup B1 --fold 0

WHY THIS SCRIPT REPLACES build_drug_protein_multigraph_pyg.py
-------------------------------------------------------------
The March bundle built ONE graph for the whole dataset and shipped drug-drug synergy edges
inside it. Those edges are derived from labels, so a graph built once over all rows contains
edges that encode test-set answers. Any model that reads them is scoring its own training
signal. The rebuilt pipeline (2026-08) therefore treats the graph as TWO things:

    STATIC, label-independent, built once:
        drug nodes, protein nodes
        protein -> drug   target edges          (from DrugBank)
        protein -- protein PPI edges            (from STRING)
        drug -- drug      similarity edges      (AtomPair-Tanimoto top-30; from CHEMISTRY only)

    LABEL-DERIVED, rebuilt INSIDE EVERY FOLD from TRAINING ROWS ONLY:
        drug -- drug      synergy edges

If you take one thing from this script, take that split. The static half can be built once and
cached. The synergy half must be rebuilt for every (setup, fold, seed) or the evaluation is
meaningless -- and this is not hypothetical: a symmetry bug that left the reverse direction of
held-out edges in the propagation graph leaked for two months in the old pipeline.

OTHER THINGS THAT CHANGED SINCE THE MARCH BUNDLE
------------------------------------------------
  * Two node types, not five. Cell lines and tissues are NOT nodes. Cell context enters at the
    decoder (and optionally as an edge ATTRIBUTE, see below). Making the cell a node was tried
    and did not pay for itself.
  * Labels are BINARY two-band, not ternary: mean ZIP > +10 positive, < -10 negative, and the
    |ZIP| <= 10 middle band is EXCLUDED from the dataset rather than labelled negative.
  * Drug features are AtomPair-1024 fingerprints, not PubMedBERT text embeddings. Text was
    measured as worth about -0.003 AUROC versus the fingerprint, i.e. nothing.
  * Protein features are ESM-2 (320-d, 99.4% coverage), not text.
  * The default triplet table is `extmap` (42,874 triplets, 1,108 drugs, 106 cells), not the
    older canonical 40,831.

THE EDGE KEY / EDGE ATTRIBUTE DISTINCTION -- the subtle part
------------------------------------------------------------
A synergy edge has a KEY (what makes two edges distinct) and an ATTRIBUTE (what it carries).
The LABEL key is (drug_i, drug_j, cell). So:

    key = (i, j, cell)  ->  the edge key EQUALS the label key, so an edge exists for a test row
                            only if that exact triplet was in training, which the split forbids.
                            Test coverage is 0.0% BY CONSTRUCTION.
    key = (i, j)        ->  17.9% of test rows have a usable edge.

**The edge key must be strictly COARSER than the label key.** To use the cell line, put it in the
ATTRIBUTE on a coarse key, not in the key. This script prints test coverage for every keying so a
collapse can never be mistaken for a modelling result.

Edge multiplicity: one edge per distinct (key, attribute value). A pair measured in six cell lines
becomes six edges, each carrying its own cell vector. Collapsing to one edge and keeping "the"
cell of an arbitrary row would hand the model a meaningless code.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not train anything. It builds and inspects the graph so you can see what the model sees.
For training use the real pipeline: `sbatch slurm_dpsyn.sh run --setup B3 --channels ego dp ppd`.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── labels ───────────────────────────────────────────────────────────────────────────────────
TAU = 10.0


def two_band_label(mean_zip: float, tau: float = TAU):
    """Positive / negative / EXCLUDED. Returns None for the neutral band.

    The neutral band is dropped, not labelled 0. Labelling it negative is the single most common
    misreading of this dataset and it changes prevalence from 0.318 to something else entirely.
    """
    if mean_zip > tau:
        return 1
    if mean_zip < -tau:
        return 0
    return None


# ── static, label-independent graph ──────────────────────────────────────────────────────────
def tanimoto_topk_edges(fp: np.ndarray, k: int = 30, block: int = 512):
    """Chemical-similarity kNN over binary fingerprints. Uses NO labels, so it is leak-free.

    Blocked so a 5,519 x 5,519 similarity matrix never has to exist at once.
    Returns a symmetric (2, E) edge_index.
    """
    B = (fp > 0).astype(np.float32)
    cnt = B.sum(1)
    src, dst = [], []
    for s in range(0, len(B), block):
        e = min(s + block, len(B))
        inter = B[s:e] @ B.T
        union = cnt[s:e, None] + cnt[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        for r in range(e - s):
            T[r, s + r] = -1.0                       # never a self-loop
            nb = np.argpartition(-T[r], k)[:k]
            for j in nb:
                src.append(s + r); dst.append(int(j))
    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.cat([ei, ei.flip(0)], dim=1)        # symmetrise


# ── label-derived synergy edges: THE PART THAT MUST BE REBUILT PER FOLD ──────────────────────
def build_synergy_edges(d1, d2, cell, y, train_idx, test_idx, *,
                        key="pair", attr="none", sign="pos", tissue_of_cell=None):
    """Synergy edges from TRAINING ROWS ONLY, symmetrised, with test coverage reported.

    `train_idx` is the ONLY row set this function is ever given. That structural fact -- not the
    assertions below -- is the real safety property.
    """
    if key == "none":
        return torch.zeros(2, 0, dtype=torch.long), {}, {"coverage": 0.0, "n_edges": 0}

    src_rows = [k for k in train_idx if (sign == "signed" or y[k] == 1)]

    def _key(k):
        i, j = int(d1[k]), int(d2[k])
        a, b = (i, j) if i < j else (j, i)
        if key == "pair":
            return (a, b)
        if key == "pair_cell":
            return (a, b, int(cell[k]))
        return (a, b, int(tissue_of_cell[int(cell[k])]))

    def _attr(k):
        if attr == "cell":
            return int(cell[k])
        if attr == "tissue":
            return int(tissue_of_cell[int(cell[k])])
        return -1

    # ONE EDGE PER DISTINCT (key, attribute value) -- see the module docstring
    emit = defaultdict(list)
    for k in src_rows:
        emit[(_key(k), _attr(k))].append(k)

    s, t, ec, es = [], [], [], []
    for (kk, _av), rws in emit.items():
        a, b = kk[0], kk[1]
        labs = np.array([y[r] for r in rws])
        sgn = 1.0 if labs.mean() > 0.5 else -1.0
        c = int(cell[rws[0]])                        # well-defined: rows here share the attribute
        for u, v in ((a, b), (b, a)):                # ALWAYS symmetric
            s.append(u); t.append(v); ec.append(c); es.append(sgn)

    ei = torch.tensor([s, t], dtype=torch.long) if s else torch.zeros(2, 0, dtype=torch.long)

    # ---- leak checks -------------------------------------------------------------------------
    train_keys = {(min(int(d1[k]), int(d2[k])), max(int(d1[k]), int(d2[k])), int(cell[k]))
                  for k in train_idx}
    held = set(range(len(y))) - set(int(k) for k in train_idx)
    held_keys = {(min(int(d1[k]), int(d2[k])), max(int(d1[k]), int(d2[k])), int(cell[k]))
                 for k in held}
    edge_keys = {(min(int(d1[k]), int(d2[k])), max(int(d1[k]), int(d2[k])), int(cell[k]))
                 for rws in emit.values() for k in rws}
    assert edge_keys <= train_keys, "a synergy edge came from a non-training row"
    overlap = edge_keys & held_keys
    assert not overlap, f"{len(overlap)} synergy edges coincide with held-out triplets"

    # ---- test coverage: the number that exposes a key collapse --------------------------------
    present = {kk for (kk, _av) in emit}
    seen = sum(1 for k in test_idx if _key(k) in present)
    diag = {"n_edges": int(ei.size(1)), "n_keys": len(present),
            "coverage": seen / max(len(test_idx), 1)}
    return ei, emit, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="directory holding the fold CSV, id maps and feature npz files")
    ap.add_argument("--fold-csv", default="canonical_folds_with_zip.csv")
    ap.add_argument("--setup", default="B1", choices=["B1", "B2", "B3"],
                    help="B1 = ID (fold over rows), B2 = cell-OOD, B3 = drug-OOD (OR rule)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    # ── 1. triplets ─────────────────────────────────────────────────────────────────────────
    rows = list(csv.DictReader(open(a.data_root / a.fold_csv)))
    drugs, cells = {}, {}
    d1, d2, cell, y = [], [], [], []
    n_neutral = 0
    for r in rows:
        z = r.get("mean_zip")
        if z in (None, ""):
            lab = int(r["label"])                     # already-binarised table
        else:
            lab = two_band_label(float(z))
            if lab is None:
                n_neutral += 1
                continue
        for d in (r["drug1_dbid"], r["drug2_dbid"]):
            drugs.setdefault(d.strip().upper(), len(drugs))
        # the column is `cell_name` in the extmap/canonical tables and `cell_line` in some
        # older exports; accept either rather than failing on a rename
        raw_cell = r.get("cell_name", r.get("cell_line", ""))
        cn = raw_cell.strip().upper().replace("\\", "")   # ingest artifact: NCI\\/ADR-RES
        cells.setdefault(cn, len(cells))
        d1.append(drugs[r["drug1_dbid"].strip().upper()])
        d2.append(drugs[r["drug2_dbid"].strip().upper()])
        cell.append(cells[cn]); y.append(lab)
    d1 = np.array(d1); d2 = np.array(d2); cell = np.array(cell); y = np.array(y)
    print(f"triplets {len(y):,}   drugs {len(drugs):,}   cells {len(cells)}   "
          f"prevalence {y.mean():.3f}   neutral-band excluded {n_neutral:,}")

    # ── 2. split ────────────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(a.seed)
    if a.setup == "B1":
        perm = rng.permutation(len(y)); f = np.array_split(perm, a.num_folds)
        te, va = f[a.fold], f[(a.fold + 1) % a.num_folds]
    elif a.setup == "B2":
        cs = rng.permutation(len(cells)); f = np.array_split(cs, a.num_folds)
        te = np.flatnonzero(np.isin(cell, f[a.fold]))
        va = np.flatnonzero(np.isin(cell, f[(a.fold + 1) % a.num_folds]))
    else:                                             # B3, OR rule
        dsx = rng.permutation(len(drugs)); f = np.array_split(dsx, a.num_folds)
        hot, hov = set(f[a.fold].tolist()), set(f[(a.fold + 1) % a.num_folds].tolist())
        isin = lambda arr, s: np.array([x in s for x in arr])
        te = np.flatnonzero(isin(d1, hot) | isin(d2, hot))
        va = np.flatnonzero((isin(d1, hov) | isin(d2, hov)) & ~(isin(d1, hot) | isin(d2, hot)))
    tr = np.setdiff1d(np.arange(len(y)), np.union1d(te, va))
    print(f"{a.setup} fold {a.fold}: train {len(tr):,} ({len(tr)/len(y):.1%})  "
          f"val {len(va):,}  test {len(te):,}   test prevalence {y[te].mean():.3f}")
    if len(tr) / len(y) < 0.40:
        print("  LOW_TRAIN_FRAC: entity folds compound. A B3 number is a SMALLER-TRAIN model "
              "than a B1 number -- never compare the two naively.")

    # ── 3. static graph ─────────────────────────────────────────────────────────────────────
    fp_path = a.data_root / "drug_features" / "drug_atompair.npz"
    if fp_path.exists():
        z = np.load(fp_path, allow_pickle=True)
        ids = [str(x).strip().upper() for x in z["ids"]]
        emb = z["embeddings"]
        fp = np.zeros((len(drugs), emb.shape[1]), np.float32)
        hit = 0
        for i, d in enumerate(ids):
            if d in drugs:
                fp[drugs[d]] = emb[i]; hit += 1
        print(f"AtomPair fingerprints: {hit:,}/{len(drugs):,} drugs matched")
        e_sim = tanimoto_topk_edges(fp, k=a.topk)
        print(f"similarity edges (top-{a.topk}, symmetric): {e_sim.size(1):,}  -- LABEL-FREE")
    else:
        print(f"[skip] no {fp_path}; similarity edges not built")
        e_sim = torch.zeros(2, 0, dtype=torch.long)

    # ── 4. synergy edges, per fold, train-only ──────────────────────────────────────────────
    print("\nsynergy edges rebuilt from TRAINING ROWS ONLY:")
    print(f"  {'key':12s} {'attr':7s} {'sign':7s} {'edges':>8s} {'keys':>7s} {'test coverage':>14s}")
    for key in ("pair", "pair_tissue", "pair_cell"):
        if key == "pair_tissue":
            continue                                  # needs a tissue map; see dpsyn/graph.py
        for sign in ("pos", "signed"):
            ei, _, diag = build_synergy_edges(d1, d2, cell, y, tr, te, key=key,
                                              attr="cell", sign=sign)
            flag = "  <-- KEY COLLAPSE" if diag["coverage"] == 0.0 and key == "pair_cell" else ""
            print(f"  {key:12s} {'cell':7s} {sign:7s} {diag['n_edges']:8,d} "
                  f"{diag['n_keys']:7,d} {diag['coverage']:13.1%}{flag}")
    print("\n  The label key is (drug_i, drug_j, cell). An edge key EQUAL to it can never be")
    print("  observed for a test row. Use a coarse key and carry the cell as an ATTRIBUTE.")
    if a.setup == "B3":
        print("  At drug-OOD every keying gives 0.0%: a test row contains a cold drug, so its")
        print("  pair cannot have been in training under ANY key. The synergy channel is a")
        print("  transductive-only device.")

    if a.out:
        ei, _, diag = build_synergy_edges(d1, d2, cell, y, tr, te, key="pair", attr="cell")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"e_sim": e_sim, "e_dd": ei, "d1": d1, "d2": d2, "cell": cell, "y": y,
                    "train_idx": tr, "val_idx": va, "test_idx": te,
                    "meta": {"setup": a.setup, "fold": a.fold, "seed": a.seed, **diag}}, a.out)
        print(f"\nwrote {a.out}")
        print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
