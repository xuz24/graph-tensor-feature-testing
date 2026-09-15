#!/usr/bin/env python3
"""TensoGraph-style Tucker drug embeddings + a basic GraphSAGE, evaluated on the rebuilt graph.

    python drug_feature_testing.py --data-root data --splits-dir /tmp/splits \
        --setup B3 --fold 0 --seed 0 --compare

WHAT CHANGED FROM tensograph_sage.py
-------------------------------------
This is tensograph_sage.py's model + Tucker step, rewired onto the project's actual data/split/
metric machinery instead of the copies that script carried for standalone use:

  * Data and drug/cell indexing come from generate_all_splits.load(), not a hand-rolled
    load_triplets(). That means the two silent indexing traps documented at the top of
    generate_all_splits.py (drug indices from drug_id_map.csv, cell indices in first-appearance
    order) are handled the one correct way instead of a second, divergent way.
  * Splits are READ, not regenerated: this script takes --splits-dir and loads the pre-generated
    `{setup}_f{fold}_s{seed}.npz` files (train/val/test) written by
    `generate_all_splits.py --out <dir>`. It no longer contains a make_split() of its own, and it
    now runs every FOLDED setup (B1, LCOW, LCO, B2, B2P, B3, B3A, B4), not just B1/B2/B3.
  * Evaluation uses metrics.py's evaluate() (auroc/ap/antagonism metrics, prevalence-gated
    mrr/hits10, and the tie-corrected within-query auroc/mrr/hits1) instead of the ad hoc
    within_query_mrr()/expected_reciprocal_rank() this script used to carry -- those are exactly
    duplicated, more carefully, in metrics.within_query. The antagonism decision threshold is
    selected on validation via metrics.select_antagonism_threshold(), not fixed at 0.5.
  * wq_mrr is reported next to generate_all_splits.tie_floor() for the same test split, since
    wq_mrr's floor is split-determined (see that module's docstring) and an un-normalised number
    is not comparable across setups.

Everything about *why* the tensor is built once globally rather than per cell line, what node
features/graph go in, and the four comparison arms (identity / fp_only / sage / sage_tucker) is
unchanged from tensograph_sage.py; see that file's docstring for the full rationale.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
* No protein / PPI / target-edge graph -- scoped to the drug-drug question only.
* LOO settings (C2/C3/C4/LOSO) and EXT are out of scope here; those need --loo-index or an
  external fold table and don't fit the (setup, fold, seed) .npz layout this script reads.
* Single seed per invocation, no early-stopping schedule tuned per setting -- run --compare across
  a few --seed values before trusting a delta smaller than the ~0.003 noise floor.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
from tensorly.decomposition import tucker

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise SystemExit(
        "torch_geometric is required (pip install torch_geometric). "
        f"Original error: {e}"
    )

HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loaded by path, not by package import: the containing directory is `scripts/`, has no
# __init__.py, and isn't on sys.path as a package named `script` (the previous `from
# script.metrics import *` would raise ModuleNotFoundError). generate_all_splits.py locates
# splits.py/metrics.py the same way, relative to its own __file__, so this works no matter what
# the containing folder is named or how this file is invoked.
GAS = _load_module("generate_all_splits", "generate_all_splits.py")
M = _load_module("metrics", "metrics.py")


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 1. Label-free similarity graph + train-fold-only synergy edges. No metrics.py equivalent --
#    these build the GRAPH the metrics are computed over, they don't compute a metric themselves.
#    Copied from tensograph_sage.py; keep in sync if that script's edge-building logic changes.
# ═══════════════════════════════════════════════════════════════════════════════════════════
def tanimoto_topk_edges(fp: np.ndarray, k: int = 30, block: int = 512):
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
            T[r, s + r] = -1.0
            nb = np.argpartition(-T[r], k)[:k]
            for j in nb:
                src.append(s + r); dst.append(int(j))
    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.cat([ei, ei.flip(0)], dim=1)


def build_synergy_edges(d1, d2, cell, y, train_idx, test_idx, *,
                        key="pair", attr="cell", sign="signed"):
    """Returns edge_index, emit dict, and diagnostics. The asserts below are load-bearing: they
    are what makes this a leak-safe rebuild, not decorative."""
    src_rows = [k for k in train_idx if (sign == "signed" or y[k] == 1)]

    def _key(k):
        i, j = int(d1[k]), int(d2[k])
        a, b = (i, j) if i < j else (j, i)
        return (a, b)

    emit = defaultdict(list)
    for k in src_rows:
        emit[_key(k)].append(k)

    s, t, sign_of_edge = [], [], {}
    for (a, b), rws in emit.items():
        labs = np.array([y[r] for r in rws])
        sgn = 1.0 if labs.mean() > 0.5 else -1.0
        sign_of_edge[(a, b)] = sgn
        for u, v in ((a, b), (b, a)):
            s.append(u); t.append(v)
    ei = torch.tensor([s, t], dtype=torch.long) if s else torch.zeros(2, 0, dtype=torch.long)

    train_keys = {(min(int(d1[k]), int(d2[k])), max(int(d1[k]), int(d2[k])))
                  for k in train_idx}
    edge_keys = set(emit.keys())
    assert edge_keys <= train_keys, "a synergy edge came from a non-training row"

    present = set(emit.keys())
    seen = sum(1 for k in test_idx if _key(k) in present)
    diag = {"n_edges": int(ei.size(1)), "n_keys": len(present),
            "coverage": seen / max(len(test_idx), 1)}
    return ei, sign_of_edge, diag


def load_drug_id_map(data_root: Path) -> dict:
    """drugbank_id -> drug_idx, straight from the pipeline's own graph indexing.

    This is trap #1 from generate_all_splits.py: drug indices come from drug_id_map.csv, not from
    factorising the fold table. GAS.load() already applies this correctly for d1/d2, but doesn't
    hand back the mapping itself, and we need it to align the fingerprint file below.
    """
    with open(data_root / "graphs/dp_drugs/drug_id_map.csv", newline="") as fh:
        return {r["drugbank_id"].strip().upper(): int(r["drug_idx"]) for r in csv.DictReader(fh)}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 2. TensoGraph-style Tucker global embedding, adapted: one tensor, not one per cell line.
# ═══════════════════════════════════════════════════════════════════════════════════════════
def build_tucker_embeddings(N, d1, d2, cell, y, train_idx, test_idx, e_sim, rank):
    """(N, N, 3) tensor: [synergy_pos, synergy_neg, chem_similarity] -> Tucker -> factors[0].

    Cold drugs (no train-fold synergy edges, e.g. held out at B3) get an all-zero row in the
    first two channels; their Tucker factor is carried entirely by the similarity channel. We
    additionally compute a neighbor-averaged fallback for rows that end up ~zero everywhere, since
    ALS on an (almost) all-zero row is noise, not signal.
    """
    _, sign_of_edge, diag = build_synergy_edges(d1, d2, cell, y, train_idx, test_idx,
                                                key="pair", attr="cell", sign="signed")
    print(f"  [tucker] synergy(pair,signed): {diag['n_edges']} edges, "
          f"{diag['n_keys']} keys, test coverage {diag['coverage']:.1%}")

    pos = np.zeros((N, N), np.float32)
    neg = np.zeros((N, N), np.float32)
    for (a, b), sgn in sign_of_edge.items():
        (pos if sgn > 0 else neg)[a, b] = (pos if sgn > 0 else neg)[b, a] = 1.0

    sim = np.zeros((N, N), np.float32)
    ei = e_sim.numpy()
    sim[ei[0], ei[1]] = 1.0

    tensor = tl.tensor(np.stack([pos, neg, sim], axis=-1))
    r0 = min(rank[0], N)
    core, factors = tucker(tensor, rank=[r0, r0, min(rank[2], 3)])
    drug_global = factors[0]  # (N, r0)

    # cold-row fallback: neighbor-average over the (label-free, always-available) sim graph
    has_signal = (pos.sum(1) + neg.sum(1)) > 0
    cold = ~has_signal
    if cold.any():
        deg = sim.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            neighbor_avg = np.divide(sim @ drug_global, deg, out=np.zeros_like(drug_global),
                                     where=deg > 0)
        drug_global[cold] = neighbor_avg[cold]
        print(f"  [tucker] {cold.sum()}/{N} drugs had no train-fold synergy edges; "
              f"fell back to similarity-neighbor averaging for their embedding")
    return drug_global.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 3. Model: GraphSAGE encoder (or identity table) + symmetric MLP decoder
# ═══════════════════════════════════════════════════════════════════════════════════════════
class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class Decoder(nn.Module):
    """Symmetric in (drug_i, drug_j): built from elementwise product/sum/|diff| so swapping the
    two drugs never changes the score, matching how synergy is actually reported."""
    def __init__(self, drug_dim, cell_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(drug_dim * 3 + cell_dim, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, ei_emb, ej_emb, cell_emb):
        feat = torch.cat([ei_emb * ej_emb, ei_emb + ej_emb, (ei_emb - ej_emb).abs(), cell_emb], 1)
        return self.net(feat).squeeze(-1)


class SynergyModel(nn.Module):
    def __init__(self, arm, num_drugs, num_cells, node_feat, edge_index, cell_dim=32, out_dim=64):
        super().__init__()
        self.arm = arm
        self.edge_index = edge_index
        self.cell_emb = nn.Embedding(num_cells, cell_dim)
        if arm == "identity":
            self.drug_emb = nn.Embedding(num_drugs, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        elif arm == "fp_only":
            self.register_buffer("node_feat", node_feat)
            self.decoder = Decoder(node_feat.shape[1], cell_dim)
        else:  # sage, sage_tucker
            self.register_buffer("node_feat", node_feat)
            self.encoder = SAGEEncoder(node_feat.shape[1], 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)

    def drug_embeddings(self, device):
        if self.arm == "identity":
            return self.drug_emb.weight
        if self.arm == "fp_only":
            return self.node_feat
        return self.encoder(self.node_feat, self.edge_index.to(device))

    def forward(self, i_idx, j_idx, cell_idx):
        E = self.drug_embeddings(i_idx.device)
        return self.decoder(E[i_idx], E[j_idx], self.cell_emb(cell_idx))


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 4. Diagnostics: how many candidates does each (anchor drug, cell) query actually have?
# ═══════════════════════════════════════════════════════════════════════════════════════════
def query_group_sizes(d1, d2, cell, y, idx, label, min_candidates):
    """Smoke test: if the median candidate count is tiny, within-query wq_mrr is close to a coin
    flip regardless of model quality. Run once per (setup, fold, seed), not per epoch/arm -- it
    only depends on the split, not the model.
    """
    qmap = defaultdict(list)
    for k in idx:
        qmap[(int(d1[k]), int(cell[k]))].append(int(y[k]))
        qmap[(int(d2[k]), int(cell[k]))].append(int(y[k]))
    sizes = np.array([len(v) for v in qmap.values() if sum(v) > 0])
    if sizes.size == 0:
        print(f"  [diag:{label}] no qualifying (>=1 positive) query groups found")
        return
    print(f"  [diag:{label}] qualifying query groups (>=1 positive): {sizes.size:,}  "
          f"candidates/query -- min {sizes.min()}  p25 {np.percentile(sizes,25):.0f}  "
          f"median {np.median(sizes):.0f}  mean {sizes.mean():.1f}  max {sizes.max()}")
    kept = int((sizes >= min_candidates).sum())
    print(f"  [diag:{label}] at min_candidates={min_candidates}: {kept:,}/{sizes.size:,} "
          f"queries survive ({kept/sizes.size:.1%})")
    if np.median(sizes) <= min_candidates:
        print(f"  [diag:{label}] WARNING: median group size <= min_candidates -- wq_mrr here is "
              f"close to the tie floor regardless of model skill")


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 5. Training loop -- scores full-length arrays and hands them to metrics.py's evaluate()
#    rather than computing auroc/mrr locally.
# ═══════════════════════════════════════════════════════════════════════════════════════════
def train_arm(arm, N, num_cells, node_feat, edge_index, d1, d2, cell, y, tr, va, te,
             epochs, lr, device, out_dim=64, min_candidates=3):
    model = SynergyModel(arm, N, num_cells, node_feat, edge_index, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state = -float("inf"), None
    batch = 2048

    def score_idx(idx):
        """Full-length score array (NaN outside `idx`) -- what metrics.evaluate()/within_query
        expect, since within_query's queries reference row indices into the full arrays."""
        model.eval()
        out = np.full(len(y), np.nan, dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                i = torch.from_numpy(d1[b]).long().to(device)
                j = torch.from_numpy(d2[b]).long().to(device)
                c = torch.from_numpy(cell[b]).long().to(device)
                out[b] = torch.sigmoid(model(i, j, c)).cpu().numpy()
        return out

    # queries only depend on the split, so build them once rather than per epoch
    val_queries = M.build_queries(d1, d2, cell, va, min_candidates)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(tr)
        epoch_loss = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            i = torch.from_numpy(d1[b]).long().to(device)
            j = torch.from_numpy(d2[b]).long().to(device)
            c = torch.from_numpy(cell[b]).long().to(device)
            yt = torch.from_numpy(y[b]).float().to(device)
            opt.zero_grad()
            logit = model(i, j, c)
            loss = loss_fn(logit, yt)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)

        val_metrics = M.evaluate(y, score_idx(va), test_idx=va, cell=cell, d1=d1, d2=d2,
                                 queries=val_queries, min_candidates=min_candidates)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"    [{arm}] epoch {epoch:3d}  train_loss {epoch_loss/len(tr):.4f}  "
                  f"val_auroc {val_metrics['auroc']:.3f}  val_wq_mrr {val_metrics['wq_mrr']:.3f}")
        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Antagonism decision threshold is selected on validation predictions, per
    # metrics.select_antagonism_threshold's docstring -- never on test.
    val_scores = score_idx(va)
    threshold = M.select_antagonism_threshold(y[va], val_scores[va])

    test_queries = M.build_queries(d1, d2, cell, te, min_candidates)
    test_metrics = M.evaluate(y, score_idx(te), test_idx=te, cell=cell, d1=d1, d2=d2,
                              queries=test_queries, min_candidates=min_candidates,
                              decision_threshold=threshold)
    return test_metrics


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="data/ directory (contains graphs/dp_drugs/drug_id_map.csv, "
                         "drug_features/, and the fold table)")
    ap.add_argument("--fold-table", default=None,
                    help="defaults to <data-root>/canonical_folds_extmap.csv")
    ap.add_argument("--splits-dir", type=Path, required=True,
                    help="directory of {setup}_f{fold}_s{seed}.npz files written by "
                         "`generate_all_splits.py --out <dir>`")
    ap.add_argument("--setup", default="B1",
                    choices=["B1", "LCOW", "LCO", "B2", "B2P", "B3", "B3A", "B4"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--tucker-rank", type=int, nargs=3, default=[64, 64, 3])
    ap.add_argument("--min-query-size", type=int, default=3,
                    help="drop within-query groups with fewer than this many candidates "
                         "(passed straight through to metrics.build_queries)")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arm", default="sage_tucker",
                    choices=["identity", "fp_only", "sage", "sage_tucker"])
    ap.add_argument("--compare", action="store_true",
                    help="run all four arms back to back and print a summary table")
    ap.add_argument("--gpu", type=int, default=None)
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if (a.gpu is not None and torch.cuda.is_available())
                          else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    fold_table = a.fold_table or (a.data_root / "canonical_folds_extmap.csv")
    idx = load_drug_id_map(a.data_root)
    N = max(idx.values()) + 1
    d1, d2, cell, y, fold_col, table = GAS.load(fold_table, a.data_root)
    num_cells = int(cell.max()) + 1
    print(f"{table.name}: {len(y):,} triplets  drugs {N:,}  cells {num_cells}  "
          f"prevalence {y.mean():.3f}")

    split_path = a.splits_dir / f"{a.setup}_f{a.fold}_s{a.seed}.npz"
    if not split_path.exists():
        raise SystemExit(f"missing {split_path} -- run `generate_all_splits.py --out "
                         f"{a.splits_dir}` first")
    sp = np.load(split_path)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"{a.setup} fold {a.fold} seed {a.seed}: train {len(tr):,} ({len(tr)/len(y):.1%})  "
          f"val {len(va):,}  test {len(te):,}")

    query_group_sizes(d1, d2, cell, y, va, "val", min_candidates=a.min_query_size)
    query_group_sizes(d1, d2, cell, y, te, "test", min_candidates=a.min_query_size)
    tie = GAS.tie_floor(d1, d2, cell, y, te, min_candidates=a.min_query_size)
    print(f"  [diag:test] wq_mrr tie floor for this split: {tie:.4f} "
          f"(uninformative-ranker expectation; wq_mrr has no fixed zero -- see "
          f"generate_all_splits.py)")

    fp_path = a.data_root / "drug_features" / "drug_atompair.npz"
    z = np.load(fp_path, allow_pickle=True)
    ids = [str(x).strip().upper() for x in z["ids"]]
    emb = z["embeddings"]
    fp = np.zeros((N, emb.shape[1]), np.float32)
    for i, dname in enumerate(ids):
        if dname in idx:
            fp[idx[dname]] = emb[i]
    e_sim = tanimoto_topk_edges(fp, k=a.topk)
    print(f"similarity edges: {e_sim.size(1):,} (label-free)")

    tucker_emb = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank)

    fp_t = torch.from_numpy(fp).float()
    fp_norm = fp_t / (fp_t.norm(dim=1, keepdim=True) + 1e-8)
    tuck_t = torch.from_numpy(tucker_emb).float()
    tuck_norm = tuck_t / (tuck_t.norm(dim=1, keepdim=True) + 1e-8)
    node_feat_plain = fp_norm
    node_feat_tucker = torch.cat([fp_norm, tuck_norm], dim=1)

    arms = ["identity", "fp_only", "sage", "sage_tucker"] if a.compare else [a.arm]
    results = {}
    for arm in arms:
        print(f"\n== arm: {arm} ==")
        node_feat = node_feat_tucker if arm == "sage_tucker" else node_feat_plain
        results[arm] = train_arm(arm, N, num_cells, node_feat, e_sim, d1, d2, cell, y,
                                 tr, va, te, a.epochs, a.lr, device, out_dim=a.embed_dim,
                                 min_candidates=a.min_query_size)

    print(f"\n{'arm':14s} {'auroc':>7s} {'ap':>7s} {'wq_auroc':>9s} {'wq_mrr':>8s} "
          f"{'wq_mrr_norm':>12s} {'mrr':>7s} {'n_queries':>10s}")
    for arm, m in results.items():
        wq_norm = (m["wq_mrr"] - tie) / (1 - tie) if tie < 1 else float("nan")
        mrr = m.get("mrr", float("nan"))
        print(f"{arm:14s} {m['auroc']:7.3f} {m['ap']:7.3f} {m['wq_auroc']:9.3f} "
              f"{m['wq_mrr']:8.3f} {wq_norm:12.3f} {mrr:7.3f} {m['n_queries']:10,d}")
    print(f"\nwq_mrr_norm = (wq_mrr - tie_floor) / (1 - tie_floor), tie_floor = {tie:.4f} for this "
          "exact (setup, fold, seed) test split -- compare that column across arms, not raw "
          "wq_mrr, since the floor itself moves across settings (generate_all_splits.py docstring). "
          "`mrr`/`hits10` print NaN above prevalence "
          f"{M.SATURATION_PREVALENCE} (see metrics.rank_metrics) -- expected for most FOLDED "
          "settings; that's not a bug. If sage_tucker doesn't clear sage by a meaningful margin on "
          "wq_mrr_norm, the Tucker step isn't earning its complexity here -- a legitimate finding, "
          "not one to chase away.")


if __name__ == "__main__":
    main()