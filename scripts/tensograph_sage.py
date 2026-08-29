#!/usr/bin/env python3
"""TensoGraph-style Tucker drug embeddings + a basic GraphSAGE, evaluated on the rebuilt graph.

    python tensograph_sage.py --data-root data --fold-csv canonical_folds_extmap.csv \
        --setup B1 --fold 0 --compare

WHAT THIS DOES
--------------
1. Reuses build_rebuilt_graph.py's leak-safe machinery: labels, split, label-free similarity
   edges, and label-derived synergy edges rebuilt from TRAINING ROWS ONLY (functions below are
   copied from that script so this file runs standalone -- keep them in sync if that script
   changes).
2. Stacks TRAIN-ONLY synergy edges (positive / negative) and the label-free similarity edges into
   an (N_drug, N_drug, 3) tensor and runs Tucker decomposition (tensorly) to get ONE global drug
   embedding table -- not one per cell line. See the note on why "one per cell" is the wrong port
   of TensoGraph's design for this graph.
3. Trains a small GraphSAGE (torch_geometric) over the label-free similarity graph, with node
   features = fingerprint, optionally concatenated with the Tucker embedding.
4. Trains a symmetric MLP decoder over (drug_i, drug_j, cell) and reports AUROC/AUPRC plus an
   approximate within-query MRR (fix anchor drug + cell, rank candidate partners), the same
   headline metric the real pipeline reports.
5. With --compare, runs four arms back to back so they're directly comparable:
     identity      -- learnable per-drug embedding, no features, no graph (the floor to beat)
     fp_only       -- raw fingerprint, no graph, no Tucker
     sage          -- GraphSAGE over similarity edges, fingerprint features, no Tucker
     sage_tucker   -- GraphSAGE over similarity edges, fingerprint + Tucker embedding
   Comparing sage vs sage_tucker isolates whether the Tucker/global-embedding step is doing
   anything at all, which is the actual question the task is asking.

WHY THE TUCKER TENSOR IS BUILT ONCE, NOT PER CELL LINE (departure from TensoGraph)
-----------------------------------------------------------------------------------
TensoGraph builds a SEPARATE (drug, drug, relation) tensor per cell line, using only that cell's
own training edges. On this dataset that is the `pair_cell` keying the graph-builder script flags
as a collapse: ~400 rows spread over 106 cells leaves each per-cell tensor almost empty, so a
held-out cell (B2) or a held-out drug (B3) gets an all-zero, uninformative factor row. Building
ONE tensor across all cells (the `pair` key) keeps the density the diagnostic printout showed you
(21.3% / 38.9% coverage) and matches the "coarser than the label key, cell as an attribute"
discipline the rest of the pipeline already enforces. Cell identity re-enters at the decoder, not
in the tensor -- exactly where the project README says cell context belongs now.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
* No protein / PPI / target-edge graph. This is scoped to the drug-drug question the task asked
  about; wiring in the heterogeneous protein graph is a natural next step (see docstring bottom).
* Only B1 / B2 / B3, matching build_rebuilt_graph.py. B4 (joint-OOD) and LCO (pair-OOD) need the
  same split logic that script would need extending with.
* The within-query MRR here is a reasonable reconstruction (reciprocal rank of the first positive
  candidate per (anchor drug, cell) query) but is NOT guaranteed to match dpsyn's exact wq_mrr
  implementation bit-for-bit. Treat absolute numbers as approximate; treat the relative ordering
  between arms (identity / fp_only / sage / sage_tucker) as the meaningful result, and cross-check
  against `sbatch slurm_dpsyn.sh run` before reporting a delta as real.
* Single seed, no early stopping tuned per setting. This is a diagnostic run, not a leaderboard
  number -- rerun with a few seeds before trusting anything smaller than the ~0.003 noise floor.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import tensorly as tl
from tensorly.decomposition import tucker

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise SystemExit(
        "torch_geometric is required (pip install torch_geometric). "
        f"Original error: {e}"
    )

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 1. Copied from build_rebuilt_graph.py so this file runs standalone. Keep in sync.
# ═══════════════════════════════════════════════════════════════════════════════════════════
TAU = 10.0


def two_band_label(mean_zip: float, tau: float = TAU):
    if mean_zip > tau:
        return 1
    if mean_zip < -tau:
        return 0
    return None


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
    """Returns edge_index, emit dict, and diagnostics. See build_rebuilt_graph.py for the full
    leak-check commentary -- the asserts below are load-bearing, not decorative."""
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


def load_triplets(data_root: Path, fold_csv: str):
    rows = list(csv.DictReader(open(data_root / fold_csv)))
    drugs, cells = {}, {}
    d1, d2, cell, y = [], [], [], []
    for r in rows:
        z = r.get("mean_zip")
        if z in (None, ""):
            lab = int(r["label"])
        else:
            lab = two_band_label(float(z))
            if lab is None:
                continue
        for d in (r["drug1_dbid"], r["drug2_dbid"]):
            drugs.setdefault(d.strip().upper(), len(drugs))
        raw_cell = r.get("cell_name", r.get("cell_line", ""))
        cn = raw_cell.strip().upper().replace("\\", "")
        cells.setdefault(cn, len(cells))
        d1.append(drugs[r["drug1_dbid"].strip().upper()])
        d2.append(drugs[r["drug2_dbid"].strip().upper()])
        cell.append(cells[cn]); y.append(lab)
    return drugs, cells, np.array(d1), np.array(d2), np.array(cell), np.array(y)


def make_split(setup, fold, num_folds, seed, drugs, cells, d1, d2, cell, y):
    rng = np.random.default_rng(seed)
    if setup == "B1":
        perm = rng.permutation(len(y)); f = np.array_split(perm, num_folds)
        te, va = f[fold], f[(fold + 1) % num_folds]
    elif setup == "B2":
        cs = rng.permutation(len(cells)); f = np.array_split(cs, num_folds)
        te = np.flatnonzero(np.isin(cell, f[fold]))
        va = np.flatnonzero(np.isin(cell, f[(fold + 1) % num_folds]))
    else:  # B3, OR rule
        dsx = rng.permutation(len(drugs)); f = np.array_split(dsx, num_folds)
        hot, hov = set(f[fold].tolist()), set(f[(fold + 1) % num_folds].tolist())
        isin = lambda arr, s: np.array([x in s for x in arr])
        te = np.flatnonzero(isin(d1, hot) | isin(d2, hot))
        va = np.flatnonzero((isin(d1, hov) | isin(d2, hov)) & ~(isin(d1, hot) | isin(d2, hot)))
    tr = np.setdiff1d(np.arange(len(y)), np.union1d(te, va))
    return tr, va, te


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
# 4. Metrics: AUROC/AUPRC + approximate within-query MRR
# ═══════════════════════════════════════════════════════════════════════════════════════════
def within_query_mrr(anchors, cells_, labels, scores):
    """Fix (anchor drug, cell), rank candidates by score, take reciprocal rank of the first
    positive. Each undirected test row contributes two queries (drug1 as anchor, drug2 as anchor)
    since "fix the anchor, rank partners" is direction-specific."""
    groups = defaultdict(list)
    for a, c, y, s in zip(anchors, cells_, labels, scores):
        groups[(a, c)].append((s, y))
    rr = []
    for (_a, _c), items in groups.items():
        if not any(y == 1 for _, y in items):
            continue
        items.sort(key=lambda t: -t[0])
        for rank, (_s, y) in enumerate(items, start=1):
            if y == 1:
                rr.append(1.0 / rank)
                break
    return float(np.mean(rr)) if rr else float("nan"), len(rr)


def evaluate(model, idx, d1, d2, cell, y, device, batch=4096):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            i = torch.from_numpy(d1[b]).long().to(device)
            j = torch.from_numpy(d2[b]).long().to(device)
            c = torch.from_numpy(cell[b]).long().to(device)
            logit = model(i, j, c)
            all_scores.append(torch.sigmoid(logit).cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = y[idx]
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    auprc = average_precision_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    # symmetrize the query set: each row queried from both drugs' perspective
    anchors = np.concatenate([d1[idx], d2[idx]])
    cells_ = np.concatenate([cell[idx], cell[idx]])
    labs = np.concatenate([labels, labels])
    scs = np.concatenate([scores, scores])
    mrr, n_q = within_query_mrr(anchors, cells_, labs, scs)
    return {"auroc": auroc, "auprc": auprc, "wq_mrr": mrr, "n_queries": n_q}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 5. Training loop
# ═══════════════════════════════════════════════════════════════════════════════════════════
def train_arm(arm, N, num_cells, node_feat, edge_index, d1, d2, cell, y, tr, va, te,
             epochs, lr, device, out_dim=64):
    model = SynergyModel(arm, N, num_cells, node_feat, edge_index, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state = float("inf"), None
    batch = 2048
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
        val_metrics = evaluate(model, va, d1, d2, cell, y, device)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"    [{arm}] epoch {epoch:3d}  train_loss {epoch_loss/len(tr):.4f}  "
                  f"val_auroc {val_metrics['auroc']:.3f}  val_wq_mrr {val_metrics['wq_mrr']:.3f}")
        if -val_metrics["auroc"] < best_val:  # maximize auroc == minimize -auroc
            best_val = -val_metrics["auroc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, te, d1, d2, cell, y, device)
    return test_metrics


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--fold-csv", default="canonical_folds_extmap.csv")
    ap.add_argument("--setup", default="B1", choices=["B1", "B2", "B3"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--tucker-rank", type=int, nargs=3, default=[64, 64, 3])
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

    drugs, cells, d1, d2, cell, y = load_triplets(a.data_root, a.fold_csv)
    N, num_cells = len(drugs), len(cells)
    print(f"triplets {len(y):,}  drugs {N:,}  cells {num_cells}  prevalence {y.mean():.3f}")

    tr, va, te = make_split(a.setup, a.fold, a.num_folds, a.seed, drugs, cells, d1, d2, cell, y)
    print(f"{a.setup} fold {a.fold}: train {len(tr):,} ({len(tr)/len(y):.1%})  "
          f"val {len(va):,}  test {len(te):,}")

    fp_path = a.data_root / "drug_features" / "drug_atompair.npz"
    z = np.load(fp_path, allow_pickle=True)
    ids = [str(x).strip().upper() for x in z["ids"]]
    emb = z["embeddings"]
    fp = np.zeros((N, emb.shape[1]), np.float32)
    for i, dname in enumerate(ids):
        if dname in drugs:
            fp[drugs[dname]] = emb[i]
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
                                 tr, va, te, a.epochs, a.lr, device, out_dim=a.embed_dim)

    print(f"\n{'arm':14s} {'auroc':>7s} {'auprc':>7s} {'wq_mrr':>8s} {'n_queries':>10s}")
    for arm, m in results.items():
        print(f"{arm:14s} {m['auroc']:7.3f} {m['auprc']:7.3f} {m['wq_mrr']:8.3f} {m['n_queries']:10,d}")
    print("\nCompare wq_mrr above against the identity-floor and random-ranking rows in the "
          "project README's table for this setup before drawing any conclusion. If sage_tucker "
          "doesn't clear sage by more than ~0.003, the Tucker step isn't earning its complexity "
          "here -- that's a legitimate, reportable finding, not a bug to chase.")


if __name__ == "__main__":
    main()
