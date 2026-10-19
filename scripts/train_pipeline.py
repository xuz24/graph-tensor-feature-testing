#!/usr/bin/env python3
"""Training pipeline: the legacy 10-arm suite (unchanged) + the TensoGraph five-component ablation.

    # everything (legacy arms, then tg_ arms)
    python train_pipeline.py --data-root data --splits-dir /tmp/splits --setup B1 --fold 0 \
        --seed 0 --gpu 0 --smiles-file data/molecules.csv --out-csv results.csv
    # only the TensoGraph ablation / only the legacy suite / a single arm
    python train_pipeline.py ... --suite tensograph
    python train_pipeline.py ... --suite legacy
    python train_pipeline.py ... --arm tg_full

LEGACY SUITE  (identity, fp_only, fp_protein, tucker_only, tucker_protein, sage, sage_protein,
sage_tucker, sage_tucker_protein, sage_protein_tensor, sage_coupled_tucker)
    Same models, features, optimiser and selection rule as drug_feature_testing_10arms.py
    (SynergyModel with the symmetric Decoder, lr --lr, batch 2048).

TENSOGRAPH ABLATION  (arms prefixed tg_, nothing from the legacy arms is mixed in)
    1  GTN on the drug-drug graph + node features  -> 32-d per drug
    2  GCN on the molecular graph                  -> 156-d per drug
    3  MLP on cell-line gene expression            -> 768-d per cell line
    4  one x -> 2x -> x MLP per stream (GTN, GCN, cell, and the existing Tucker embedding)
    5  fusion MLP on [drug_i streams | drug_j streams | cell] -> one logit

    | arm               | what is removed from tg_full                                  |
    |-------------------|---------------------------------------------------------------|
    | tg_full           | nothing                                                       |
    | tg_no_gtn         | component 1                                                   |
    | tg_no_gcn         | component 2                                                   |
    | tg_no_expr        | component 3 (cell-ID embedding instead; --cell-fallback none  |
    |                   | omits the cell input altogether)                              |
    | tg_no_stream_mlps | component 4 (raw streams go straight to the head)             |
    | tg_no_fusion_mlp  | component 5 (a single linear readout replaces the MLP)        |
    | tg_no_tucker      | the Tucker stream (extra: not one of the five)                |
    | tg_gtn_only / tg_gcn_only / tg_tucker_only | all other drug streams            |

    Choices to know about:
    * The Tucker stream is the existing synergy Tucker embedding (build_tucker_embeddings,
      mode="synergy"), L2-normalised exactly like the legacy tuck_norm.
    * GTN edge types are [synergy_pos, synergy_neg, similarity, identity]: the same three channels
      the Tucker tensor uses, built from training rows only (no label leakage), plus identity.
    * Infomax features are not in the data bundle, so the GTN node features are AtomPair-1024.
    * GTN and GCN weights are trained end to end with the predictor. Their full-drug embeddings
      are computed once per epoch and shared across that epoch's minibatches.
    * GTN edge-type weights start uniform (softmax of a constant), so early on it cannot tell a
      synergistic edge from an antagonistic one; read tg_no_gtn with that in mind.
    * The fusion input is order-dependent (as in the original): drugs are swapped at random in
      training and the two orders' logits are averaged at eval.
    * Expression is z-scored per gene on training cells only; cell lines without expression get a
      zero row. The 19k-wide input goes through a 512-wide hidden layer (--expr-hidden) rather
      than the original C -> C layer (~370M parameters).
    * tg_ arms train with --tg-lr / --tg-batch (original TensoGraph values: 1e-4 / 256); the
      legacy arms keep --lr / batch 2048. Only compare tg_ arms with each other.

Single seed per invocation -- run several --seed values before trusting small deltas.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loaded by path (see drug_feature_testing_10arms.py): generate_all_splits.py and metrics.py sit
# next to this file and are not importable as a package.
GAS = _load_module("generate_all_splits", "generate_all_splits.py")
M = _load_module("metrics", "metrics.py")

from src.models import (SynergyModel, TensoGraphAblation, TG_ARMS,       # noqa: E402
                        TrainableGCNEncoder, TrainableGTNEncoder)
from src.utils import (build_coupled_tucker_embeddings, build_drug_protein_dict,  # noqa: E402,F401
                   build_gcn_embeddings, build_gtn_embeddings, build_mol_batch,
                   build_synergy_channels, build_tucker_embeddings, infer_cell_names, l2norm,
                   load_cell_expression, load_drug_id_map, load_fingerprints, load_smiles,
                   tanimoto_topk_edges)

LEGACY_ARMS = ["identity", "fp_only", "fp_protein", "tucker_only", "tucker_protein",
               "sage", "sage_protein", "sage_tucker", "sage_tucker_protein",
               "sage_protein_tensor", "sage_coupled_tucker"]
TG_ARM_NAMES = list(TG_ARMS)

CSV_COLS = ["auroc", "ap", "antagonism_auroc", "antagonism_ap", "antagonism_precision",
            "antagonism_recall", "antagonism_f1", "antagonism_threshold", "mrr", "hits10",
            "wq_auroc", "wq_mrr", "wq_hits1", "n_queries"]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 1. Diagnostics: how many candidates does each (anchor drug, cell) query actually have?
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
# 2. Training loop (model-agnostic) -- scores full-length arrays and hands them to metrics.py's
#    evaluate() rather than computing auroc/mrr locally.
# ═══════════════════════════════════════════════════════════════════════════════════════════
def train_model(name, model, d1, d2, cell, y, tr, va, te, epochs, lr, device, *,
                batch=2048, min_candidates=3):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [{name}] trainable params {n_params:,}  batch {batch}  lr {lr:g}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state = -float("inf"), None
    eval_batch = 2048

    def score_idx(idx):
        """Full-length score array (NaN outside `idx`) -- what metrics.evaluate()/within_query
        expect, since within_query's queries reference row indices into the full arrays."""
        model.eval()
        if hasattr(model, "refresh_drug_streams"):
            model.refresh_drug_streams()
        out = np.full(len(y), np.nan, dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(idx), eval_batch):
                b = idx[s:s + eval_batch]
                i = torch.from_numpy(d1[b]).long().to(device)
                j = torch.from_numpy(d2[b]).long().to(device)
                c = torch.from_numpy(cell[b]).long().to(device)
                out[b] = torch.sigmoid(model(i, j, c)).cpu().numpy()
        return out

    # queries only depend on the split, so build them once rather than per epoch
    val_queries = M.build_queries(d1, d2, cell, va, min_candidates)

    for epoch in range(1, epochs + 1):
        model.train()
        end_to_end = (getattr(model, "gtn_encoder", None) is not None or
                      getattr(model, "gcn_encoder", None) is not None)
        if end_to_end:
            model.refresh_drug_streams()
            opt.zero_grad()
        perm = np.random.permutation(tr)
        epoch_loss, seen = 0.0, 0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            if len(b) < 2:          # BatchNorm1d cannot train on a single row
                continue
            i = torch.from_numpy(d1[b]).long().to(device)
            j = torch.from_numpy(d2[b]).long().to(device)
            c = torch.from_numpy(cell[b]).long().to(device)
            yt = torch.from_numpy(y[b]).float().to(device)
            logit = model(i, j, c)
            loss = loss_fn(logit, yt)
            if end_to_end:
                (loss * (len(b) / len(perm))).backward(retain_graph=(s + batch < len(perm)))
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()
            epoch_loss += loss.item() * len(b)
            seen += len(b)

        if end_to_end:
            opt.step()

        val_metrics = M.evaluate(y, score_idx(va), test_idx=va, cell=cell, d1=d1, d2=d2,
                                 queries=val_queries, min_candidates=min_candidates)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"    [{name}] epoch {epoch:3d}  train_loss {epoch_loss/max(seen,1):.4f}  "
                  f"val_auroc {val_metrics['auroc']:.3f}  val_wq_mrr {val_metrics['wq_mrr']:.3f}")
        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if k not in {"stream_gtn", "stream_gcn"}}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # Antagonism decision threshold is selected on validation predictions, per
    # metrics.select_antagonism_threshold's docstring -- never on test.
    val_scores = score_idx(va)
    threshold = M.select_antagonism_threshold(y[va], val_scores[va])

    test_queries = M.build_queries(d1, d2, cell, te, min_candidates)
    return M.evaluate(y, score_idx(te), test_idx=te, cell=cell, d1=d1, d2=d2,
                      queries=test_queries, min_candidates=min_candidates,
                      decision_threshold=threshold)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 3. Helpers
# ═══════════════════════════════════════════════════════════════════════════════════════════
def legacy_uses_tucker(arm):
    """Legacy arms whose node_feat is FP || synergy-Tucker (same rule as the legacy main)."""
    return "tucker" in arm and "protein_tensor" not in arm and "coupled" not in arm


def legacy_node_feat(arm, fp_norm, tuck_norm, protein_tensor_norm, coupled_norm):
    if arm == "identity":
        return fp_norm            # identity doesn't use it, but pass something reasonable
    if arm == "sage_protein_tensor":
        return torch.cat([fp_norm, protein_tensor_norm], dim=1)
    if arm == "sage_coupled_tucker":
        return torch.cat([fp_norm, coupled_norm], dim=1)
    if legacy_uses_tucker(arm):
        return torch.cat([fp_norm, tuck_norm], dim=1)
    return fp_norm


def append_results_csv(path, setup, fold, seed, results, tie):
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["setup", "fold", "seed", "arm"] + CSV_COLS + ["wq_mrr_norm", "tie_floor"])
        for arm, m in results.items():
            norm = (m["wq_mrr"] - tie) / (1 - tie) if tie < 1 else float("nan")
            w.writerow([setup, fold, seed, arm] + [m.get(k, float("nan")) for k in CSV_COLS]
                       + [norm, tie])


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 4. Main
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=None)

    g = ap.add_argument_group("arm selection")
    g.add_argument("--suite", default="all", choices=["legacy", "tensograph", "all"],
                   help="which arms to run when --arm is not given")
    g.add_argument("--arm", default=None, choices=LEGACY_ARMS + TG_ARM_NAMES,
                   help="run a single arm instead of a suite")
    g.add_argument("--compare", action="store_true",
                   help="kept for compatibility with the old script; running without --arm "
                        "already runs the whole selected suite")
    g.add_argument("--out-csv", type=Path, default=None,
                   help="append one row per arm (setup, fold, seed, metrics) to this CSV")

    g = ap.add_argument_group("legacy arms")
    g.add_argument("--embed-dim", type=int, default=64)
    g.add_argument("--lr", type=float, default=1e-3)

    g = ap.add_argument_group("tg_ (TensoGraph ablation) arms")
    g.add_argument("--tg-lr", type=float, default=1e-4,
                   help="original TensoGraph value (args.lr[1])")
    g.add_argument("--tg-batch", type=int, default=256, help="original TensoGraph value")
    g.add_argument("--gtn-dim", type=int, default=32)
    g.add_argument("--gtn-channels", type=int, default=2)
    g.add_argument("--gtn-layers", type=int, default=2)
    g.add_argument("--cell-dim", type=int, default=768, help="expression-MLP output width")
    g.add_argument("--expr-hidden", type=int, default=512,
                   help="hidden width of the expression MLP (original: = #genes, ~370M params)")
    g.add_argument("--fusion-hidden", type=int, default=8192,
                   help="first hidden width of the fusion MLP (original: args.hidden[2])")
    g.add_argument("--cell-fallback", default="embedding", choices=["embedding", "none"],
                   help="cell representation when component 3 is removed (tg_no_expr)")
    g.add_argument("--smiles-file", type=Path, default=None,
                   help="molecules.csv (molecule_id, canonical_isomeric_smiles); "
                        "default <data-root>/molecules.csv")
    g.add_argument("--expr-file", type=Path, default=None,
                   help="default <data-root>/cell_features/cell_line_expression.npz")
    g.add_argument("--ccle-map", type=Path, default=None,
                   help="default <data-root>/cell_line_ccle_map.csv (or cell_features/)")
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

    # ── which arms, and therefore which features, are needed ─────────────────────────────────
    if a.arm:
        arms = [a.arm]
    elif a.suite == "legacy":
        arms = list(LEGACY_ARMS)
    elif a.suite == "tensograph":
        arms = list(TG_ARM_NAMES)
    else:
        arms = LEGACY_ARMS + TG_ARM_NAMES
    run_legacy = [x for x in arms if x in LEGACY_ARMS]
    run_tg = [x for x in arms if x in TG_ARMS]
    print(f"arms: {', '.join(arms)}")

    def tg_needs(key):
        return any(TG_ARMS[x][key] for x in run_tg)

    # ── shared features ──────────────────────────────────────────────────────────────────────
    fp = load_fingerprints(a.data_root, idx, N)
    e_sim = tanimoto_topk_edges(fp, k=a.topk)
    print(f"similarity edges: {e_sim.size(1):,} (label-free)")
    fp_norm = l2norm(torch.from_numpy(fp).float())

    # synergy Tucker embedding: the existing one, used by legacy tucker arms and the tg_ Tucker stream
    tuck_norm = None
    if any(legacy_uses_tucker(x) for x in run_legacy) or tg_needs("tucker"):
        tucker_emb = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank,
                                             mode="synergy")
        tuck_norm = l2norm(torch.from_numpy(tucker_emb).float())

    # ── legacy-only features (protein targets, protein / coupled tensors) ────────────────────
    drug_protein_dict = n_proteins = protein_tensor_norm = coupled_norm = None
    if any("protein" in x or "coupled" in x for x in run_legacy):
        drug_protein_dict, n_proteins = build_drug_protein_dict(a.data_root, fold_table)
    if "sage_protein_tensor" in run_legacy:
        pt = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank,
                                     drug_protein_dict=drug_protein_dict, mode="protein_tensor")
        protein_tensor_norm = l2norm(torch.from_numpy(pt).float())
    if "sage_coupled_tucker" in run_legacy:
        cp = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank,
                                     drug_protein_dict=drug_protein_dict, mode="coupled")
        coupled_norm = l2norm(torch.from_numpy(cp).float())

    # ── tg_ features: components 1, 2, 3 (+ the Tucker stream above) ─────────────────────────
    streams, expr = {}, None
    gtn_graph = gcn_graph = None
    if run_tg:
        if tg_needs("tucker"):
            streams["tucker"] = tuck_norm
        if tg_needs("gtn"):
            print(f"preparing trainable GTN on {device} (N={N:,}; dense graph needs several GB)")
            pos, neg, sim = build_synergy_channels(N, d1, d2, cell, y, tr, te, e_sim)
            adjacency = torch.from_numpy(np.stack([pos, neg, sim], axis=-1))
            adjacency = torch.cat([adjacency, torch.eye(N).unsqueeze(-1)], dim=-1)
            gtn_graph = (adjacency, fp_norm)
            del pos, neg, sim
            streams["gtn"] = torch.zeros(N, a.gtn_dim)
        if tg_needs("gcn"):
            smiles_path = a.smiles_file or (a.data_root / "molecules.csv")
            if not smiles_path.exists():
                raise SystemExit(f"SMILES file not found: {smiles_path} (use --smiles-file)")
            smiles = load_smiles(smiles_path, idx)
            batch_mol, rows = build_mol_batch(smiles)
            labelled = set(np.union1d(d1, d2).tolist())
            covered = len(labelled & set(rows.tolist()))
            print(f"  [gcn] SMILES for {len(smiles):,}/{N:,} graph drugs, parsed {len(rows):,}; "
                  f"labelled drugs covered {covered}/{len(labelled)} "
                  f"({covered/len(labelled):.1%}); the rest get a zero vector")
            gcn_graph = (batch_mol, rows)
            streams["gcn"] = torch.zeros(N, 156)
        if tg_needs("expr"):
            expr_path = a.expr_file or (a.data_root / "cell_features" / "cell_line_expression.npz")
            ccle_map = a.ccle_map
            if ccle_map is None:
                for cand in (a.data_root / "cell_line_ccle_map.csv",
                             a.data_root / "cell_features" / "cell_line_ccle_map.csv"):
                    if cand.exists():
                        ccle_map = cand
                        break
            cell_names = infer_cell_names(fold_table, cell, y, num_cells)
            expr, _have = load_cell_expression(expr_path, ccle_map, cell_names,
                                               np.unique(cell[tr]))

    # ── run the arms ─────────────────────────────────────────────────────────────────────────
    results = {}
    for arm in arms:
        print(f"\n== arm: {arm} ==")
        # re-seed per arm so a result does not depend on which other arms ran before it
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        if arm in TG_ARMS:
            cfg = TG_ARMS[arm]
            gtn_encoder = (TrainableGTNEncoder(
                *gtn_graph, node_dim=a.gtn_dim, num_channels=a.gtn_channels,
                num_layers=a.gtn_layers, seed=a.seed) if cfg["gtn"] else None)
            gcn_encoder = (TrainableGCNEncoder(*gcn_graph, N, seed=a.seed)
                           if cfg["gcn"] else None)
            model = TensoGraphAblation.from_arm(
                arm, streams, expr, num_cells, cell_dim=a.cell_dim, expr_hidden=a.expr_hidden,
                fusion_hidden=a.fusion_hidden,
                id_cell_dim=32 if a.cell_fallback == "embedding" else 0,
                gtn_encoder=gtn_encoder, gcn_encoder=gcn_encoder)
            lr, batch = a.tg_lr, a.tg_batch
        else:
            node_feat = legacy_node_feat(arm, fp_norm, tuck_norm, protein_tensor_norm,
                                         coupled_norm)
            model = SynergyModel(arm, N, num_cells, node_feat, e_sim, n_proteins=n_proteins,
                                 drug_protein_dict=drug_protein_dict, out_dim=a.embed_dim)
            lr, batch = a.lr, 2048
        results[arm] = train_model(arm, model, d1, d2, cell, y, tr, va, te, a.epochs, lr, device,
                                   batch=batch, min_candidates=a.min_query_size)

    # ── report ───────────────────────────────────────────────────────────────────────────────
    print(f"\n{'arm':24s} {'auroc':>7s} {'ap':>7s} {'wq_auroc':>9s} {'wq_mrr':>8s} "
          f"{'wq_mrr_norm':>12s} {'mrr':>7s} {'n_queries':>10s}")
    for arm in LEGACY_ARMS + TG_ARM_NAMES:
        if arm not in results:
            continue
        m = results[arm]
        wq_norm = (m["wq_mrr"] - tie) / (1 - tie) if tie < 1 else float("nan")
        mrr = m.get("mrr", float("nan"))
        print(f"{arm:24s} {m['auroc']:7.3f} {m['ap']:7.3f} {m['wq_auroc']:9.3f} "
              f"{m['wq_mrr']:8.3f} {wq_norm:12.3f} {mrr:7.3f} {m['n_queries']:10,d}")
    print(f"\nwq_mrr_norm = (wq_mrr - tie_floor) / (1 - tie_floor), tie_floor = {tie:.4f}")

    if a.out_csv:
        append_results_csv(a.out_csv, a.setup, a.fold, a.seed, results, tie)
        print(f"appended {len(results)} rows to {a.out_csv}")

    if run_legacy:
        print("\nKey ablations to interpret (legacy suite):")
        print("\n  PROTEIN TARGET INFORMATION:")
        print("    fp_protein - fp_only:                does protein data help without other structure?")
        print("    sage_protein - sage:                 does protein help alongside graph smoothing?")
        print("    sage_tucker_protein - sage_tucker:   does protein help when synergy is present?")
        print("\n  SYNERGY LABELS:")
        print("    tucker_only - fp_only:               does synergy help without graph?")
        print("    sage_tucker - sage:                  does synergy help alongside graph?")
        print("\n  TENSOR FORMULATIONS (new variants):")
        print("    sage_protein_tensor - sage:          does co-targeting pattern (drug x drug x protein) help?")
        print("    sage_coupled_tucker - sage_tucker:   does coupling synergy+protein improve over synergy alone?")
        print("    sage_protein_tensor - sage_tucker:   is protein co-targeting better than synergy labels?")
        print("    sage_coupled_tucker - sage_protein:  is coupled factorization better than protein features?")
        print("\n  GRAPH STRUCTURE:")
        print("    sage - fp_only:                      does graph smoothing help (raw signal from similarity graph)?")

    if run_tg:
        print("\nKey ablations to interpret (TensoGraph, each arm minus tg_full; negative = the")
        print("removed component was helping):")
        print("    tg_no_gtn - tg_full:          component 1, GTN on the drug-drug graph")
        print("    tg_no_gcn - tg_full:          component 2, molecular GCN")
        print("    tg_no_expr - tg_full:         component 3, cell-line expression MLP")
        print("    tg_no_stream_mlps - tg_full:  component 4, per-stream MLPs")
        print("    tg_no_fusion_mlp - tg_full:   component 5, fusion MLP (linear readout instead)")
        print("    tg_no_tucker - tg_full:       the Tucker stream (extra)")
        print("    tg_*_only - tg_full:          what a single drug stream recovers on its own")

    print("\nInterpretation notes:")
    print("  - All deltas <0.003 AUROC are within noise floor; aggregate across seeds before concluding.")
    print("  - At B1 (in-dist): expect small effects; graph+features mostly sufficient.")
    print("  - At B3 (drug-OOD): expect larger effects; structured signals matter more.")
    if run_tg:
        print("  - tg_ arms use the same cell-ID embedding fallback in tg_no_expr; at cell-OOD setups")
        print("    (B2, B2P, B4) unseen-cell rows of that embedding are untrained, so tg_no_expr is")
        print("    expected to collapse there.")


if __name__ == "__main__":
    main()
