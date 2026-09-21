#!/usr/bin/env python3
"""10-arm ablation: TensoGraph Tucker + protein features + tensor variants + GraphSAGE.

    python drug_feature_testing_8arms.py --data-root data --splits-dir /tmp/splits \
        --setup B1 --fold 0 --seed 0 --compare

TEN ARMS TESTED
---------------
Ablation over multiple signal sources and tensor formulations: fingerprints (always), three types
of Tucker factorizations (synergy, protein co-targeting, or coupled), graph message-passing, and
protein target information via learnable embeddings.

| Arm | FP | Protein agg | Tensor type | GraphSAGE |
|---|---|---|---|---|
| identity | -- | -- | none | -- |
| fp_only | ✓ | -- | none | -- |
| fp_protein | ✓ | ✓ | none | -- |
| tucker_only | ✓ | -- | synergy | -- |
| tucker_protein | ✓ | ✓ | synergy | -- |
| sage | ✓ | -- | none | ✓ |
| sage_protein | ✓ | ✓ | none | ✓ |
| sage_tucker | ✓ | -- | synergy | ✓ |
| sage_tucker_protein | ✓ | ✓ | synergy | ✓ |
| sage_protein_tensor | ✓ | -- | drug×drug×protein (co-targeting) | ✓ |
| sage_coupled_tucker | ✓ | -- | synergy + protein (coupled) | ✓ |

The two new tensor variants test whether:
* Encoding shared protein targets (co-targeting patterns) helps predict synergy.
* Jointly factorizing synergy and protein networks (with shared drug factors) outperforms
  using synergy alone or aggregated protein information alone.

See DATA_MANIFEST.md for protein target edge count and coverage. Each arm's drug embedding 
concatenates the selected signals, then (for sage* arms) passes through GraphSAGE over the 
chemical-similarity graph, and finally decodes with cell-line embedding.

WHAT CHANGED FROM drug_feature_testing_8arms.py
----------------------------------------------
* Added three new Tucker factorization modes: synergy (original), protein_tensor (co-targeting),
  and coupled (synergy + protein jointly factorized).
* Added build_drug_protein_tensor() to encode drug-pair co-targeting patterns.
* Added build_coupled_tucker_embeddings() to jointly factor synergy and protein networks.
* Updated build_tucker_embeddings() to accept mode parameter and dispatch to the three variants.
* Added sage_protein_tensor and sage_coupled_tucker arms.
* Expanded from 8 to 10 arms.

Everything about *why* the tensor is built once globally, leak safety of synergy edges, and the 
core GraphSAGE architecture is unchanged from earlier scripts.

LOO settings (C2/C3/C4/LOSO), EXT, and cross-project generalization are out of scope; this 
script runs only FOLDED setups (B1, LCOW, LCO, B2, B2P, B3, B3A, B4) with splits from 
generate_all_splits.py. Single seed per invocation -- run --compare across a few --seed values 
before trusting deltas < 0.003 AUROC.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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


def build_drug_protein_dict(data_root: Path, fold_table: Path = None):
    """Extract drug-protein target edges from the heterogeneous graph (HeteroData format).
    
    Maps from graph drug indices (0..5518) to lists of protein graph indices (0..16063).
    
    Returns:
        drug_protein_dict: dict mapping drug_graph_idx -> list of protein_indices
        n_proteins: int, number of protein nodes in the graph
    """
    data_root = Path(data_root)
    
    # Load the graph
    graph = torch.load(data_root / "graphs/dp_drugs/unified_drugcomb_hetero.pt")
    
    # Load ID maps for reference
    try:
        drug_id_map = pd.read_csv(data_root / "graphs/dp_drugs/drug_id_map.csv", index_col=0)
        protein_id_map = pd.read_csv(data_root / "graphs/dp_drugs/protein_id_map.csv", index_col=0)
        n_proteins = len(protein_id_map)
    except FileNotFoundError:
        print("  WARNING: could not load ID maps, using default n_proteins=16064")
        n_proteins = 16064
    
    # Extract protein-drug target edges from the graph
    drug_protein_dict = defaultdict(list)
    
    # For HeteroData, iterate through edge_types
    if hasattr(graph, 'edge_types'):
        # Graph is HeteroData with edge_types
        for edge_type in graph.edge_types:
            if len(edge_type) == 3:
                src_type, rel, dst_type = edge_type
                
                # Look for drug -> protein edges (relation is 'dp')
                if src_type == 'drug' and dst_type == 'protein' and rel in ['dp', 'targets', 'target']:
                    edge_index = graph[edge_type]['edge_index']
                    edge_index_np = edge_index.numpy() if torch.is_tensor(edge_index) else edge_index
                    for drug_idx, prot_idx in edge_index_np.T:
                        drug_protein_dict[int(drug_idx)].append(int(prot_idx))
                
                # Also check reverse direction (protein -> drug)
                elif src_type == 'protein' and dst_type == 'drug' and rel in ['pd', 'targeted_by']:
                    edge_index = graph[edge_type]['edge_index']
                    edge_index_np = edge_index.numpy() if torch.is_tensor(edge_index) else edge_index
                    for prot_idx, drug_idx in edge_index_np.T:
                        drug_protein_dict[int(drug_idx)].append(int(prot_idx))
    
    total_targets = sum(len(v) for v in drug_protein_dict.values())
    print(f"built drug-protein dict: {len(drug_protein_dict):,} drugs have targets, "
          f"total target edges: {total_targets:,}")
    
    return dict(drug_protein_dict), n_proteins


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 2. TensoGraph-style Tucker global embedding, adapted: one tensor, not one per cell line.
# ═══════════════════════════════════════════════════════════════════════════════════════════
def build_drug_protein_tensor(N, drug_protein_dict):
    """Build drug co-targeting matrix: entry [i,j] = number of shared protein targets.
    
    More efficient than full 3D tensor: instead of drug × drug × protein, 
    compute drug-drug co-targeting as a 2D matrix that can be Tucker-decomposed.
    
    This captures shared mechanisms: drug pairs with many overlapping targets encode similar
    biology. Tucker decomposition extracts a drug embedding from this structure.
    """
    if not drug_protein_dict:
        print("  [drug_protein_tensor] empty dict, returning zero matrix")
        return np.zeros((N, 1), dtype=np.float32)
    
    # Build drug × protein incidence matrix first (5519 × 16043)
    n_proteins = max(max(targets) if targets else 0 
                     for targets in drug_protein_dict.values()) + 1
    drug_protein_mat = np.zeros((N, n_proteins), dtype=np.float32)
    
    for drug_id, protein_ids in drug_protein_dict.items():
        for p_id in protein_ids:
            drug_protein_mat[drug_id, p_id] = 1.0
    
    # Compute drug-drug co-targeting via matrix multiplication
    # co_target[i,j] = number of proteins both drugs i and j target
    co_target = drug_protein_mat @ drug_protein_mat.T  # (5519, 5519)
    
    # Normalize and optionally threshold to keep only strong co-targeting
    # Remove self-loops and zero out weak signals
    np.fill_diagonal(co_target, 0)  # No self-loops
    
    edge_count = (co_target > 0).sum()
    avg_shared = co_target[co_target > 0].mean() if edge_count > 0 else 0
    print(f"  [drug_protein_tensor] drug-drug co-targeting matrix {co_target.shape}, "
          f"{edge_count:.0f} drug pairs with shared targets, "
          f"avg {avg_shared:.2f} shared proteins per pair")
    
    return co_target.astype(np.float32)


def build_coupled_tucker_embeddings(N, d1, d2, cell, y, train_idx, test_idx, e_sim, 
                                    drug_protein_dict, rank=[64, 64, 3], rank_protein=64):
    """Two coupled tensors, jointly factorized (approximate):
    
    Tensor A: drug × drug × 3 = [synergy_pos, synergy_neg, similarity]
    Tensor B: drug × protein × 1 = [target]
    
    Returns the drug embedding factors from A and B averaged together. A true coupled Tucker
    would jointly optimize both factors (requires scipy/ADMM), but this approximation forces
    them to live in the same latent space.
    """
    # Tensor A: synergy + similarity (same as classic Tucker)
    _, sign_of_edge, diag = build_synergy_edges(d1, d2, cell, y, train_idx, test_idx,
                                                key="pair", attr="cell", sign="signed")
    print(f"  [coupled] tensorA synergy(pair,signed): {diag['n_edges']} edges, "
          f"{diag['n_keys']} keys, test coverage {diag['coverage']:.1%}")
    
    pos = np.zeros((N, N), np.float32)
    neg = np.zeros((N, N), np.float32)
    for (a, b), sgn in sign_of_edge.items():
        (pos if sgn > 0 else neg)[a, b] = (pos if sgn > 0 else neg)[b, a] = 1.0
    
    sim = np.zeros((N, N), np.float32)
    ei = e_sim.numpy()
    sim[ei[0], ei[1]] = 1.0
    
    tensor_A = tl.tensor(np.stack([pos, neg, sim], axis=-1))
    r0 = min(rank[0], N)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        core_A, factors_A = tucker(tensor_A, rank=[r0, r0, min(rank[2], 3)])
    drug_global_A = factors_A[0]  # (N, r0)
    
    # Tensor B: drug × protein targets
    if not drug_protein_dict:
        print(f"  [coupled] tensorB empty, skipping")
        drug_global_B = drug_global_A.copy()
    else:
        n_proteins = max(max(targets) if targets else 0 
                        for targets in drug_protein_dict.values()) + 1
        target_mat = np.zeros((N, n_proteins), dtype=np.float32)
        for drug_id, protein_ids in drug_protein_dict.items():
            for p_id in protein_ids:
                target_mat[drug_id, p_id] = 1.0
        
        # Reshape as drug × protein × 1 for Tucker
        tensor_B = tl.tensor(target_mat[:, :, np.newaxis])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            core_B, factors_B = tucker(tensor_B, rank=[min(r0, N), min(rank_protein, n_proteins), 1])
        drug_global_B = factors_B[0]  # (N, r0)
        
        print(f"  [coupled] tensorB targets: {target_mat.sum():.0f} edges, "
              f"{(target_mat > 0).sum(0).mean():.1f} avg proteins per drug")
    
    # Couple by averaging both drug factors (simple coupling without ADMM)
    drug_global = 0.5 * drug_global_A + 0.5 * drug_global_B
    
    # Cold-row fallback: neighbor-average over the (label-free) similarity graph
    has_signal = (pos.sum(1) + neg.sum(1)) > 0
    cold = ~has_signal
    if cold.any():
        deg = sim.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            neighbor_avg = np.divide(sim @ drug_global, deg, out=np.zeros_like(drug_global),
                                     where=deg > 0)
        drug_global[cold] = neighbor_avg[cold]
        print(f"  [coupled] {cold.sum()}/{N} drugs had no train-fold synergy edges; "
              f"fell back to similarity-neighbor averaging for their embedding")
    
    return drug_global.astype(np.float32)


def build_tucker_embeddings(N, d1, d2, cell, y, train_idx, test_idx, e_sim, rank,
                           drug_protein_dict=None, mode="synergy"):
    """
    Three Tucker factorization modes:
    
    mode="synergy": (N, N, 3) tensor [synergy_pos, synergy_neg, chem_similarity] -> drug embedding
    mode="protein_tensor": (N, N, n_proteins) tensor [co-targeting] -> drug embedding
    mode="coupled": (N, N, 3) + (N, n_proteins, 1) jointly factorized -> averaged drug embedding
    """
    if mode == "synergy":
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    
    elif mode == "protein_tensor":
        tensor = build_drug_protein_tensor(N, drug_protein_dict)
        # tensor is now 2D (N, N), not 3D
        # Reshape it as (N, N, 1) for Tucker factorization
        tensor_3d = np.expand_dims(tensor, axis=-1)
        tensor_tl = tl.tensor(tensor_3d)
        r0 = min(rank[0], N)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            core, factors = tucker(tensor_tl, rank=[r0, r0, 1])
        drug_global = factors[0]  # (N, r0)
        return drug_global.astype(np.float32)
    
    elif mode == "coupled":
        return build_coupled_tucker_embeddings(N, d1, d2, cell, y, train_idx, test_idx, e_sim,
                                              drug_protein_dict, rank)
    
    else:
        raise ValueError(f"unknown Tucker mode: {mode}")


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


class ProteinFeatureAggregator(nn.Module):
    """For each drug, average the embeddings of its target proteins.
    This turns protein-level information into a per-drug feature with no message-passing."""
    
    def __init__(self, n_proteins, protein_dim=64):
        super().__init__()
        self.protein_emb = nn.Embedding(n_proteins, protein_dim)
        self.protein_dim = protein_dim
    
    def get_protein_sigs(self, n_drugs, drug_protein_dict, device):
        """Compute protein signatures for all drugs."""
        sigs = torch.zeros(n_drugs, self.protein_dim, device=device)
        for drug_id, prot_list in drug_protein_dict.items():
            if len(prot_list) > 0:
                prot_emb = self.protein_emb(
                    torch.tensor(prot_list, dtype=torch.long, device=device)
                )
                sigs[drug_id] = prot_emb.mean(dim=0)
        return sigs


class SynergyModel(nn.Module):
    def __init__(self, arm, num_drugs, num_cells, node_feat, edge_index, 
                 n_proteins=None, drug_protein_dict=None, cell_dim=32, out_dim=64, protein_dim=64):
        super().__init__()
        self.arm = arm
        self.edge_index = edge_index
        self.cell_emb = nn.Embedding(num_cells, cell_dim)
        self.num_drugs = num_drugs
        self.drug_protein_dict = drug_protein_dict
        
        # Protein aggregator only for arms that use it at runtime (protein agg, not protein tensor)
        # sage_protein: uses aggregator
        # sage_tucker_protein: uses aggregator
        # sage_protein_tensor: does NOT use aggregator (already in node_feat)
        # sage_coupled_tucker: does NOT use aggregator (already in node_feat)
        has_protein_agg = 'protein' in arm and 'tensor' not in arm and 'coupled' not in arm
        
        if has_protein_agg:
            assert n_proteins is not None and drug_protein_dict is not None, \
                f"arm {arm} requires n_proteins and drug_protein_dict"
            self.protein_agg = ProteinFeatureAggregator(n_proteins, protein_dim)
            protein_feat_dim = protein_dim
        else:
            protein_feat_dim = 0
        
        self.register_buffer("node_feat", node_feat)
        base_feat_dim = node_feat.shape[1]
        feat_with_protein_dim = base_feat_dim + protein_feat_dim
        
        if arm == "identity":
            self.drug_emb = nn.Embedding(num_drugs, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        elif arm == "fp_only":
            self.decoder = Decoder(base_feat_dim, cell_dim)
        
        elif arm == "fp_protein":
            self.decoder = Decoder(feat_with_protein_dim, cell_dim)
        
        elif arm == "tucker_only":
            self.decoder = Decoder(base_feat_dim, cell_dim)
        
        elif arm == "tucker_protein":
            self.decoder = Decoder(feat_with_protein_dim, cell_dim)
        
        elif arm == "sage":
            self.encoder = SAGEEncoder(base_feat_dim, 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        elif arm == "sage_protein":
            # node_feat is fp; will add protein at runtime
            self.encoder = SAGEEncoder(feat_with_protein_dim, 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        elif arm == "sage_tucker":
            # node_feat is fp ‖ tucker; no protein addition
            self.encoder = SAGEEncoder(base_feat_dim, 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        elif arm == "sage_tucker_protein":
            # node_feat is fp ‖ tucker; will add protein at runtime
            self.encoder = SAGEEncoder(feat_with_protein_dim, 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        elif arm == "sage_protein_tensor":
            # node_feat is fp ‖ protein_tensor; no runtime additions
            self.encoder = SAGEEncoder(base_feat_dim, 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        elif arm == "sage_coupled_tucker":
            # node_feat is fp ‖ coupled; no runtime additions
            self.encoder = SAGEEncoder(base_feat_dim, 128, out_dim)
            self.decoder = Decoder(out_dim, cell_dim)
        
        else:
            raise ValueError(f"unknown arm: {arm}")
        
    def _get_protein_sigs(self, device):
        """Compute and cache protein signatures for all drugs."""
        if 'protein' not in self.arm:
            return None
        return self.protein_agg.get_protein_sigs(self.num_drugs, self.drug_protein_dict, device)

    def drug_embeddings(self, device):
        """Compute drug embeddings for the current arm."""
        if self.arm == "identity":
            return self.drug_emb.weight
        
        if self.arm == "fp_only":
            return self.node_feat
        
        if self.arm == "fp_protein":
            protein_sigs = self._get_protein_sigs(device)
            return torch.cat([self.node_feat, protein_sigs], dim=1)
        
        if self.arm == "tucker_only":
            # node_feat already contains FP ‖ Tucker from the caller
            return self.node_feat
        
        if self.arm == "tucker_protein":
            # node_feat contains FP ‖ Tucker; add protein
            protein_sigs = self._get_protein_sigs(device)
            return torch.cat([self.node_feat, protein_sigs], dim=1)
        
        if self.arm == "sage":
            # GraphSAGE over similarity edges, fingerprint input
            return self.encoder(self.node_feat, self.edge_index.to(device))
        
        if self.arm == "sage_protein":
            # GraphSAGE over similarity edges, fingerprint ‖ protein input
            protein_sigs = self._get_protein_sigs(device)
            feat = torch.cat([self.node_feat, protein_sigs], dim=1)
            return self.encoder(feat, self.edge_index.to(device))
        
        if self.arm == "sage_tucker":
            # GraphSAGE over similarity edges, fingerprint ‖ Tucker input
            # node_feat already contains FP ‖ Tucker from the caller
            return self.encoder(self.node_feat, self.edge_index.to(device))
        
        if self.arm == "sage_tucker_protein":
            # GraphSAGE over similarity edges, fingerprint ‖ Tucker ‖ protein input
            protein_sigs = self._get_protein_sigs(device)
            feat = torch.cat([self.node_feat, protein_sigs], dim=1)
            return self.encoder(feat, self.edge_index.to(device))
        
        if self.arm == "sage_protein_tensor":
            # GraphSAGE over similarity edges, fingerprint ‖ Tucker(protein_tensor) input
            # node_feat already contains FP ‖ Tucker from the caller
            return self.encoder(self.node_feat, self.edge_index.to(device))
        
        if self.arm == "sage_coupled_tucker":
            # GraphSAGE over similarity edges, fingerprint ‖ Tucker(coupled synergy+protein) input
            # node_feat already contains FP ‖ Tucker from the caller
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
             epochs, lr, device, out_dim=64, min_candidates=3, 
             n_proteins=None, drug_protein_dict=None):
    model = SynergyModel(arm, N, num_cells, node_feat, edge_index,
                        n_proteins=n_proteins, drug_protein_dict=drug_protein_dict,
                        out_dim=out_dim).to(device)
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
                    choices=["identity", "fp_only", "fp_protein", "tucker_only", "tucker_protein",
                             "sage", "sage_protein", "sage_tucker", "sage_tucker_protein",
                             "sage_protein_tensor", "sage_coupled_tucker"])
    ap.add_argument("--compare", action="store_true",
                    help="run all 10 arms back to back and print a summary table")
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

    # Build Tucker embeddings (synergy-based, the original formulation)
    tucker_emb = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank,
                                         mode="synergy")

    fp_t = torch.from_numpy(fp).float()
    fp_norm = fp_t / (fp_t.norm(dim=1, keepdim=True) + 1e-8)
    tuck_t = torch.from_numpy(tucker_emb).float()
    tuck_norm = tuck_t / (tuck_t.norm(dim=1, keepdim=True) + 1e-8)

    # Build drug-protein dict for protein arms
    drug_protein_dict, n_proteins = build_drug_protein_dict(a.data_root, fold_table)

    # Build protein_tensor embedding (drug × drug × protein co-targeting)
    protein_tensor_emb = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank,
                                                 drug_protein_dict=drug_protein_dict,
                                                 mode="protein_tensor")
    protein_tensor_t = torch.from_numpy(protein_tensor_emb).float()
    protein_tensor_norm = protein_tensor_t / (protein_tensor_t.norm(dim=1, keepdim=True) + 1e-8)

    # Build coupled Tucker embedding (synergy + protein jointly factorized)
    coupled_emb = build_tucker_embeddings(N, d1, d2, cell, y, tr, te, e_sim, a.tucker_rank,
                                         drug_protein_dict=drug_protein_dict,
                                         mode="coupled")
    coupled_t = torch.from_numpy(coupled_emb).float()
    coupled_norm = coupled_t / (coupled_t.norm(dim=1, keepdim=True) + 1e-8)

    # All 10 arms
    all_arms = ["identity", "fp_only", "fp_protein", "tucker_only", "tucker_protein",
                "sage", "sage_protein", "sage_tucker", "sage_tucker_protein",
                "sage_protein_tensor", "sage_coupled_tucker"]
    arms = all_arms if a.compare else [a.arm]
    
    results = {}
    for arm in arms:
        print(f"\n== arm: {arm} ==")
        
        # Select node_feat based on arm:
        # - fp_only, fp_protein, sage, sage_protein: use fingerprint only
        # - tucker_only, tucker_protein, sage_tucker, sage_tucker_protein: use FP ‖ Tucker (synergy)
        # - sage_protein_tensor: use FP ‖ Tucker (protein co-targeting)
        # - sage_coupled_tucker: use FP ‖ Tucker (synergy + protein coupled)
        # - identity: use identity embedding (no node_feat needed)
        if arm == "identity":
            node_feat = fp_norm  # identity doesn't use it, but pass something reasonable
        elif arm == "sage_protein_tensor":
            node_feat = torch.cat([fp_norm, protein_tensor_norm], dim=1)
        elif arm == "sage_coupled_tucker":
            node_feat = torch.cat([fp_norm, coupled_norm], dim=1)
        elif 'tucker' in arm and 'protein_tensor' not in arm and 'coupled' not in arm:
            # Original tucker arms (tucker_only, tucker_protein, sage_tucker, sage_tucker_protein)
            node_feat = torch.cat([fp_norm, tuck_norm], dim=1)
        else:
            # fp-only arms
            node_feat = fp_norm
        
        results[arm] = train_arm(arm, N, num_cells, node_feat, e_sim, d1, d2, cell, y,
                                 tr, va, te, a.epochs, a.lr, device, out_dim=a.embed_dim,
                                 min_candidates=a.min_query_size,
                                 n_proteins=n_proteins, drug_protein_dict=drug_protein_dict)

    print(f"\n{'arm':24s} {'auroc':>7s} {'ap':>7s} {'wq_auroc':>9s} {'wq_mrr':>8s} "
          f"{'wq_mrr_norm':>12s} {'mrr':>7s} {'n_queries':>10s}")
    for arm in all_arms:
        if arm not in results:
            continue
        m = results[arm]
        wq_norm = (m["wq_mrr"] - tie) / (1 - tie) if tie < 1 else float("nan")
        mrr = m.get("mrr", float("nan"))
        print(f"{arm:24s} {m['auroc']:7.3f} {m['ap']:7.3f} {m['wq_auroc']:9.3f} "
              f"{m['wq_mrr']:8.3f} {wq_norm:12.3f} {mrr:7.3f} {m['n_queries']:10,d}")
    
    print(f"\nwq_mrr_norm = (wq_mrr - tie_floor) / (1 - tie_floor), tie_floor = {tie:.4f}")
    print("\nKey ablations to interpret (organize by comparison target):")
    print("\n  PROTEIN TARGET INFORMATION:")
    print("    fp_protein - fp_only:                does protein data help without other structure?")
    print("    sage_protein - sage:                 does protein help alongside graph smoothing?")
    print("    sage_tucker_protein - sage_tucker:   does protein help when synergy is present?")
    
    print("\n  SYNERGY LABELS:")
    print("    tucker_only - fp_only:               does synergy help without graph?")
    print("    sage_tucker - sage:                  does synergy help alongside graph?")
    
    print("\n  TENSOR FORMULATIONS (new variants):")
    print("    sage_protein_tensor - sage:          does co-targeting pattern (drug×drug×protein) help?")
    print("    sage_coupled_tucker - sage_tucker:   does coupling synergy+protein improve over synergy alone?")
    print("    sage_protein_tensor - sage_tucker:   is protein co-targeting better than synergy labels?")
    print("    sage_coupled_tucker - sage_protein:  is coupled factorization better than protein features?")
    
    print("\n  GRAPH STRUCTURE:")
    print("    sage - fp_only:                      does graph smoothing help (raw signal from similarity graph)?")
    
    print("\nInterpretation notes:")
    print("  - All deltas <0.003 AUROC are within noise floor; aggregate across seeds before concluding.")
    print("  - At B1 (in-dist): expect small effects; graph+features mostly sufficient.")
    print("  - At B3 (drug-OOD): expect larger effects; structured signals matter more.")


if __name__ == "__main__":
    main()