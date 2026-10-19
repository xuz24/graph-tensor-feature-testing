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

import re

from torch_geometric.data import Batch

from .models import DrugGCN, GTN, SemanticAttention, smiles_to_data



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
# 5. TensoGraph five-component ablation: data builders. Nothing above this line depends on it.
# ═══════════════════════════════════════════════════════════════════════════════════════════
def l2norm(t):
    """Row-wise L2 normalisation, identical to the inline fp_norm / tuck_norm in the legacy main."""
    return t / (t.norm(dim=1, keepdim=True) + 1e-8)


def load_fingerprints(data_root, idx, N):
    """AtomPair fingerprints aligned to graph drug indices (rows without a fingerprint stay 0)."""
    d = Path(data_root) / "drug_features"
    for name in ("drug_atompair.npz", "drug_atompair1024_full_embeddings.npz"):
        if (d / name).exists():
            path = d / name
            break
    else:
        raise FileNotFoundError(f"no AtomPair file in {d}")
    z = np.load(path, allow_pickle=True)
    ids = [str(x).strip().upper() for x in z["ids"]]
    emb = z["embeddings"]
    fp = np.zeros((N, emb.shape[1]), np.float32)
    for i, dname in enumerate(ids):
        if dname in idx:
            fp[idx[dname]] = emb[i]
    return fp


# ── component 1: GTN on the drug-drug graph ──────────────────────────────────────────────────
def build_synergy_channels(N, d1, d2, cell, y, train_idx, test_idx, e_sim):
    """The three channels of the existing Tucker tensor -- [synergy_pos, synergy_neg, similarity]
    -- as separate (N, N) matrices. Built exactly as in build_tucker_embeddings(mode="synergy"):
    same leak-safe build_synergy_edges() call (train rows only), same similarity graph."""
    _, sign_of_edge, diag = build_synergy_edges(d1, d2, cell, y, train_idx, test_idx,
                                                key="pair", attr="cell", sign="signed")
    print(f"  [gtn] synergy channels (pair,signed): {diag['n_edges']} edges, "
          f"{diag['n_keys']} keys, test coverage {diag['coverage']:.1%}")
    pos = np.zeros((N, N), np.float32)
    neg = np.zeros((N, N), np.float32)
    for (a, b), sgn in sign_of_edge.items():
        (pos if sgn > 0 else neg)[a, b] = (pos if sgn > 0 else neg)[b, a] = 1.0
    sim = np.zeros((N, N), np.float32)
    ei = e_sim.numpy()
    sim[ei[0], ei[1]] = 1.0
    return pos, neg, sim


@torch.no_grad()
def build_gtn_embeddings(pos, neg, sim, X, device, node_dim=32, num_channels=2, num_layers=2,
                         seed=0):
    """Global (not per-cell-line) GTN embedding, [N, node_dim].

    Edge types are [pos, neg, sim, identity] -- the TensoGraph layout (three relation matrices +
    identity) with our three channels. X is the node-feature matrix (AtomPair here; the original
    used infomax features, which are not in the data bundle).

    Like the original, the GTN and SemanticAttention are NOT trained: weights come from a seeded
    init and forward() returns detached outputs. Peak memory at N=5,519 is a few GB on `device`.
    """
    N = pos.shape[0]
    with torch.random.fork_rng(devices=[]):          # don't disturb the global RNG stream
        torch.manual_seed(seed)
        gtn = GTN(num_edge=4, num_channels=num_channels, w_in=X.shape[1], w_out=node_dim,
                  num_class=2, num_layers=num_layers, norm="true").to(device)
        sem = SemanticAttention(in_size=node_dim).to(device)
    A = torch.from_numpy(np.stack([pos, neg, sim], axis=-1)).to(device)          # [N, N, 3]
    A = torch.cat([A, torch.eye(N, device=device).unsqueeze(-1)], dim=-1)        # [N, N, 4]
    out = sem(gtn(A, X.to(device)))                                              # [N, node_dim]
    del A
    return out.cpu()


# ── component 2: molecular GCN ───────────────────────────────────────────────────────────────
def load_smiles(path, dbid_to_idx):
    """molecules.csv (molecule_id = DrugBank id, canonical_isomeric_smiles) -> {drug_idx: smiles}."""
    df = pd.read_csv(path)
    out = {}
    for dbid, smi in zip(df["molecule_id"], df["canonical_isomeric_smiles"]):
        i = dbid_to_idx.get(str(dbid).strip().upper())
        if i is not None and isinstance(smi, str) and smi:
            out.setdefault(i, smi)
    return out


def build_mol_batch(smiles_by_idx):
    """Parse every SMILES (largest fragment, so salts/counter-ions drop out) into one PyG Batch.
    Returns (batch, rows): rows[k] is the drug index of graph k. Unparseable SMILES are skipped."""
    graphs, rows, failed = [], [], 0
    errors = []
    for i, smi in sorted(smiles_by_idx.items()):
        try:
            graphs.append(smiles_to_data(max(smi.split("."), key=len)))
            rows.append(i)
        except Exception as exc:
            failed += 1
            if len(errors) < 3:
                errors.append((i, type(exc).__name__, str(exc)))
    if failed:
        print(f"  [gcn] {failed} molecules could not be converted and get a zero vector")
        for i, kind, message in errors:
            print(f"  [gcn] example conversion error for drug {i}: {kind}: {message}")
    if not graphs:
        raise ValueError("no SMILES could be parsed")
    return Batch.from_data_list(graphs), torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def build_gcn_embeddings(batch, rows, N, device, seed=0):
    """Frozen DrugGCN (78 -> 78 -> 156, mean pool), eval mode so dropout is off. Seeded init, not
    trained -- as in the original. Drugs without a parsed SMILES keep a zero row. -> [N, 156]."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gcn = DrugGCN()
    gcn = gcn.to(device).eval()
    out = torch.zeros(N, gcn.out_dim)
    out[rows] = gcn(batch.to(device)).cpu()
    return out


# ── component 3: cell-line gene expression ───────────────────────────────────────────────────
def _norm_key(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _to_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def infer_cell_names(fold_table, cell, y, num_cells):
    """cell index -> cell_name, recovered from the fold table itself.

    GAS.load() returns integer `cell` ids but not the name mapping, so rather than guessing its
    indexing rule we read the (cell_name, cell id) pairs off the table and require them to be
    one-to-one. Fails loudly if the table and GAS.load() are not row-aligned."""
    df = pd.read_csv(fold_table)
    if "cell_name" not in df.columns:
        raise ValueError(f"{fold_table} has no 'cell_name' column (columns: {list(df.columns)})")
    cell = np.asarray(cell).astype(int)
    if len(df) != len(cell):
        raise ValueError(f"fold table has {len(df)} rows but GAS.load() returned {len(cell)}; "
                         f"cannot recover cell names by row alignment")
    if "label" in df.columns and not np.allclose(df["label"].to_numpy(dtype=float),
                                                 np.asarray(y, dtype=float)):
        raise ValueError("fold-table labels differ from GAS.load() labels: rows are not aligned")
    pairs = pd.DataFrame({"name": df["cell_name"].astype(str).str.strip(),
                          "idx": cell}).drop_duplicates()
    if not (pairs["name"].is_unique and pairs["idx"].is_unique):
        raise ValueError("cell_name <-> cell index is not one-to-one in the fold table")
    names = [None] * num_cells
    for n, i in zip(pairs["name"], pairs["idx"]):
        names[i] = n
    return names


def _expression_matrix(z):
    """(ids, X[n_cells, n_genes]) from an expression .npz without assuming key names: the matrix
    is the largest 2-D numeric array, the ids the 1-D string array matching one of its axes."""
    arr = {k: z[k] for k in z.files}
    mats = [k for k, v in arr.items() if v.ndim == 2 and v.dtype.kind in "fiu"]
    if not mats:
        raise ValueError(f"no 2-D numeric array in expression file (keys: {list(arr)})")
    X = arr[max(mats, key=lambda k: arr[k].size)]
    strs = [k for k, v in arr.items() if v.ndim == 1 and v.dtype.kind in "OUS"]
    pref = ("ids", "cell_ids", "cell_lines", "cell_line_names", "cell_names", "cells",
            "model_ids", "models", "names", "index")

    def pick(n):
        c = [k for k in strs if len(arr[k]) == n]
        return min(c, key=lambda k: pref.index(k) if k in pref else len(pref)) if c else None

    k = pick(X.shape[0])
    if k is None:
        k = pick(X.shape[1])
        if k is not None:
            X = X.T
    if k is None:
        raise ValueError("could not find a cell-id array matching the expression matrix "
                         f"(shape {X.shape}, string arrays: {[(s, len(arr[s])) for s in strs]})")
    return [_to_str(v).strip() for v in arr[k]], np.nan_to_num(X.astype(np.float32))


def load_cell_expression(expr_path, ccle_map_path, cell_names, train_cells):
    """[num_cells, n_genes] standardised expression + boolean mask of cells that have data.

    Cell names are matched to expression rows directly (normalised, also trying the part of a CCLE
    id before the first '_'), then through cell_line_ccle_map.csv (any two columns whose values hit
    one cell name and one expression id on the same row). Genes are z-scored with statistics from
    TRAINING cells only; cells without expression keep an all-zero row (= the training mean).
    """
    z = np.load(expr_path, allow_pickle=True)
    print(f"  [expr] {Path(expr_path).name}: " + ", ".join(f"{k}{z[k].shape}" for k in z.files))
    ids, X = _expression_matrix(z)

    index = {}
    for r, rid in enumerate(ids):
        for key in {_norm_key(rid), _norm_key(rid.split("_")[0])}:
            index.setdefault(key, r)
    wanted = {_norm_key(n): i for i, n in enumerate(cell_names) if n}

    row_of = {ci: index[k] for k, ci in wanted.items() if k in index}
    n_direct = len(row_of)
    if ccle_map_path is not None and Path(ccle_map_path).exists():
        m = pd.read_csv(ccle_map_path, dtype=str).fillna("")
        for rec in m.itertuples(index=False):
            toks = [_norm_key(v) for v in rec if v]
            cells_hit = [wanted[t] for t in toks if t in wanted]
            rows_hit = [index[t] for t in toks if t in index]
            if cells_hit and rows_hit:
                for ci in cells_hit:
                    row_of.setdefault(ci, rows_hit[0])
    if not row_of:
        raise ValueError("no cell line could be matched to an expression row; inspect the npz "
                         "ids and cell_line_ccle_map.csv columns and adapt load_cell_expression")

    E = np.zeros((len(cell_names), X.shape[1]), np.float32)
    have = np.zeros(len(cell_names), bool)
    for ci, r in row_of.items():
        E[ci], have[ci] = X[r], True
    tr_have = [c for c in train_cells if have[c]]
    if len(tr_have) >= 2:
        mu, sd = E[tr_have].mean(0), E[tr_have].std(0)
    else:
        mu, sd = np.zeros(E.shape[1], np.float32), np.ones(E.shape[1], np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    E[have] = (E[have] - mu) / sd
    n_names = sum(n is not None for n in cell_names)
    print(f"  [expr] matrix {X.shape[0]} cells x {X.shape[1]} genes; matched "
          f"{int(have.sum())}/{n_names} cell lines ({n_direct} directly, "
          f"{int(have.sum()) - n_direct} via ccle map); z-scored on {len(tr_have)} training "
          f"cells; {n_names - int(have.sum())} cell lines get a zero row")
    return torch.from_numpy(E), have
