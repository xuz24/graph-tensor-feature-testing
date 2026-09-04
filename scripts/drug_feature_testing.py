from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import comb
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


from script.metrics import *

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

