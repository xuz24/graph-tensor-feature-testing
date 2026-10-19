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

from rdkit import Chem
import math
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise SystemExit(
        "torch_geometric is required (pip install torch_geometric). "
        f"Original error: {e}"
    )
    
    
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

SYMBOLS = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V',
           'K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni',
           'Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']   # 44
SMALL = list(range(11))                                         # 0..10

def _onehot_unk(x, allowed):
    if x not in allowed:
        x = allowed[-1]
    return [x == a for a in allowed]

def atom_features(atom):
    f = (_onehot_unk(atom.GetSymbol(), SYMBOLS)        # 44
         + _onehot_unk(atom.GetDegree(), SMALL)        # 11
         + _onehot_unk(atom.GetTotalNumHs(), SMALL)    # 11
         + _onehot_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), SMALL)  # 11
         + [atom.GetIsAromatic()])                     # 1  -> 78
    f = np.array(f, dtype=np.float32)
    return f / f.sum()

def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    x = torch.tensor(np.stack([atom_features(a) for a in mol.GetAtoms()]))
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty(2, 0, dtype=torch.long))
    return Data(x=x, edge_index=edge_index)

class DrugGCN(nn.Module):
    """Molecular graph -> one vector per drug (original: 78 -> 78 -> 156, mean pool)."""
    def __init__(self, in_dim=78, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, in_dim)
        self.conv2 = GCNConv(in_dim, in_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = in_dim * 2

    def forward(self, batch):                      # batch: torch_geometric Batch
        x, ei = batch.x, batch.edge_index
        x = self.dropout(F.relu(self.conv1(x, ei)))
        x = self.dropout(F.relu(self.conv2(x, ei)))
        return global_mean_pool(x, batch.batch)    # [num_drugs, 156]

class CellExprMLP(nn.Module):
    """Gene expression vector -> fixed-size cell embedding (original: C -> C -> 768)."""
    def __init__(self, in_dim, out_dim=768, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim   # original keeps hidden = C
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):          # x: [B, C]
        return self.net(x)         # [B, out_dim]


def _block(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(), nn.BatchNorm1d(d_hidden),
        nn.Linear(d_hidden, d_out),
    )


class FusionHead(nn.Module):
    """Component 5: MLP on the concatenated stream vectors -> one logit per row.

    mlp=True  : in -> hidden -> hidden//2 -> 1 with ReLU + BatchNorm (the original fc_network).
    mlp=False : a single Linear(in, 1) readout -- the "no fusion MLP" ablation.
    """
    def __init__(self, in_dim, hidden=8192, mlp=True):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.BatchNorm1d(hidden // 2),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)       # [B]


#-------------------------------#
# copied from tensograph github #
#-------------------------------#    
class GTN(nn.Module):
    #初始化
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):  # layers多个GTLayer组成的; 多头channels
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))  # 第一个GT层,edge类别构建的矩阵
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))  # GCN
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):  # 自己写了一个GCN
        X = torch.mm(X, self.weight)  # X-features; self.weight-weight
        H = self.norm(H, add=True)  # H-第i个channel下邻接矩阵;
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)  # Q1
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)  # Q2
        return H_

    def norm(self, H, add=False):
        # Device-safe rewrite of the original, which built dense CPU identity / diagonal matrices
        # (torch.eye(...).type(torch.FloatTensor)) and so failed for GPU inputs and cost several
        # N x N matmuls. Same math: transpose, zero the diagonal (or set it to 1 when add=True),
        # scale each row by 1/degree (1/0 -> 0), transpose back.
        H = H.t().clone()
        H.fill_diagonal_(1.0 if add else 0.0)
        deg_inv = H.sum(dim=1).pow(-1)
        deg_inv[torch.isinf(deg_inv)] = 0
        return (deg_inv.unsqueeze(1) * H).t()

    def forward(self, A, X):
        A = A.unsqueeze(0).permute(0,3,1,2)   # A.unsqueeze(0)=[1,N,N,edgeType]=>[1,edgeType,N,N]; 卷积输出的channel数量
        Ws = []
        for i in range(self.num_layers):  # 两层GTLayer:{edgeType}
            if i == 0:
                H, W = self.layers[i](A)  # GTN0:两层GTConv; A:edgeType的邻接矩阵; output: H(A(l)), W:归一化的Conv
            else:
                H = self.normalization(H)   # Conv矩阵，D-1*A的操作
                H, W = self.layers[i](A, H)  # 第一层计算完了A(原始矩阵), H(上一次计算后的A(l)); output: A2, W(第二层Conv1)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)
        semantic_embeddings = []
        for i in range(self.num_channels):   # conv的channel数量
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))  # X-features; H[i]-第i个channel输出的邻接矩阵Al[i]; gcn_conv:Linear
                semantic_embeddings.append(X_)
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                semantic_embeddings.append(X_tmp)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return  semantic_embeddings.detach()
    
    
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return ((beta * z).sum(1)).detach() # (N, D * K)


class GTLayer(nn.Module):
    # 不同edge类型的组合
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # 1x1卷积的channel数量
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)  # W1
            self.conv2 = GTConv(in_channels, out_channels)  # W2
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):  # A:[1,edgeType,N,N]
        if self.first == True:
            a = self.conv1(A)  # GTConv=>[2, N, N] #Q1
            b = self.conv2(A)  # Q2
            #*** 作了第一次矩阵相乘，得到A1
            H = torch.bmm(a,b)  # torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3;
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]  # conv-softmax: 是为了下一次直接使用吗？
        else:
            a = self.conv1(A)  # 第二层只有一个conv1; output:Conv输出归一化edge后的结果
            H = torch.bmm(H_,a)  # H_上一层的输出矩阵A1; 输出这一层后的结果A2;
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W   # H = A(1) ... A(l); W = 归一化后的权重矩阵


class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))  #
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):  # self.weight:带有channel的conv;
        '''
        0) 对weight(conv)进行softmax
        1) 对每个节点在每个edgeType上进行[2, 5, 1, 1]的卷积操作;
        2) 对每个edgeType进行加权求和，加权是通过0)softmax
        '''
        # F.softmax(self.weight, dim=1) 对self.weight做softmax:[2, 5, 1, 1]
        # A: [1, 5, 8994, 8994]:带有edgeType的邻接矩阵
        # [1, 5, 8994, 8994]*[2, 5, 1, 1] => [2, 5, 8994, 8994]
        # sum:[2, 8994, 8994]
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A


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


#-----------------------------------------------------------#
# TensoGraph five-component ablation (separate from the     #
# legacy SynergyModel arms; nothing above this line uses it) #
#-----------------------------------------------------------#
_STREAMS = ("gtn", "gcn", "tucker")
_FULL = dict(gtn=True, gcn=True, tucker=True, expr=True, stream_mlps=True, fusion_mlp=True)


def _without(*keys):
    return {**_FULL, **{k: False for k in keys}}


def _only(stream):
    return {**_FULL, **{s: (s == stream) for s in _STREAMS}}


# Component map:  1 = gtn   2 = gcn   3 = expr   4 = stream_mlps   5 = fusion_mlp
# `tucker` is the existing synergy Tucker embedding, entering as the fourth stream of component 4.
TG_ARMS = {
    "tg_full":           dict(_FULL),
    "tg_no_gtn":         _without("gtn"),            # drop component 1
    "tg_no_gcn":         _without("gcn"),            # drop component 2
    "tg_no_expr":        _without("expr"),           # drop component 3 (cell-ID embedding instead)
    "tg_no_stream_mlps": _without("stream_mlps"),    # drop component 4 (raw streams -> head)
    "tg_no_fusion_mlp":  _without("fusion_mlp"),     # drop component 5 (linear readout)
    "tg_no_tucker":      _without("tucker"),         # extra: the Tucker stream itself
    "tg_gtn_only":       _only("gtn"),               # single drug stream (+ expr, MLPs, head)
    "tg_gcn_only":       _only("gcn"),
    "tg_tucker_only":    _only("tucker"),
}


class TensoGraphAblation(nn.Module):
    """The five TensoGraph components, each switchable.

    drug_streams : {"gtn"|"gcn"|"tucker": [N, d] tensor}  precomputed per-drug vectors
                   (component 1 = GTN output, component 2 = GCN output, plus Tucker).
    expr         : [num_cells, C] standardised expression -> component 3 (CellExprMLP). If None the
                   cell is represented by a learned ID embedding (id_cell_dim > 0) or omitted
                   (id_cell_dim == 0).
    stream_mlps  : component 4 -- one x -> 2x -> x MLP per stream (drug streams are shared by
                   drug 1 and drug 2; the cell vector gets its own).
    fusion_mlp   : component 5 -- FusionHead on [drug_i streams | drug_j streams | cell].

    The fusion input is order-dependent, as in the original. During training the two drugs are
    swapped at random per row; at eval the logits of both orders are averaged.
    """
    def __init__(self, drug_streams, num_cells, *, expr=None, stream_mlps=True, fusion_mlp=True,
                 cell_dim=768, expr_hidden=512, fusion_hidden=8192, id_cell_dim=32):
        super().__init__()
        assert drug_streams, "need at least one drug stream"
        self.stream_names = list(drug_streams)
        for n, t in drug_streams.items():
            self.register_buffer(f"stream_{n}", t.float())
        dims = {n: int(t.shape[1]) for n, t in drug_streams.items()}

        self.use_stream_mlps = stream_mlps
        self.stream_nets = nn.ModuleDict(
            {n: _block(d, 2 * d, d) for n, d in dims.items()} if stream_mlps else {})

        self.has_expr = expr is not None
        if self.has_expr:
            self.register_buffer("expr", expr.float())
            self.cell_enc = CellExprMLP(expr.shape[1], out_dim=cell_dim, hidden_dim=expr_hidden)
            cdim = cell_dim
        elif id_cell_dim:
            self.cell_enc = nn.Embedding(num_cells, id_cell_dim)
            cdim = id_cell_dim
        else:
            self.cell_enc = None
            cdim = 0
        self.cell_post = _block(cdim, 2 * cdim, cdim) if (stream_mlps and cdim) else nn.Identity()

        self.head = FusionHead(2 * sum(dims.values()) + cdim, hidden=fusion_hidden, mlp=fusion_mlp)

    @classmethod
    def from_arm(cls, arm, streams, expr, num_cells, **kw):
        cfg = TG_ARMS[arm]
        sel = {k: streams[k] for k in _STREAMS if cfg[k]}
        return cls(sel, num_cells, expr=expr if cfg["expr"] else None,
                   stream_mlps=cfg["stream_mlps"], fusion_mlp=cfg["fusion_mlp"], **kw)

    def _drug_vec(self, idx):
        outs = []
        for n in self.stream_names:
            x = getattr(self, f"stream_{n}")[idx]
            if self.use_stream_mlps:
                x = self.stream_nets[n](x)
            outs.append(x)
        return torch.cat(outs, 1)

    def _cell_vec(self, c):
        if self.cell_enc is None:
            return None
        v = self.cell_enc(self.expr[c]) if self.has_expr else self.cell_enc(c)
        return self.cell_post(v)

    def _fuse(self, vi, vj, cv):
        parts = [vi, vj] if cv is None else [vi, vj, cv]
        return self.head(torch.cat(parts, 1))

    def forward(self, i_idx, j_idx, cell_idx):
        cv = self._cell_vec(cell_idx)
        if self.training:
            swap = torch.rand(i_idx.shape[0], device=i_idx.device) < 0.5
            i_idx, j_idx = torch.where(swap, j_idx, i_idx), torch.where(swap, i_idx, j_idx)
            return self._fuse(self._drug_vec(i_idx), self._drug_vec(j_idx), cv)
        vi, vj = self._drug_vec(i_idx), self._drug_vec(j_idx)
        return 0.5 * (self._fuse(vi, vj, cv) + self._fuse(vj, vi, cv))     # [B] logits
