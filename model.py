"""
NSSG-DimNet Model Architecture (Phases 2 to 5)
==============================================
Neuro-Symbolic Switch-Gated Dual-Graph Network for Code-Mixed Hinglish DimABSA.

Architecture Components:
1. Phase 2: Multilingual Transformer Backbone + Switch Positional Embeddings + Switch-Point Gated Self-Attention (SP-GSA).
2. Phase 3: Dual-Branch Processing Layer:
   - Branch A: Biaffine Aspect-Opinion Span Extractor (2D Bilinear Grid) -> (h_aspect, h_opinion).
   - Branch B: Heterogeneous Neuro-Symbolic Graph (H-NSG) with Lexicon Prior Anchor Injection + Relational Graph Attention Network (RGAT).
3. Phase 4: Aspect-Guided Mutual Cross-Attention -> Fused Span Representation Z in R^(3d).
4. Phase 5: Parallel Dual Regression Heads (GELU + Sigmoid) -> (Valence_hat, Arousal_hat) in [0.000, 1.000]^2.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any



# ---------------------------------------------------------------------------
# Phase 2: Switch Positional Embedding & Switch-Point Gated Self-Attention
# ---------------------------------------------------------------------------

class SwitchPositionalEmbedding(nn.Module):
    """
    Learned Switch Positional Embeddings encoding structural code-switching friction.
    Maps Signed Switch-Distance d_switch in [-max_dist, max_dist] (2*max_dist + 1 bins)
    to a continuous dense embedding space in R^d.
    """
    def __init__(self, hidden_dim: int, max_dist: int = 16):
        super().__init__()
        self.max_dist = max_dist
        self.num_embeddings = 2 * max_dist + 1
        self.embedding = nn.Embedding(self.num_embeddings, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, d_switch_indices: torch.Tensor) -> torch.Tensor:
        """
        Input: d_switch_indices [B, N] in [0, 2*max_dist]
        Output: E_switch [B, N, hidden_dim]
        """
        return self.embedding(d_switch_indices)


class SwitchPointGatedSelfAttention(nn.Module):
    """
    Switch-Point Gated Self-Attention (SP-GSA).
    Dynamically modulates attention weights at code-switch transition boundaries.
    Formula:
        gate = sigmoid(W_g [H || E_switch] + b_g)
        H_gated = LayerNorm(gate * H + H)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H: torch.Tensor,
        E_switch: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inputs:
            H: [B, N, d] Contextual representations
            E_switch: [B, N, d] Switch positional embeddings
            attention_mask: [B, N] (1 for valid token, 0 for pad)
        Output:
            H_gated: [B, N, d]
        """
        # 1. Compute adaptive switch-point gate
        concat_feat = torch.cat([H, E_switch], dim=-1)  # [B, N, 2d]
        gate = self.gate_proj(concat_feat)              # [B, N, d]

        # 2. Gated residual connection with LayerNorm
        H_mod = self.layer_norm(gate * H + H)           # [B, N, d]

        # 3. Key padding mask for multihead self-attention (True for padded positions)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # 4. Contextual refinement self-attention
        attn_out, _ = self.self_attn(
            H_mod, H_mod, H_mod,
            key_padding_mask=key_padding_mask
        )
        H_gated = self.post_norm(H_mod + self.dropout(attn_out)) # [B, N, d]
        return H_gated


# ---------------------------------------------------------------------------
# Phase 3 - Branch A: Biaffine Aspect-Opinion Span Extractor
# ---------------------------------------------------------------------------

class BiaffineScorer(nn.Module):
    """
    Memory-efficient 2D Bilinear / Biaffine scoring layer:
    Score(h_i, h_j) = h_i^T U h_j + W_start h_i + W_end h_j + b
    """
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.U = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        self.W_start = nn.Linear(in_features, out_features, bias=True)
        self.W_end = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.U, -bound, bound)

    def forward(self, h_start: torch.Tensor, h_end: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            h_start: [B, N, d]
            h_end:   [B, N, d]
        Output:
            scores: [B, N, N]
        """
        B, N, D = h_start.size()
        # Bilinear term: [B, N, d] x [d, d] x [B, d, N] -> [B, N, N]
        U_mat = self.U.squeeze(0) if self.out_features == 1 else self.U[0]
        bilinear = torch.matmul(torch.matmul(h_start, U_mat), h_end.transpose(1, 2)) # [B, N, N]

        # Efficient linear term: W_start(h_start)[:, :, None] + W_end(h_end)[:, None, :]
        lin_start = self.W_start(h_start) # [B, N, 1]
        lin_end = self.W_end(h_end)       # [B, N, 1]
        linear = lin_start + lin_end.transpose(1, 2) # [B, N, N]

        return bilinear + linear


class BiaffineSpanExtractor(nn.Module):
    """
    Branch A: Evaluates start and end token boundaries over 2D bilinear grids,
    and pools target aspect vector h_aspect and opinion vector h_opinion.
    """
    def __init__(self, hidden_dim: int, span_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Aspect Span Projectors
        self.asp_start_mlp = nn.Sequential(
            nn.Linear(hidden_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.asp_end_mlp = nn.Sequential(
            nn.Linear(hidden_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.asp_biaffine = BiaffineScorer(span_dim, out_features=1)

        # Opinion Span Projectors
        self.op_start_mlp = nn.Sequential(
            nn.Linear(hidden_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.op_end_mlp = nn.Sequential(
            nn.Linear(hidden_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.op_biaffine = BiaffineScorer(span_dim, out_features=1)

        # Span Representation Aggregator (Start boundary, End boundary, and Mean span context)
        self.span_compress = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def extract_span_vector(
        self,
        H: torch.Tensor,
        span_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable span boundary pooling with mean context aggregation.
        Inputs:
            H: [B, N, d]
            span_indices: [B, 2] (start_idx, end_idx)
        Output:
            h_span: [B, d]
        """
        B, N, D = H.size()
        span_vecs = []
        for b in range(B):
            s_idx = int(span_indices[b, 0].item())
            e_idx = int(span_indices[b, 1].item())
            s_idx = max(0, min(N - 1, s_idx))
            e_idx = max(s_idx, min(N - 1, e_idx))
            
            # Boundary & Mean features: [h_start || h_end || h_mean]
            h_start = H[b, s_idx]
            h_end = H[b, e_idx]
            h_mean = H[b, s_idx:e_idx + 1].mean(dim=0) # [d]
            h_combined = torch.cat([h_start, h_end, h_mean], dim=-1) # [3d]
            span_vecs.append(h_combined)

        h_bound_batch = torch.stack(span_vecs, dim=0) # [B, 3d]
        return self.span_compress(h_bound_batch)      # [B, d]

    def forward(
        self,
        H_gated: torch.Tensor,
        aspect_span: torch.Tensor,
        opinion_span: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Outputs:
            h_aspect: [B, d]
            h_opinion: [B, d]
            asp_scores: [B, N, N] 2D Bilinear Grid
            op_scores:  [B, N, N] 2D Bilinear Grid
        """
        # 1. Aspect 2D Bilinear Grid
        h_asp_s = self.asp_start_mlp(H_gated)
        h_asp_e = self.asp_end_mlp(H_gated)
        asp_grid = self.asp_biaffine(h_asp_s, h_asp_e)

        # 2. Opinion 2D Bilinear Grid
        h_op_s = self.op_start_mlp(H_gated)
        h_op_e = self.op_end_mlp(H_gated)
        op_grid = self.op_biaffine(h_op_s, h_op_e)

        # 3. Span representation vectors
        h_aspect = self.extract_span_vector(H_gated, aspect_span)
        h_opinion = self.extract_span_vector(H_gated, opinion_span)

        return h_aspect, h_opinion, asp_grid, op_grid


# ---------------------------------------------------------------------------
# Phase 3 - Branch B: Heterogeneous Neuro-Symbolic Graph (H-NSG) & Relational GAT
# ---------------------------------------------------------------------------

class RelationalGraphAttentionLayer(nn.Module):
    """
    Memory-Efficient Relational Graph Attention Network (RGAT).
    Computes decoupled attention over R=4 adjacency channels without large 5D outer products:
        alpha_ij^r = softmax( LeakyReLU( W_src h_i + W_dst h_j + b_r ) )
        h_i^(l+1) = sigma( sum_r sum_j alpha_ij^r W_rel^r h_j )
    """
    def __init__(self, hidden_dim: int, num_relations: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.rel_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)
        ])
        self.src_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(num_relations, num_heads, 1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, adj_matrices: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            H: [B, N, d]
            adj_matrices: [B, 4, N, N]
        Output:
            H_out: [B, N, d]
        """
        B, N, D = H.size()
        h_src = self.src_linear(H).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, h, N, d_h]
        h_dst = self.dst_linear(H).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, h, N, d_h]

        # Base scalar attention logit: [B, h, N, 1] + [B, h, 1, N] -> [B, h, N, N]
        base_logits = self.leaky_relu(h_src.sum(dim=-1, keepdim=True) + h_dst.sum(dim=-1, keepdim=True).transpose(-2, -1))

        H_rel_accum = torch.zeros_like(H)

        for r in range(self.num_relations):
            A_r = adj_matrices[:, r].unsqueeze(1) # [B, 1, N, N]
            logits_r = base_logits + self.rel_bias[r]

            # Mask non-edges with large negative value
            mask = (A_r == 0)
            logits_r = logits_r.masked_fill(mask, -1e9)
            attn_weights = F.softmax(logits_r, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights) # [B, h, N, N]

            # Project features for relation r
            H_proj = self.rel_projections[r](H).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, h, N, d_h]
            # Batch matrix multiply: [B, h, N, N] x [B, h, N, d_h] -> [B, h, N, d_h]
            out_r = torch.matmul(attn_weights, H_proj)
            out_r = out_r.permute(0, 2, 1, 3).contiguous().view(B, N, D) # [B, N, d]
            H_rel_accum = H_rel_accum + out_r

        H_out = self.layer_norm(H + H_rel_accum)
        return H_out


class HeterogeneousNeuroSymbolicGraph(nn.Module):
    """
    Branch B: Heterogeneous Neuro-Symbolic Graph Layer.
    Injects symbolic lexicon priors:
        h_i^(0) = LayerNorm( W_fuse [h_i || v_i || a_i || is_anchor] )
    followed by L layers of Relational Graph Attention Network (RGAT).
    """
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Prior anchor projection (concatenating Valence, Arousal, and Anchor Confidence)
        self.lexicon_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Stack of Relational GAT layers
        self.rgat_layers = nn.ModuleList([
            RelationalGraphAttentionLayer(
                hidden_dim=hidden_dim,
                num_relations=num_relations,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        H_gated: torch.Tensor,
        lexicon_priors: torch.Tensor,
        adj_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            H_gated:        [B, N, d]
            lexicon_priors: [B, N, 3] (Valence, Arousal, is_anchor)
            adj_matrices:   [B, 4, N, N]
        Output:
            H_graph:        [B, N, d]
        """
        # 1. Inject symbolic priors
        lex_feat = self.lexicon_proj(lexicon_priors)              # [B, N, 64]
        H_fused = self.fusion_layer(torch.cat([H_gated, lex_feat], dim=-1)) # [B, N, d]

        # 2. Multi-layer Relational GAT propagation
        H_curr = H_fused
        for layer in self.rgat_layers:
            H_curr = layer(H_curr, adj_matrices)

        return H_curr


# ---------------------------------------------------------------------------
# Phase 4: Aspect-Guided Mutual Cross-Attention
# ---------------------------------------------------------------------------

class AspectGuidedCrossAttention(nn.Module):
    """
    Fuses Aspect span representation, Opinion span representation, and Relational Graph representation
    using bidirectional multihead cross-attention.
    Outputs unified fused vector Z in R^(3d).
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aspect_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.opinion_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.graph_pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.norm_asp = nn.LayerNorm(hidden_dim)
        self.norm_op = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        H_graph: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inputs:
            h_aspect:  [B, d]
            h_opinion: [B, d]
            H_graph:   [B, N, d]
            attention_mask: [B, N]
        Output:
            Z: [B, 3d]
        """
        B = h_aspect.size(0)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # 1. Aspect Cross-Attention
        q_asp = h_aspect.unsqueeze(1)                          # [B, 1, d]
        attn_asp, _ = self.aspect_cross_attn(
            q_asp, H_graph, H_graph,
            key_padding_mask=key_padding_mask
        )
        h_aspect_fused = self.norm_asp(h_aspect + self.dropout(attn_asp.squeeze(1))) # [B, d]

        # 2. Opinion Cross-Attention
        q_op = h_opinion.unsqueeze(1)                          # [B, 1, d]
        attn_op, _ = self.opinion_cross_attn(
            q_op, H_graph, H_graph,
            key_padding_mask=key_padding_mask
        )
        h_opinion_fused = self.norm_op(h_opinion + self.dropout(attn_op.squeeze(1))) # [B, d]

        # 3. Graph Global Context Pooling
        attn_weights = self.graph_pool_attn(H_graph)          # [B, N, 1]
        if attention_mask is not None:
            attn_weights = attn_weights * attention_mask.unsqueeze(-1)
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        h_graph_pool = (attn_weights * H_graph).sum(dim=1)     # [B, d]
        h_graph_fused = self.norm_graph(h_graph_pool)          # [B, d]

        # 4. Concatenate into Unified Fused Span Representation Z in R^(3d)
        Z = torch.cat([h_aspect_fused, h_opinion_fused, h_graph_fused], dim=-1) # [B, 3d]
        return Z


# ---------------------------------------------------------------------------
# Phase 5: Parallel Dual Regression Heads
# ---------------------------------------------------------------------------

class ParallelDualRegressionHeads(nn.Module):
    """
    Independent parallel Multi-Layer Perceptrons for Valence (V) and Arousal (A).
    Neuro-symbolic skip-connection anchors continuous output to symbolic lexicon priors.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # in_features is 3*d + 2 (includes opinion lexicon prior [v_lex_op, a_lex_op])
        self.valence_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Learnable symbolic prior blend weights
        self.lex_blend_v = nn.Parameter(torch.tensor(0.35))
        self.lex_blend_a = nn.Parameter(torch.tensor(0.35))

    def forward(self, Z: torch.Tensor, op_lex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            Z: [B, 3d]
            op_lex: [B, 2] (v_lex_op, a_lex_op)
        Outputs:
            V_pred: [B] in [0.000, 1.000]
            A_pred: [B] in [0.000, 1.000]
        """
        Z_full = torch.cat([Z, op_lex], dim=-1) # [B, 3d + 2]
        v_neural = self.valence_head(Z_full).squeeze(-1) # [B]
        a_neural = self.arousal_head(Z_full).squeeze(-1) # [B]

        # Neuro-symbolic integration
        w_v = torch.sigmoid(self.lex_blend_v)
        w_a = torch.sigmoid(self.lex_blend_a)

        v_lex = op_lex[:, 0]
        a_lex = op_lex[:, 1]

        v_pred = (1.0 - w_v) * v_neural + w_v * v_lex
        a_pred = (1.0 - w_a) * a_neural + w_a * a_lex

        return torch.clamp(v_pred, 0.0, 1.0), torch.clamp(a_pred, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Master NSSG-DimNet Module
# ---------------------------------------------------------------------------

class NSSGDimNet(nn.Module):
    """
    Complete End-to-End NSSG-DimNet Architecture:
    - Backbone: HingRoBERTa / XLM-RoBERTa encoder (frozen for stability)
    - Switch-Point Gated Self-Attention (SP-GSA)
    - Branch A: Biaffine Span Extractor with mean context boundary pooling
    - Branch B: Heterogeneous Neuro-Symbolic Graph (H-NSG) & Relational GAT
    - Aspect-Guided Mutual Cross-Attention
    - Parallel Dual Regression Heads with Neuro-Symbolic Prior Integration
    """
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        hidden_dim: int = 768,
        max_switch_dist: int = 16,
        num_heads: int = 8,
        num_rgat_layers: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers

        # 1. Multilingual Transformer Backbone
        self.encoder = None
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(model_name)
            self.hidden_dim = self.encoder.config.hidden_size
            if self.freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
            print(f"[NSSG-DimNet] Loaded pretrained transformer encoder: {model_name} (d={self.hidden_dim}, frozen={self.freeze_backbone})", flush=True)
        except Exception as e:
            print(f"[NSSG-DimNet] Pretrained encoder '{model_name}' offline ({e}). Initializing robust Embedding+Transformer.", flush=True)
            self.vocab_size = 50000
            self.embed = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
            self.pos_embed = nn.Embedding(512, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True
            )
            self.encoder_fallback = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 2. Switch Positional Embedding & Switch-Point Gating
        self.switch_embed = SwitchPositionalEmbedding(self.hidden_dim, max_dist=max_switch_dist)
        self.sp_gsa = SwitchPointGatedSelfAttention(self.hidden_dim, num_heads=num_heads, dropout=dropout)

        # 3. Branch A: Biaffine Span Extractor
        self.branch_a = BiaffineSpanExtractor(self.hidden_dim, span_dim=128, dropout=dropout)

        # 4. Branch B: Heterogeneous Neuro-Symbolic Graph (H-NSG)
        self.branch_b = HeterogeneousNeuroSymbolicGraph(
            self.hidden_dim, num_relations=4, num_layers=num_rgat_layers, dropout=dropout
        )

        # 5. Aspect-Guided Mutual Cross-Attention
        self.cross_attn = AspectGuidedCrossAttention(self.hidden_dim, num_heads=num_heads, dropout=dropout)

        # 6. Parallel Dual Regression Heads (in_features = 3*d + 2)
        self.regression_heads = ParallelDualRegressionHeads(
            in_features=self.hidden_dim * 3 + 2,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all 5 phases of NSSG-DimNet.
        """
        input_ids = batch["input_ids"]                   # [B, N]
        attention_mask = batch["attention_mask"]         # [B, N]
        d_switch_indices = batch["d_switch_indices"]     # [B, N]
        lexicon_priors = batch["lexicon_priors"]         # [B, N, 3]
        adj_matrices = batch["adj_matrices"]             # [B, 4, N, N]
        aspect_span = batch["aspect_span"]               # [B, 2]
        opinion_span = batch["opinion_span"]             # [B, 2]

        # Opinion Lexicon Prior
        if "op_lexicon_prior" in batch:
            op_lex = batch["op_lexicon_prior"]           # [B, 2]
        else:
            # Fallback compute from lexicon_priors
            B = input_ids.size(0)
            op_lex_list = []
            for b in range(B):
                s_op = int(opinion_span[b, 0].item())
                e_op = int(opinion_span[b, 1].item())
                sub_priors = lexicon_priors[b, s_op:e_op + 1] # [k, 3]
                mask = sub_priors[:, 2] > 0
                if mask.sum() > 0:
                    v_m = sub_priors[mask, 0].mean().item()
                    a_m = sub_priors[mask, 1].mean().item()
                else:
                    v_m, a_m = 0.50, 0.50
                op_lex_list.append([v_m, a_m])
            op_lex = torch.tensor(op_lex_list, dtype=torch.float32, device=input_ids.device)

        # Phase 2: Encoding
        if self.encoder is not None:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                H = outputs.last_hidden_state            # [B, N, d]
        else:
            B, N = input_ids.size()
            positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, N)
            H_emb = self.embed(input_ids) + self.pos_embed(positions)
            src_key_padding_mask = (attention_mask == 0)
            H = self.encoder_fallback(H_emb, src_key_padding_mask=src_key_padding_mask)

        # Switch Positional Embeddings
        E_switch = self.switch_embed(d_switch_indices)   # [B, N, d]

        # Switch-Point Gated Self-Attention (SP-GSA)
        H_gated = self.sp_gsa(H, E_switch, attention_mask=attention_mask) # [B, N, d]

        # Phase 3: Dual-Branch Processing
        # Branch A: Biaffine Span Extractor
        h_aspect, h_opinion, asp_grid, op_grid = self.branch_a(
            H_gated, aspect_span, opinion_span
        )

        # Branch B: Heterogeneous Neuro-Symbolic Graph (H-NSG) & RGAT
        H_graph = self.branch_b(H_gated, lexicon_priors, adj_matrices) # [B, N, d]

        # Phase 4: Aspect-Guided Mutual Cross-Attention
        Z = self.cross_attn(h_aspect, h_opinion, H_graph, attention_mask=attention_mask) # [B, 3d]

        # Phase 5: Continuous Prediction Head
        v_pred, a_pred = self.regression_heads(Z, op_lex) # [B], [B]

        return {
            "valence_pred": v_pred,
            "arousal_pred": a_pred,
            "aspect_grid": asp_grid,
            "opinion_grid": op_grid,
            "fused_representation": Z,
            "h_aspect": h_aspect,
            "h_opinion": h_opinion
        }


if __name__ == "__main__":
    print("=== Testing NSSG-DimNet Architecture ===")
    from data_pipeline import build_data_loaders

    train_ld, _, _, _, _, _ = build_data_loaders(batch_size=4)
    batch = next(iter(train_ld))

    model = NSSGDimNet(model_name="xlm-roberta-base", hidden_dim=768)
    outputs = model(batch)

    print("\n--- Model Output Dimension Tracking ---")
    print(f"  Valence Prediction (V_hat): shape = {list(outputs['valence_pred'].shape)}, values = {outputs['valence_pred'].tolist()}")
    print(f"  Arousal Prediction (A_hat): shape = {list(outputs['arousal_pred'].shape)}, values = {outputs['arousal_pred'].tolist()}")
    print(f"  Fused Representation (Z)   : shape = {list(outputs['fused_representation'].shape)}")
    print(f"  Aspect Bilinear Grid       : shape = {list(outputs['aspect_grid'].shape)}")
    print(f"  Opinion Bilinear Grid      : shape = {list(outputs['opinion_grid'].shape)}")
    print("\nNSSG-DimNet Forward Pass Verification Succeeded!")
