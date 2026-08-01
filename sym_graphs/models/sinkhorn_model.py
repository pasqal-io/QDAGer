from __future__ import annotations

import torch
import torch.nn as nn

from sym_graphs.models.mpnn_layer import MPNNBaselineLayer
from sym_graphs.models.simple_model import MLP
from sym_graphs.models.transformer_layer import QDAGerLayer
from sym_graphs.utils.utils import asymm_embed_mat_l1_dist, sample_gumbel_like

# ============================================================
# PermNet (https://arxiv.org/abs/2409.17687 paper Eq. 16)
#   - project nodes with c_phi
#   - L1 cost matrix
#   - Sinkhorn on exp(-C/tau) (optionally with Gumbel noise)
#   - IMPORTANT: padded nodes are zeroed BEFORE cost (dummy rows/cols)
# ============================================================


class PermNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tau = float(cfg.sinkhorn.tau)
        self.iters = int(cfg.sinkhorn.iters)
        self.noise = float(getattr(cfg.sinkhorn, "noise_factor", 0.0))

        if cfg.transformer_layer is not None:
            self.layer = cfg.transformer_layer
        else:
            self.layer = cfg.mpnn_layer

        in_dim = int(self.layer.out_dim)
        hid_dim = int(getattr(cfg.sinkhorn, "perm_hid_dim", in_dim))
        out_dim = int(getattr(cfg.sinkhorn, "perm_out_dim", in_dim))

        self.c_phi = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    @staticmethod
    def _log_sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
        """
        logits:  [B, N, N] (paper calls it log_alpha, but it can be any real-valued scores)
        returns: [B, N, N] doubly-stochastic approx (prob space)
        """
        B, N, _ = logits.shape
        for _ in range(iters):
            logits = logits - torch.logsumexp(logits, dim=2, keepdim=True).view(B, N, 1)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True).view(B, 1, N)
        return torch.exp(logits)

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        m1_x: torch.Tensor,
        m2_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        X1,X2:   [B,N,D]
        m1_x,m2_x: [B,N] (0/1 or bool)
        returns: P_node [B,N,N]
        """
        dtype = X1.dtype
        device = X1.device

        v1 = (m1_x != 0).to(dtype=dtype, device=device)  # [B,N]
        v2 = (m2_x != 0).to(dtype=dtype, device=device)

        # project
        z1 = self.c_phi(X1)  # [B,N,d']
        z2 = self.c_phi(X2)

        # padded nodes must be true dummies (no bias leakage into cost)
        z1 = z1 * v1.unsqueeze(-1)
        z2 = z2 * v2.unsqueeze(-1)

        # L1 cost
        C = (z1[:, :, None, :] - z2[:, None, :, :]).abs().sum(dim=-1)  # [B,N,N]

        # logits = -C/tau (+ optional gumbel noise)
        logits = (-C / max(self.tau, 1e-8)).clamp(-30.0, 30.0)

        if self.noise > 0.0 and self.training:
            logits = logits + self.noise * sample_gumbel_like(logits)

        # no explicit masking: padded rows/cols are handled as dummy supply/demand
        return self._log_sinkhorn(logits, self.iters)


# ============================================================
# GED Surrogate (https://arxiv.org/abs/2409.17687)
#   - edge plan S derived from P_node via straight/cross
#   - XOR on edges is the key trick to avoid "0-edge matches dominate"
#   - node surrogate: diffalign vs aligndiff
# ============================================================


class GEDSurrogate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.transformer_layer is not None:
            self.layer = cfg.transformer_layer
        else:
            self.layer = cfg.mpnn_layer

        self.N_max = int(self.layer.N_max)

        # costs
        self.node_ins_cost = float(cfg.surrogate_loss.node_add)
        self.node_del_cost = float(cfg.surrogate_loss.node_del)
        self.edge_ins_cost = float(cfg.surrogate_loss.edge_add)
        self.edge_del_cost = float(cfg.surrogate_loss.edge_del)
        self.LAMBDA = float(cfg.surrogate_loss.LAMBDA)

        # structure / switches
        self.use_max = bool(cfg.surrogate_loss.use_max)
        self.use_second_sinkhorn = bool(cfg.surrogate_loss.use_second_sinkhorn)
        self.use_second_sinkhorn_log = bool(cfg.surrogate_loss.use_second_sinkhorn_log)

        self.use_h_hp_node = bool(cfg.surrogate_loss.use_h_hp_node)
        self.use_m_ms_edge = bool(cfg.surrogate_loss.use_m_ms_edge)

        self.xor_on_edge = bool(cfg.surrogate_loss.xor_on_edge)
        self.xor_on_node = bool(cfg.surrogate_loss.xor_on_node)

        self.node_surrogate = str(
            getattr(cfg.surrogate_loss, "node_surrogate", "diffalign"),
        ).lower()
        if self.node_surrogate not in {"diffalign", "aligndiff"}:
            raise ValueError(
                "cfg.surrogate_loss.node_surrogate must be 'diffalign' or 'aligndiff'",
            )

        # Optional: compute edge transport plan from edge embeddings (E1/E2).
        # Does nothing unless we turn it on in the cfg.
        self.edge_plan_from_embeddings = bool(
            getattr(cfg.surrogate_loss, "edge_plan_from_embeddings", False),
        )
        self.edge_cost_p = float(getattr(cfg.surrogate_loss, "edge_cost_p", 1))

        # sinkhorn knobs (for optional second sinkhorn on edges)
        self.sinkhorn_temp = float(cfg.sinkhorn.tau)
        self.sinkhorn_noise = float(cfg.sinkhorn.noise_factor)
        self.sinkhorn_iters = int(cfg.sinkhorn.iters)

        # combinatorics: K = nC2 edges for N_max nodes
        K = (self.N_max * (self.N_max - 1)) // 2
        self.node_set_size_nC2 = K

        # list of all (i,k) with i<k
        src_dst = torch.ones(self.N_max, self.N_max).triu(1).nonzero()
        self.register_buffer("source_destination_list", src_dst, persistent=False)
        self.register_buffer("source_list", src_dst[:, 0], persistent=False)
        self.register_buffer("destination_list", src_dst[:, 1], persistent=False)

        # all pairs of edge indices (e1, e2) out of K
        edge_source_dest_idx = torch.ones(K, K).nonzero()
        self.register_buffer("edge_source_dest_idx", edge_source_dest_idx, persistent=False)
        self.register_buffer("edge_source_idx", edge_source_dest_idx[:, 0], persistent=False)
        self.register_buffer("edge_dest_idx", edge_source_dest_idx[:, 1], persistent=False)

        # lookup: [K^2,4] = (i1,k1,i2,k2)
        src_pairs = src_dst[self.edge_source_idx]  # [K^2,2]
        dst_pairs = src_dst[self.edge_dest_idx]  # [K^2,2]
        node_perm_lookup_idx = torch.cat((src_pairs, dst_pairs), dim=-1)
        self.register_buffer("node_perm_lookup_idx", node_perm_lookup_idx, persistent=False)
        self.mask_dummy_edges_in_surrogate = bool(
            getattr(cfg.surrogate_loss, "mask_dummy_edges_in_surrogate", False),
        )

    def _sinkhorn_edges(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Paper-style log-space Sinkhorn (same algebra as model_utils.pytorch_sinkhorn_iters).

        logits: [B,K,K] (paper calls it log_alpha)
        returns: [B,K,K] approx doubly-stochastic (prob space)
        """
        B, K, _ = logits.shape

        if self.sinkhorn_noise > 0.0 and self.training:
            logits = logits + self.sinkhorn_noise * sample_gumbel_like(logits)

        logits = logits / max(self.sinkhorn_temp, 1e-8)

        for _ in range(self.sinkhorn_iters):
            logits = logits - torch.logsumexp(logits, dim=2, keepdim=True).view(B, K, 1)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True).view(B, 1, K)

        return torch.exp(logits)

    def forward(self, Batch, P_node: torch.Tensor) -> torch.Tensor:
        """
        P_node: [B,N,N] from PermNet
        Returns: [B] surrogate GED
        """
        X1, X2 = Batch.X1, Batch.X2
        E1, E2 = Batch.E1, Batch.E2
        m1_x, m2_x = Batch.m1_x, Batch.m2_x
        m1_e, m2_e = Batch.m1_e, Batch.m2_e

        B, N, _ = X1.shape
        dtype = X1.dtype
        device = X1.device

        # Ensure P_node dtype/device consistent
        P_node = P_node.to(device=device, dtype=dtype)

        # STRICT: zero padded embeddings before distance computations
        X1 = X1 * (m1_x != 0).to(dtype).unsqueeze(-1)
        X2 = X2 * (m2_x != 0).to(dtype).unsqueeze(-1)
        E1 = E1 * (m1_e != 0).to(dtype).unsqueeze(-1)
        E2 = E2 * (m2_e != 0).to(dtype).unsqueeze(-1)

        # valid indicators
        valid1 = (m1_x != 0).long()  # [B,N]
        valid2 = (m2_x != 0).long()

        # ------------------------------
        # Node XOR / indicator
        # ------------------------------
        if self.xor_on_node:
            node_pairwise_indicator = (valid1[:, :, None] ^ valid2[:, None, :]).to(
                dtype,
            )  # [B,N,N]
        else:
            node_pairwise_indicator = torch.ones((B, N, N), device=device, dtype=dtype)

        # ------------------------------
        # Node surrogate
        # ------------------------------
        if self.node_surrogate == "diffalign":
            weights_node = P_node * node_pairwise_indicator

            if self.use_h_hp_node:
                node_diff = X1[:, :, None, :] - X2[:, None, :, :]
                del_term = torch.relu(node_diff).sum(dim=-1)
                ins_term = torch.relu(-node_diff).sum(dim=-1)
                node_align_dist = (
                    weights_node * (self.node_del_cost * del_term + self.node_ins_cost * ins_term)
                ).sum(dim=(1, 2))
            else:
                node_align_dist = asymm_embed_mat_l1_dist(
                    X1,
                    X2,
                    weights_node,
                    ins_cost=self.node_ins_cost,
                    del_cost=self.node_del_cost,
                )

        elif self.node_surrogate == "aligndiff":
            X2_bar = torch.matmul(P_node, X2)  # [B,N,D]
            diff = X1 - X2_bar
            del_term = torch.relu(diff).sum(dim=-1)  # [B,N]
            ins_term = torch.relu(-diff).sum(dim=-1)  # [B,N]
            gate = (m1_x != 0).to(dtype)  # only charge on real nodes of G1
            node_align_dist = (
                gate * (self.node_del_cost * del_term + self.node_ins_cost * ins_term)
            ).sum(dim=1)

        else:
            raise RuntimeError("unreachable")

        # ------------------------------
        # Edge transport plan from P_node (straight/cross)
        # ------------------------------
        idx = self.node_perm_lookup_idx  # [K^2,4]
        i1, k1, i2, k2 = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

        straight_score = P_node[:, i1, i2] * P_node[:, k1, k2]
        cross_score = P_node[:, i1, k2] * P_node[:, k1, i2]

        edge_scores_flat = (
            torch.maximum(straight_score, cross_score)
            if self.use_max
            else (straight_score + cross_score)
        )

        K = self.node_set_size_nC2
        edge_logits = edge_scores_flat.view(B, K, K)  # [B,K,K]

        # optional second sinkhorn on edges
        # NOTE: _sinkhorn_edges is log-space Sinkhorn.
        # - If we feed straight/cross weights (edge_logits in weight space),
        # we must use use_second_sinkhorn_log.
        # - If we feed embedding costs, build log_alpha = -cost and pass directly.
        if self.use_second_sinkhorn or self.use_second_sinkhorn_log:
            if self.edge_plan_from_embeddings:
                # log_alpha = -cost(E1,E2)
                E1_c = E1.float() if E1.dtype in (torch.float16, torch.bfloat16) else E1
                E2_c = E2.float() if E2.dtype in (torch.float16, torch.bfloat16) else E2
                edge_log_alpha = -torch.cdist(E1_c, E2_c, p=self.edge_cost_p).to(dtype)
                edge_transport_plan = self._sinkhorn_edges(edge_log_alpha)

            elif self.use_second_sinkhorn_log:
                edge_transport_plan = self._sinkhorn_edges(torch.log(edge_logits.clamp_min(1e-12)))

            else:
                edge_transport_plan = self._sinkhorn_edges(edge_logits)

        else:
            edge_transport_plan = edge_logits

        # OPTIONAL: mask out dummy edges in the plan
        if self.mask_dummy_edges_in_surrogate:
            v1e = (m1_e != 0).to(dtype)  # [B,K]
            v2e = (m2_e != 0).to(dtype)  # [B,K]
            edge_transport_plan = edge_transport_plan * v1e[:, :, None] * v2e[:, None, :]

        # ------------------------------
        # XOR gate on edges
        # ------------------------------
        if self.xor_on_edge:
            if not (hasattr(Batch, "A1") and hasattr(Batch, "A2")):
                raise RuntimeError(
                    "xor_on_edge=True but Batch has no A1/A2. "
                    "Ensure make_batch_pairs exports upper-tri adjacency vectors A1/A2.",
                )

            A1 = Batch.A1
            A2 = Batch.A2
            if A1.device != device:
                A1 = A1.to(device=device, non_blocking=True)
            if A2.device != device:
                A2 = A2.to(device=device, non_blocking=True)

            alpha_query = (A1 != 0).long()  # [B,K]
            alpha_corpus = (A2 != 0).long()  # [B,K]
            pairwise_xor = (alpha_query[:, :, None] ^ alpha_corpus[:, None, :]).to(
                dtype,
            )  # [B,K,K]

            # OPTIONAL: forbid dummy edges from contributing to XOR weights
            if self.mask_dummy_edges_in_surrogate:
                v1e_b = m1_e != 0
                v2e_b = m2_e != 0
                pairwise_xor = (
                    pairwise_xor * v1e_b.to(dtype)[:, :, None] * v2e_b.to(dtype)[:, None, :]
                )

        else:
            pairwise_xor = 1.0  # scalar broadcasts

        weights_edge = edge_transport_plan * pairwise_xor

        # ------------------------------
        # Edge surrogate (diffalign) only
        # ------------------------------
        if self.use_m_ms_edge:
            diff_e = E1[:, :, None, :] - E2[:, None, :, :]
            del_e = torch.relu(diff_e).sum(dim=-1)
            ins_e = torch.relu(-diff_e).sum(dim=-1)
            edge_align_dist = (
                weights_edge * (self.edge_del_cost * del_e + self.edge_ins_cost * ins_e)
            ).sum(dim=(1, 2))
        else:
            edge_align_dist = asymm_embed_mat_l1_dist(
                E1,
                E2,
                weights_edge,
                ins_cost=self.edge_ins_cost,
                del_cost=self.edge_del_cost,
            )

        return edge_align_dist + self.LAMBDA * node_align_dist


# ============================================================
# Model wrapper: Transformer + (optional enc/head) + PermNet + Surrogate
# ============================================================


class SinkhornQDAGerModel(nn.Module):
    """
    Transformer + PermNet + GED surrogate.
    Output: predicted GED per pair, shape [B,1]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # optional encoders
        if self.cfg.enc.use:
            self.X_encoder = MLP(
                input_dim=self.cfg.enc.in_dim,
                hidden_dims=(self.cfg.enc.num_layers - 1) * [self.cfg.enc.mid_dim],
                output_dim=self.cfg.enc.out_dim,
                dropout_p=self.cfg.enc.enc_dropout,
            )
            self.E_encoder = MLP(
                input_dim=self.cfg.enc.in_dim,
                hidden_dims=(self.cfg.enc.num_layers - 1) * [self.cfg.enc.mid_dim],
                output_dim=self.cfg.enc.out_dim,
                dropout_p=self.cfg.enc.enc_dropout,
            )
            assert (
                self.cfg.enc.out_dim == self.cfg.transformer_layer.in_dim
            ), "when using the encoder, enc.out_dim must equal transformer_layer.in_dim"

        # transformer stack
        layers = [QDAGerLayer(self.cfg)]
        for _ in range(int(self.cfg.transformer_layer.num_layers) - 1):
            layers.append(QDAGerLayer(self.cfg))
        self.transf_layers = nn.Sequential(*layers)

        # optional out heads (cfg.out_head must exist even if disabled)
        if self.cfg.out_head.use:
            self.X_head = MLP(
                input_dim=self.cfg.transformer_layer.out_dim,
                hidden_dims=(self.cfg.out_head.num_layers - 1) * [self.cfg.out_head.mid_dim],
                output_dim=self.cfg.out_head.out_dim,
                dropout_p=self.cfg.out_head.dropout,
            )
            self.E_head = MLP(
                input_dim=self.cfg.transformer_layer.out_dim,
                hidden_dims=(self.cfg.out_head.num_layers - 1) * [self.cfg.out_head.mid_dim],
                output_dim=self.cfg.out_head.out_dim,
                dropout_p=self.cfg.out_head.dropout,
            )

        # PermNet + surrogate
        self.permnet = PermNet(self.cfg)
        self.surrogate = GEDSurrogate(self.cfg)

    def forward(self, Batch):
        # unify dtype with model parameters
        model_dtype = next(self.parameters()).dtype

        Batch.X1 = Batch.X1.to(model_dtype)
        Batch.X2 = Batch.X2.to(model_dtype)
        Batch.E1 = Batch.E1.to(model_dtype)
        Batch.E2 = Batch.E2.to(model_dtype)

        # encoders
        if self.cfg.enc.use:
            Batch.X1 = self.X_encoder(Batch.X1, mask=Batch.m1_x)
            Batch.X2 = self.X_encoder(Batch.X2, mask=Batch.m2_x)
            Batch.E1 = self.E_encoder(Batch.E1, mask=Batch.m1_e)
            Batch.E2 = self.E_encoder(Batch.E2, mask=Batch.m2_e)

        # transformer
        Batch = self.transf_layers(Batch)

        # out heads
        if self.cfg.out_head.use:
            Batch.X1 = self.X_head(Batch.X1, mask=Batch.m1_x)
            Batch.X2 = self.X_head(Batch.X2, mask=Batch.m2_x)
            Batch.E1 = self.E_head(Batch.E1, mask=Batch.m1_e)
            Batch.E2 = self.E_head(Batch.E2, mask=Batch.m2_e)

        # PermNet transport plan (https://arxiv.org/abs/2409.17687 Eq.16)
        P_node = self.permnet(Batch.X1, Batch.X2, Batch.m1_x, Batch.m2_x)

        # surrogate GED
        ged = self.surrogate(Batch, P_node)  # [B]
        return ged.reshape(-1, 1)


class SinkhornMPNNModel(nn.Module):
    """
    MPNN + PermNet + GED surrogate.
    Drop-in replacement for SinkhornQDAGerModel.
    No edge encoder, no edge head — only node message passing.
    Output: predicted GED per pair, shape [B, 1]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.mpnn_layer.in_dim
        out_dim = cfg.mpnn_layer.out_dim
        num_layers = int(cfg.mpnn_layer.num_layers)

        # Optional node encoder only (no edge encoder)
        if self.cfg.enc.use:
            self.X_encoder = MLP(
                input_dim=self.cfg.enc.in_dim,
                hidden_dims=(self.cfg.enc.num_layers - 1) * [self.cfg.enc.mid_dim],
                output_dim=self.cfg.enc.out_dim,
                dropout_p=self.cfg.enc.enc_dropout,
            )
            assert (
                self.cfg.enc.out_dim == in_dim
            ), "when using the encoder, enc.out_dim must equal mpnn_layer.in_dim"

        # ----- MPNN stack -----
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else out_dim
            layers.append(MPNNBaselineLayer(cfg, d_in, out_dim))
        self.mpnn_layers = nn.Sequential(*layers)

        # Optional node output head only (no edge head)
        if self.cfg.out_head.use:
            self.X_head = MLP(
                input_dim=out_dim,
                hidden_dims=(self.cfg.out_head.num_layers - 1) * [self.cfg.out_head.mid_dim],
                output_dim=self.cfg.out_head.out_dim,
                dropout_p=self.cfg.out_head.dropout,
            )

        # PermNet + surrogate (unchanged from SinkhornQDAGerModel)
        self.permnet = PermNet(self.cfg)
        self.surrogate = GEDSurrogate(self.cfg)

    def forward(self, Batch):
        model_dtype = next(self.parameters()).dtype
        Batch.X1 = Batch.X1.to(model_dtype)
        Batch.X2 = Batch.X2.to(model_dtype)

        # 1) Encode node features
        if self.cfg.enc.use:
            Batch.X1 = self.X_encoder(Batch.X1, mask=Batch.m1_x)
            Batch.X2 = self.X_encoder(Batch.X2, mask=Batch.m2_x)

        # 2) MPNN layers (only updates X1/X2)
        Batch = self.mpnn_layers(Batch)

        # 3) Output head on nodes
        if self.cfg.out_head.use:
            Batch.X1 = self.X_head(Batch.X1, mask=Batch.m1_x)
            Batch.X2 = self.X_head(Batch.X2, mask=Batch.m2_x)

        # 4) PermNet transport plan (https://arxiv.org/abs/2409.17687 Eq. 16)
        P_node = self.permnet(Batch.X1, Batch.X2, Batch.m1_x, Batch.m2_x)

        # 5) Surrogate GED
        ged = self.surrogate(Batch, P_node)  # [B]

        return ged.reshape(-1, 1)
