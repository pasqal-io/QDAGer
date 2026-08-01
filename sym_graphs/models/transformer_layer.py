import torch
import torch.nn as nn

from sym_graphs.models.attention import MultiHeadAttention
from sym_graphs.utils.data_utils import get_triu_indices
from sym_graphs.utils.utils import (
    masked_batch_norm_scatter,
    masked_dropout,
    masked_layer_norm_scatter,
    signed_sqrt,
)


# ----------------------------- PairUpdater -----------------------------
class PairUpdater(nn.Module):
    """
    Multi-head edge update with learned head mixing written back to Batch.E*.
    Returns per-head edge features (E*_h) for reuse by NodeUpdater.
    """

    def __init__(
        self,
        num_heads: int,
        dim_in: int,
        dim_out: int,
        N_max: int,
        per_head_proj: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        H, T, self.D = num_heads, dim_in, dim_out
        k = (1.0 / max(1, T)) ** 0.5

        self.H = H
        self.N = N_max
        self.per_head_proj = per_head_proj
        self.normalize = normalize

        # Cache upper-triangular indices once (moved with the module)
        u, v = get_triu_indices(N_max, device="cpu")
        self.register_buffer("u", u, persistent=False)
        self.register_buffer("v", v, persistent=False)

        # Per-head parameters (same math as before)
        self.W_X_Q = nn.Parameter(torch.randn(H, T, self.D) * k)  # [H,T,D]
        self.W_X_K = nn.Parameter(torch.randn(H, T, self.D) * k)  # [H,T,D]

        self.W_Ew_W = nn.Parameter(torch.randn(H, T, self.D) * k)  # [H,T,D]
        self.W_Eb_W = nn.Parameter(torch.randn(H, T, self.D) * k)  # [H,T,D]

        if per_head_proj:
            self.head_weights = nn.Parameter(torch.zeros(H, self.D))  # [H,D]
        else:
            self.head_weights = nn.Parameter(torch.zeros(H))  # [H]

    def forward(self, Batch):
        # Shapes
        B, N, T = Batch.X1.shape
        u, v = self.u, self.v  # [P_E], cached

        # Inputs already harmonized in QDAGerLayer
        X1, X2 = Batch.X1, Batch.X2  # [B,N,T]
        E1, E2 = Batch.E1, Batch.E2  # [B,P_E,T]
        m1_e, m2_e = Batch.m1_e, Batch.m2_e  # [B,P_E]

        # Node projections per head → [B,H,N,D]
        # [B,N,T] x [H,T,D] -> [B,H,N,D]
        XK1 = torch.matmul(X1.unsqueeze(1), self.W_X_K.unsqueeze(0))
        XK2 = torch.matmul(X2.unsqueeze(1), self.W_X_K.unsqueeze(0))
        XQ1 = torch.matmul(X1.unsqueeze(1), self.W_X_Q.unsqueeze(0))
        XQ2 = torch.matmul(X2.unsqueeze(1), self.W_X_Q.unsqueeze(0))

        # Average incident node reps for each edge (per head) → [B,H,P_E,D]
        Xsum1 = XQ1[:, :, u, :] + XK1[:, :, v, :]
        Xsum2 = XQ2[:, :, u, :] + XK2[:, :, v, :]

        # Edge projections per head → [B,H,P_E,D]
        # [B,P,T] x [H,T,D] -> [B,H,P,D]
        Ew1 = torch.matmul(E1.unsqueeze(1), self.W_Ew_W.unsqueeze(0))
        Eb1 = torch.matmul(E1.unsqueeze(1), self.W_Eb_W.unsqueeze(0))
        Ew2 = torch.matmul(E2.unsqueeze(1), self.W_Ew_W.unsqueeze(0))
        Eb2 = torch.matmul(E2.unsqueeze(1), self.W_Eb_W.unsqueeze(0))

        E1_h = torch.sigmoid(signed_sqrt(Xsum1) * Ew1 + Eb1)  # [B,H,P_E,D]
        E2_h = torch.sigmoid(signed_sqrt(Xsum2) * Ew2 + Eb2)  # [B,H,P_E,D]

        # Head mixing
        if self.per_head_proj:
            W = (
                torch.softmax(self.head_weights, dim=0) if self.normalize else self.head_weights
            )  # [H,D]
            E1_mix = (E1_h * W[None, :, None, :]).sum(dim=1)  # [B,P,D]
            E2_mix = (E2_h * W[None, :, None, :]).sum(dim=1)
        else:
            w = (
                torch.softmax(self.head_weights, dim=0) if self.normalize else self.head_weights
            )  # [H]
            E1_mix = (E1_h * w[None, :, None, None]).sum(dim=1)  # [B,P,D]
            E2_mix = (E2_h * w[None, :, None, None]).sum(dim=1)

        # ---- STRICT edge masking ----
        edge_mask1 = m1_e[:, None, :, None]  # [B,1,P_E,1]
        edge_mask2 = m2_e[:, None, :, None]
        E1_h = E1_h * edge_mask1
        E2_h = E2_h * edge_mask2

        E1_mix = E1_mix * m1_e[:, :, None]
        E2_mix = E2_mix * m2_e[:, :, None]
        # --------------------------------

        Batch.E1, Batch.E2 = E1_mix, E2_mix
        return E1_h, E2_h, Batch


# ----------------------------- NodeUpdater-----------------------------
class NodeUpdater(nn.Module):
    """
    Updates X using:
      (1) node attention (from K_out_sub), and
      (2) UNWEIGHTED edge→node aggregation from per-head edges E_h (no edge attention on top).
    """

    def __init__(
        self,
        num_heads: int,
        dim_in: int,
        dim_out: int,
        N_max: int,
        per_head_proj: bool = True,
    ):
        super().__init__()
        self.H, self.T_in, self.D_out = num_heads, dim_in, dim_out
        self.N = N_max
        self.P_E = N_max * (N_max - 1) // 2
        self.P_sym = self.P_E + N_max

        kx = (1.0 / max(1, dim_in)) ** 0.5
        self.per_head_proj = per_head_proj

        # Cache u,v once
        u, v = get_triu_indices(N_max, device="cpu")
        self.register_buffer("u", u, persistent=False)
        self.register_buffer("v", v, persistent=False)

        if per_head_proj:
            self.W_V_W = nn.Parameter(
                torch.randn(self.H, self.T_in, self.D_out) * kx,
            )  # [H,T_in,D]
            self.W_V_b = nn.Parameter(torch.zeros(self.H, self.D_out))  # [H,D]
        else:
            self.W_V = nn.Linear(self.T_in, self.D_out)  # [T_in,D]

        kne = (1.0 / max(1, dim_out)) ** 0.5
        self.W_NE_W = nn.Parameter(torch.randn(self.H, self.D_out, self.D_out) * kne)  # [H,D,D]
        self.W_NE_b = nn.Parameter(torch.zeros(self.H, self.D_out))  # [H,D]

        self.W_O = nn.Parameter(torch.zeros(self.H))  # [H]

    def _proj_X(self, X):  # [B,N,T_in] -> [B,H,N,D]
        if self.per_head_proj:
            # [B,N,T] x [H,T,D] -> [B,H,N,D]
            return (
                torch.matmul(X.unsqueeze(1), self.W_V_W.unsqueeze(0))
                + self.W_V_b[None, :, None, :]
            )

        xs = self.W_V(X)  # [B,N,D]
        return xs[:, None, :, :].expand(-1, self.H, -1, -1)

    def _edge_to_node(self, E_h):
        B, H, P_E, D = E_h.shape
        u, v = self.u, self.v  # cached
        M = E_h.new_zeros(B, H, self.N, D)  # [B,H,N,D]
        M.index_add_(2, u, E_h)
        M.index_add_(2, v, E_h)
        return M

    def forward(self, Batch, K_out, E1_h, E2_h):
        assert K_out.dim() == 4 and K_out.shape[0] == 2 and K_out.shape[1] == self.H

        X1, X2 = Batch.X1, Batch.X2  # [B,N,T_in]
        m1_x, m2_x = Batch.m1_x, Batch.m2_x  # [B,N]

        # attention slices
        ATT_X_1 = K_out[0, :, :, self.P_E :]  # [H,B,N]
        ATT_X_2 = K_out[1, :, :, self.P_E :]
        ATT_E_1 = K_out[0, :, :, : self.P_E]  # [H,B,P_E]
        ATT_E_2 = K_out[1, :, :, : self.P_E]

        # Node branch per head
        X1_h = self._proj_X(X1)  # [B,H,N,D]
        X2_h = self._proj_X(X2)

        att_x1 = ATT_X_1.permute(1, 0, 2).unsqueeze(-1)  # [B,H,N,1]
        att_x2 = ATT_X_2.permute(1, 0, 2).unsqueeze(-1)
        X1_h = X1_h * att_x1
        X2_h = X2_h * att_x2

        # ---- STRICT node masking ----
        node_mask1 = m1_x[:, None, :, None]  # [B,1,N,1]
        node_mask2 = m2_x[:, None, :, None]
        X1_h = X1_h * node_mask1
        X2_h = X2_h * node_mask2
        # --------------------------------

        # Weight E_h by attention values
        E1_h = ATT_E_1.permute(1, 0, 2).unsqueeze(-1) * E1_h  # [B,H,P_E,D]
        E2_h = ATT_E_2.permute(1, 0, 2).unsqueeze(-1) * E2_h

        # Edge→node branch, then per-head linear W_NE
        M1_h = self._edge_to_node(E1_h)  # [B,H,N,D]
        M2_h = self._edge_to_node(E2_h)

        # [B,H,N,D] x [H,D,F] -> [B,H,N,F]
        M1_h = torch.matmul(M1_h, self.W_NE_W.unsqueeze(0)) + self.W_NE_b[None, :, None, :]
        M2_h = torch.matmul(M2_h, self.W_NE_W.unsqueeze(0)) + self.W_NE_b[None, :, None, :]

        # ---- STRICT node masking on edge-aggregated message ----
        M1_h = M1_h * node_mask1
        M2_h = M2_h * node_mask2
        # -------------------------------------------------------

        # Combine branches per head (sum) and re-mask once (idempotent but cheap)
        X1_h = (X1_h + M1_h) * node_mask1
        X2_h = (X2_h + M2_h) * node_mask2

        # Learned mixing across heads
        w = torch.softmax(self.W_O, dim=0)  # [H]
        X1_new = (X1_h * w[None, :, None, None]).sum(dim=1)  # [B,N,D]
        X2_new = (X2_h * w[None, :, None, None]).sum(dim=1)

        Batch.X1 = X1_new
        Batch.X2 = X2_new

        return Batch


# ----------------------------- GlobalUpdate ----------------------------
class GlobalUpdate(nn.Module):
    """Runs PairUpdater then NodeUpdater, in that order."""

    def __init__(self, edge_updater: PairUpdater, x_updater: NodeUpdater):
        super().__init__()
        self.edge_updater = edge_updater
        self.x_updater = x_updater

    def forward(self, Batch, K_out):
        E1_h, E2_h, Batch = self.edge_updater(Batch)
        return self.x_updater(Batch, K_out, E1_h, E2_h)


# ----------------------------- QDAGerLayer -----------------------------
class QDAGerLayer(nn.Module):
    """
    Our (yet to be named) Transformer Layer
    """

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.N_max = cfg.transformer_layer.N_max
        self.in_dim = cfg.transformer_layer.in_dim
        self.out_dim = cfg.transformer_layer.out_dim
        self.num_heads = cfg.transformer_layer.num_heads
        self.dropout = cfg.transformer_layer.dropout
        self.residual = cfg.transformer_layer.residual
        self.layer_norm = cfg.transformer_layer.layer_norm
        self.batch_norm = cfg.transformer_layer.batch_norm

        self.attention = MultiHeadAttention(
            T=self.in_dim,
            P=int(self.N_max * (self.N_max + 1) / 2),
            num_heads=self.num_heads,
            dropout=cfg.transformer_layer.attn_dropout,
        )

        # ---- degree scaler ----
        self.use_degree_scaler = bool(getattr(cfg.transformer_layer, "degree_scaler", False))

        if self.use_degree_scaler:
            d = self.out_dim
            self.deg_theta1 = nn.Parameter(torch.ones(d))
            self.deg_theta2 = nn.Parameter(torch.zeros(d))
        # -----------------------------------------

        edge_updater = PairUpdater(
            self.num_heads,
            self.in_dim,
            self.out_dim,
            self.N_max,
            cfg.transformer_layer.edge_updater.per_head_proj,
            normalize=cfg.transformer_layer.edge_updater.normalize,
        )
        x_updater = NodeUpdater(
            self.num_heads,
            self.in_dim,
            self.out_dim,
            self.N_max,
            cfg.transformer_layer.node_updater.per_head_proj,
        )
        self.update_all = GlobalUpdate(edge_updater, x_updater)

        if self.layer_norm:
            self.layer_norm_x = nn.LayerNorm(self.out_dim)
            self.layer_norm_e = (
                nn.LayerNorm(self.out_dim) if cfg.transformer_layer.norm_e else nn.Identity()
            )

        if self.batch_norm:
            self.batch_norm_x = nn.BatchNorm1d(
                self.out_dim,
                track_running_stats=True,
                eps=1e-5,
                momentum=cfg.transformer_layer.bn_momentum,
            )
            self.batch_norm_e = (
                nn.BatchNorm1d(
                    self.out_dim,
                    track_running_stats=True,
                    eps=1e-5,
                    momentum=cfg.transformer_layer.bn_momentum,
                )
                if cfg.transformer_layer.norm_e
                else nn.Identity()
            )

        self.FFN_x_layer1 = nn.Linear(self.out_dim, self.out_dim * 2)
        self.FFN_x_layer2 = nn.Linear(self.out_dim * 2, self.out_dim)

    def forward(self, Batch):

        # ---- single-point dtype/device harmonization for the whole layer (guarded) ----
        anchor = self.FFN_x_layer1.weight
        dtype, device = anchor.dtype, anchor.device

        def _ensure_td(t: torch.Tensor) -> torch.Tensor:
            if t.device != device or t.dtype != dtype:
                return t.to(device=device, dtype=dtype)
            return t

        Batch.X1 = _ensure_td(Batch.X1)
        Batch.X2 = _ensure_td(Batch.X2)
        Batch.E1 = _ensure_td(Batch.E1)
        Batch.E2 = _ensure_td(Batch.E2)

        def _ensure_mask(m: torch.Tensor) -> torch.Tensor:
            if m.device != device:
                m = m.to(device=device)

            if m.dtype == torch.bool:
                return m

            if m.dtype != dtype:
                return m.to(dtype=dtype)

            return m

        Batch.m1_x = _ensure_mask(Batch.m1_x)
        Batch.m2_x = _ensure_mask(Batch.m2_x)
        Batch.m1_e = _ensure_mask(Batch.m1_e)
        Batch.m2_e = _ensure_mask(Batch.m2_e)
        # ---------------------------------------------------------------------------

        X1_in, X2_in = Batch.X1, Batch.X2
        m1_e, m2_e = Batch.m1_e, Batch.m2_e
        m1_x, m2_x = Batch.m1_x, Batch.m2_x

        # main update
        K_out = self.attention(Batch)
        Batch = self.update_all(Batch, K_out)

        X1, X2 = Batch.X1, Batch.X2
        E1, E2 = Batch.E1, Batch.E2

        # ---- Degree scaler (GRIT Eq. 5): after attention/node update, before FFN ----
        if self.use_degree_scaler:
            if not (hasattr(Batch, "deg1") and hasattr(Batch, "deg2")):
                raise RuntimeError(
                    "degree_scaler.use=True but Batch has no deg1/deg2. "
                    "Enable degree payload in make_batch_pairs/DataLoader.",
                )

            deg1 = Batch.deg1.to(device=device, dtype=dtype)
            deg2 = Batch.deg2.to(device=device, dtype=dtype)

            # log(1 + d_i)
            logd1 = torch.log1p(deg1.clamp_min(0)).unsqueeze(-1)  # [B,N,1]
            logd2 = torch.log1p(deg2.clamp_min(0)).unsqueeze(-1)

            theta1 = self.deg_theta1.view(1, 1, -1)  # [1,1,D]
            theta2 = self.deg_theta2.view(1, 1, -1)

            # x' = x ⊙ θ1 + log(1+d) * x ⊙ θ2
            X1 = X1 * theta1 + (X1 * theta2) * logd1
            X2 = X2 * theta1 + (X2 * theta2) * logd2

            # keep padding strictly zero
            X1 = X1 * m1_x[:, :, None]
            X2 = X2 * m2_x[:, :, None]

            Batch.X1, Batch.X2 = X1, X2
        # -------- Norm E --------
        if self.layer_norm:
            E1 = masked_layer_norm_scatter(E1, m1_e, self.layer_norm_e)
            E2 = masked_layer_norm_scatter(E2, m2_e, self.layer_norm_e)

        if self.batch_norm:
            E1 = masked_batch_norm_scatter(E1, m1_e, self.batch_norm_e)
            E2 = masked_batch_norm_scatter(E2, m2_e, self.batch_norm_e)

        # -------- FFN for X --------
        if self.residual:
            X1 = self.FFN_x_layer1(X1 + X1_in).relu() * m1_x[:, :, None]
            X2 = self.FFN_x_layer1(X2 + X2_in).relu() * m2_x[:, :, None]
        else:
            X1 = self.FFN_x_layer1(X1).relu() * m1_x[:, :, None]
            X2 = self.FFN_x_layer1(X2).relu() * m2_x[:, :, None]

        X1 = self.FFN_x_layer2(X1) * m1_x[:, :, None]
        X2 = self.FFN_x_layer2(X2) * m2_x[:, :, None]

        # -------- Norm (X) --------
        if self.layer_norm:
            X1 = masked_layer_norm_scatter(X1, m1_x, self.layer_norm_x)
            X2 = masked_layer_norm_scatter(X2, m2_x, self.layer_norm_x)

        if self.batch_norm:
            X1 = masked_batch_norm_scatter(X1, m1_x, self.batch_norm_x)
            X2 = masked_batch_norm_scatter(X2, m2_x, self.batch_norm_x)

        Batch.X1 = masked_dropout(
            X1,
            p=self.dropout,
            mask=m1_x.unsqueeze(-1).expand(-1, -1, self.out_dim),
            training=self.training,
        )
        Batch.X2 = masked_dropout(
            X2,
            p=self.dropout,
            mask=m2_x.unsqueeze(-1).expand(-1, -1, self.out_dim),
            training=self.training,
        )
        Batch.E1 = masked_dropout(
            E1,
            p=self.dropout,
            mask=m1_e.unsqueeze(-1).expand(-1, -1, self.out_dim),
            training=self.training,
        )
        Batch.E2 = masked_dropout(
            E2,
            p=self.dropout,
            mask=m2_e.unsqueeze(-1).expand(-1, -1, self.out_dim),
            training=self.training,
        )

        return Batch
