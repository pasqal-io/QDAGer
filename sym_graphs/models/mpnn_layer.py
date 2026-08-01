import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv

from sym_graphs.utils.data_utils import get_triu_indices


class MPNNBaselineLayer(nn.Module):
    """
    Minimal MPNN baseline matching the QDAGerLayer Batch interface.
    No normalization, no FFN, no edge features — only node message passing.

    Args:
        cfg: config object with cfg.mpnn_layer.{N_max, residual, conv_type}
        in_dim: input feature dimension for this layer
        out_dim: output feature dimension for this layer
    """

    def __init__(self, cfg, in_dim: int, out_dim: int):
        super().__init__()
        self.N_max = cfg.mpnn_layer.N_max
        self.out_dim = out_dim
        self.residual = cfg.mpnn_layer.residual

        conv_type = getattr(cfg.mpnn_layer, "conv_type", None) or "GIN"
        self.conv = self._make_conv(conv_type, in_dim, out_dim)

        # Linear projection for residual when dims differ
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

        # Cache triu indices for edge_index reconstruction
        u, v = get_triu_indices(self.N_max, device="cpu")
        self.register_buffer("u", u, persistent=False)
        self.register_buffer("v", v, persistent=False)

    @staticmethod
    def _make_conv(conv_type: str, in_dim: int, out_dim: int):
        conv_type = conv_type.upper()
        if conv_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)
        if conv_type == "GCN":
            return GCNConv(in_dim, out_dim)
        if conv_type == "GAT":
            return GATConv(in_dim, out_dim, heads=1)
        raise ValueError(f"Unknown conv_type: {conv_type}")

    def _triu_to_edge_index(self, mask_e):
        """Fully vectorized: no Python loop over batch."""
        B = mask_e.shape[0]
        N = self.N_max
        u, v = self.u, self.v  # [P_E]

        # Expand to [B, P_E] and filter by mask
        alive = mask_e.bool()  # [B, P_E]

        # Batch offsets: [B, 1] * N added to node indices
        offsets = torch.arange(B, device=mask_e.device).unsqueeze(1) * N  # [B, 1]

        # Broadcast: [B, P_E]
        u_all = u.unsqueeze(0) + offsets  # [B, P_E]
        v_all = v.unsqueeze(0) + offsets  # [B, P_E]

        # Flatten and select alive edges
        u_alive = u_all[alive]  # [total_alive]
        v_alive = v_all[alive]  # [total_alive]

        # Undirected: both directions

        return torch.stack(
            [
                torch.cat([u_alive, v_alive]),
                torch.cat([v_alive, u_alive]),
            ],
            dim=0,
        )  # [2, 2 * total_alive]

    def _run_conv(self, X, m_x, m_e):
        B, N, T = X.shape
        edge_index = self._triu_to_edge_index(m_e)
        x_flat = X.reshape(B * N, T)
        x_flat = self.conv(x_flat, edge_index)
        return x_flat.reshape(B, N, -1) * m_x[:, :, None]

    def forward(self, Batch):
        p = next(self.parameters())
        dtype, device = p.dtype, p.device

        def _td(t):
            return (
                t.to(device=device, dtype=dtype) if (t.device != device or t.dtype != dtype) else t
            )

        def _mask(m):
            m = m.to(device=device) if m.device != device else m
            return m if m.dtype in (torch.bool, dtype) else m.to(dtype=dtype)

        Batch.X1, Batch.X2 = _td(Batch.X1), _td(Batch.X2)
        Batch.m1_x, Batch.m2_x = _mask(Batch.m1_x), _mask(Batch.m2_x)
        Batch.m1_e, Batch.m2_e = _mask(Batch.m1_e), _mask(Batch.m2_e)

        X1_in, X2_in = Batch.X1, Batch.X2

        # ---- Message passing ----
        X1 = self._run_conv(Batch.X1, Batch.m1_x, Batch.m1_e)
        X2 = self._run_conv(Batch.X2, Batch.m2_x, Batch.m2_e)

        # ---- Residual ----
        if self.residual:
            if self.res_proj:
                X1 = X1 + self.res_proj(X1_in) * Batch.m1_x[:, :, None]
                X2 = X2 + self.res_proj(X2_in) * Batch.m2_x[:, :, None]
            else:
                X1 = (X1 + X1_in) * Batch.m1_x[:, :, None]
                X2 = (X2 + X2_in) * Batch.m2_x[:, :, None]

        Batch.X1 = X1
        Batch.X2 = X2

        return Batch
