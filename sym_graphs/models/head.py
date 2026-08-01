import torch
import torch.nn as nn
import torch.nn.functional as F

from sym_graphs.utils.utils import _masked_softmax_1d, masked_count, pooled_tokens


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        bias: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h, bias=bias))
            layers.append(activation())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:    [..., D_in]
        mask: broadcastable to x[..., 0] (e.g. [B,N] for x [B,N,D])
            1/True for valid, 0/False for padded.
        """
        out = self.net(x.float())

        if mask is not None:
            # Ensure mask has same rank as out and broadcast over feature dim
            while mask.dim() < out.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.to(out.dtype)
            return out * mask

        return out


class PooledPairHead(nn.Module):
    """
    Pool X/E into [B, D], then concat (sum/diff), optional size feats, MLP -> [B,1].
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.transformer_layer.out_dim
        self.use_size = bool(cfg.head_layer.use_size_feat)
        self.pooling = cfg.head_layer.pooled.mode

        in_dim = 4 * D + (8 if self.use_size else 0)
        self.mlp = MLP(
            in_dim,
            (cfg.head_layer.num_layers - 1) * [cfg.head_layer.head_dim_mid],
            1,
            dropout_p=cfg.head_layer.head_dropout,
        )

    def forward(self, Batch):
        X1 = pooled_tokens(Batch.X1, self.pooling, mask=Batch.m1_x)
        X2 = pooled_tokens(Batch.X2, self.pooling, mask=Batch.m2_x)
        E1 = pooled_tokens(Batch.E1, self.pooling, mask=Batch.m1_e)
        E2 = pooled_tokens(Batch.E2, self.pooling, mask=Batch.m2_e)

        H = torch.cat((X1 + X2, X1 - X2, E1 + E2, E1 - E2), dim=1)

        if self.use_size:
            n1 = masked_count(Batch.m1_x).to(dtype=H.dtype)
            n2 = masked_count(Batch.m2_x).to(dtype=H.dtype)
            p1 = masked_count(Batch.m1_e).to(dtype=H.dtype)
            p2 = masked_count(Batch.m2_e).to(dtype=H.dtype)

            size_feat = torch.stack([n1, n2, n1 + n2, n1 - n2, p1, p2, p1 + p2, p1 - p2], dim=1)
            H = torch.cat([H, size_feat], dim=1)

        return self.mlp(H)


class TokenScalarHead(nn.Module):
    """
    New head type:
      - X1,X2: [B,N,D] -> lin -> [B,N] -> masked softmax -> [B,N]
      - E1,E2: [B,P,D] -> lin -> [B,P] -> masked softmax -> [B,P]
      - concat (sum/diff) for X and E, + optional size feats, MLP -> [B,1]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.transformer_layer.out_dim
        N = cfg.transformer_layer.N_max
        P = N * (N - 1) // 2

        self.use_size = bool(cfg.head_layer.use_size_feat)
        ts_cfg = getattr(cfg.head_layer, "token_scalar", None)
        self.temp = float(getattr(ts_cfg, "temperature", 1.0)) if ts_cfg is not None else 1.0
        hl = getattr(cfg, "head_layer", None)

        num_layers = getattr(hl, "num_layers", None)
        head_dim_mid = getattr(hl, "head_dim_mid", None)
        head_dropout = getattr(hl, "head_dropout", None)

        # tolerate missing OR explicit None
        num_layers = 2 if (num_layers is None) else int(num_layers)
        head_dim_mid = 16 if (head_dim_mid is None) else int(head_dim_mid)
        head_dropout = 0.0 if (head_dropout is None) else float(head_dropout)

        hidden_dims = [head_dim_mid] * max(0, num_layers - 1)

        self.lin_x = nn.Linear(D, 1, bias=True)
        self.lin_e = nn.Linear(D, 1, bias=True)

        in_dim = (2 * N) + (2 * P) + (8 if self.use_size else 0)
        self.mlp = MLP(
            in_dim,
            hidden_dims,
            1,
            dropout_p=head_dropout,
        )

    def forward(self, Batch):
        # project to scalars
        sx1 = self.lin_x(Batch.X1).squeeze(-1)  # [B,N]
        sx2 = self.lin_x(Batch.X2).squeeze(-1)
        se1 = self.lin_e(Batch.E1).squeeze(-1)  # [B,P]
        se2 = self.lin_e(Batch.E2).squeeze(-1)

        # masked softmax -> distributions over valid tokens
        px1 = _masked_softmax_1d(sx1 / self.temp, Batch.m1_x)
        px2 = _masked_softmax_1d(sx2 / self.temp, Batch.m2_x)
        pe1 = _masked_softmax_1d(se1 / self.temp, Batch.m1_e)
        pe2 = _masked_softmax_1d(se2 / self.temp, Batch.m2_e)

        # build fixed-size embedding (N_max and P_E are fixed by padding)
        H = torch.cat((px1 + px2, px1 - px2, pe1 + pe2, pe1 - pe2), dim=1)

        if self.use_size:
            n1 = masked_count(Batch.m1_x).to(dtype=H.dtype)
            n2 = masked_count(Batch.m2_x).to(dtype=H.dtype)
            p1 = masked_count(Batch.m1_e).to(dtype=H.dtype)
            p2 = masked_count(Batch.m2_e).to(dtype=H.dtype)

            size_feat = torch.stack([n1, n2, n1 + n2, n1 - n2, p1, p2, p1 + p2, p1 - p2], dim=1)
            H = torch.cat([H, size_feat], dim=1)

        return self.mlp(H)


class TokenGatedHead(nn.Module):
    """
    Non-softmax token weighting:
      - compute per-token scalar gates with a linear layer
      - gates = sigmoid(logits / temperature)
      - masked weighted average to get one embedding per graph
      - combine (X1+X2, X1-X2, E1+E2, E1-E2) (+ optional size feat) -> MLP -> scalar
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.transformer_layer.out_dim)

        tg_cfg = getattr(cfg.head_layer, "token_gated", None)
        self.temperature = (
            float(getattr(tg_cfg, "temperature", 1.0)) if tg_cfg is not None else 1.0
        )
        self.eps = float(getattr(tg_cfg, "eps", 1e-8)) if tg_cfg is not None else 1e-8

        self.output_positive = bool(getattr(cfg.head_layer, "output_positive", True))
        self.use_size_feat = bool(getattr(cfg.head_layer, "use_size_feat", False))

        # linear -> scalar gates
        self.lin_x = nn.Linear(D, 1, bias=True)
        self.lin_e = nn.Linear(D, 1, bias=True)

        # MLP input dim
        in_dim = 4 * D
        if self.use_size_feat:
            # same 8 size features used earlier
            in_dim += 8

        num_layers = int(cfg.head_layer.num_layers)
        head_dim_mid = int(cfg.head_layer.head_dim_mid)
        dropout = float(cfg.head_layer.head_dropout)

        self.mlp = MLP(
            input_dim=in_dim,
            hidden_dims=(num_layers - 1) * [head_dim_mid],
            output_dim=1,
            dropout_p=dropout,
        )

    def _masked_gated_pool(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        lin: nn.Linear,
    ) -> torch.Tensor:
        """
        X:    [B, T, D]
        mask: [B, T] (0/1 or bool)
        returns pooled: [B, D]
        """
        # logits: [B,T,1]
        logits = lin(X) / max(self.temperature, self.eps)

        # gates in (0,1): [B,T,1]
        gates = torch.sigmoid(logits)

        # apply mask
        m = mask.to(dtype=X.dtype).unsqueeze(-1)  # [B,T,1]
        gates = gates * m

        # weighted mean (stable when all masked)
        num = (gates * X).sum(dim=1)  # [B,D]
        den = gates.sum(dim=1).clamp_min(self.eps)  # [B,1]
        return num / den

    def forward(self, Batch):
        # pooled node/edge embeddings
        X1 = self._masked_gated_pool(Batch.X1, Batch.m1_x, self.lin_x)  # [B,D]
        X2 = self._masked_gated_pool(Batch.X2, Batch.m2_x, self.lin_x)
        E1 = self._masked_gated_pool(Batch.E1, Batch.m1_e, self.lin_e)
        E2 = self._masked_gated_pool(Batch.E2, Batch.m2_e, self.lin_e)

        H = torch.cat((X1 + X2, X1 - X2, E1 + E2, E1 - E2), dim=1)  # [B,4D]

        if self.use_size_feat:
            n1 = masked_count(Batch.m1_x).to(dtype=H.dtype)
            n2 = masked_count(Batch.m2_x).to(dtype=H.dtype)
            p1 = masked_count(Batch.m1_e).to(dtype=H.dtype)
            p2 = masked_count(Batch.m2_e).to(dtype=H.dtype)

            size_feat = torch.stack([n1, n2, n1 + n2, n1 - n2, p1, p2, p1 + p2, p1 - p2], dim=1)
            H = torch.cat([H, size_feat], dim=1)

        out = self.mlp(H)
        return F.softplus(out) if self.output_positive else out


class PooledNodeOnlyHead(nn.Module):
    """
    Node-only pooled head for MPNN baseline.
    Pools X1/X2 only (no E1/E2), then concat (sum/diff), optional size feats, MLP -> [B,1].
    """

    def __init__(self, D: int, cfg):
        super().__init__()
        self.use_size = bool(cfg.head_layer.use_size_feat)
        self.pooling = cfg.head_layer.pooled.mode

        # 2 * D from (X1+X2, X1-X2), + 8 optional size features
        in_dim = 2 * D + (8 if self.use_size else 0)

        self.mlp = MLP(
            in_dim,
            (cfg.head_layer.num_layers - 1) * [cfg.head_layer.head_dim_mid],
            1,
            dropout_p=cfg.head_layer.head_dropout,
        )

    def forward(self, Batch):
        X1 = pooled_tokens(Batch.X1, self.pooling, mask=Batch.m1_x)
        X2 = pooled_tokens(Batch.X2, self.pooling, mask=Batch.m2_x)

        H = torch.cat((X1 + X2, X1 - X2), dim=1)  # [B, 2*D]

        if self.use_size:
            n1 = masked_count(Batch.m1_x).to(dtype=H.dtype)
            n2 = masked_count(Batch.m2_x).to(dtype=H.dtype)
            p1 = masked_count(Batch.m1_e).to(dtype=H.dtype)
            p2 = masked_count(Batch.m2_e).to(dtype=H.dtype)
            size_feat = torch.stack([n1, n2, n1 + n2, n1 - n2, p1, p2, p1 + p2, p1 - p2], dim=1)
            H = torch.cat([H, size_feat], dim=1)

        return self.mlp(H)


def build_head(cfg):
    ht = getattr(cfg.head_layer, "head_type", "pooled")

    if ht == "token_scalar":
        return TokenScalarHead(cfg)

    if ht == "pooled":
        return PooledPairHead(cfg)  # the pooled head class is called

    if ht == "token_gated":
        return TokenGatedHead(cfg)

    raise ValueError(f"Unknown head_type='{ht}' (expected: pooled | token_scalar | token_gated)")
