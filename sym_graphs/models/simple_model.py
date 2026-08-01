import torch.nn as nn
import torch.nn.functional as F

from sym_graphs.models.head import MLP, PooledNodeOnlyHead, build_head
from sym_graphs.models.mpnn_layer import MPNNBaselineLayer
from sym_graphs.models.transformer_layer import QDAGerLayer


class DirectQDAGerModel(nn.Module):
    """
    Our (yet to be named) Transformer model
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.enc.use:
            # Encoder for node features
            self.X_encoder = MLP(
                input_dim=self.cfg.enc.in_dim,
                hidden_dims=(self.cfg.enc.num_layers - 1) * [self.cfg.enc.mid_dim],
                output_dim=self.cfg.enc.out_dim,
                dropout_p=self.cfg.enc.enc_dropout,
            )
            # Encoder for edge features
            self.E_encoder = MLP(
                input_dim=self.cfg.enc.in_dim,
                hidden_dims=(self.cfg.enc.num_layers - 1) * [self.cfg.enc.mid_dim],
                output_dim=self.cfg.enc.out_dim,
                dropout_p=self.cfg.enc.enc_dropout,
            )

            assert self.cfg.enc.out_dim == self.cfg.transformer_layer.in_dim, (
                "when using the encoder, its out dim = "
                f"{self.cfg.enc.out_dim} "
                "must equal the in dim of the following transformer layer = "
                f"{self.cfg.transformer_layer.in_dim}"
            )

        # ----- transformer stack -----
        layers = []
        for _ in range(self.cfg.transformer_layer.num_layers):
            layers.append(QDAGerLayer(self.cfg))
        self.transf_layers = nn.Sequential(*layers)
        self.head = build_head(cfg)
        self.output_positive = bool(getattr(self.cfg.head_layer, "output_positive", True))

    def forward(self, Batch):
        # --- unify dtype with model parameters ---
        model_dtype = next(self.parameters()).dtype

        Batch.X1 = Batch.X1.to(model_dtype)
        Batch.X2 = Batch.X2.to(model_dtype)
        Batch.E1 = Batch.E1.to(model_dtype)
        Batch.E2 = Batch.E2.to(model_dtype)

        # 1) encoder (if config requires it)
        if self.cfg.enc.use:
            Batch.X1 = self.X_encoder(Batch.X1, mask=Batch.m1_x)
            Batch.X2 = self.X_encoder(Batch.X2, mask=Batch.m2_x)
            Batch.E1 = self.E_encoder(Batch.E1, mask=Batch.m1_e)
            Batch.E2 = self.E_encoder(Batch.E2, mask=Batch.m2_e)

        # 2) transformer
        Batch = self.transf_layers(Batch)
        out = self.head(Batch)  # [B,1]

        if self.output_positive:
            return F.softplus(out)
        return out


class MPNNBaselineModel(nn.Module):
    """
    Minimal MPNN baseline model — drop-in replacement for DirectQDAGerModel.
    No edge encoder, no edge features. Only node message passing.
    Uses a node-only head (no edge pooling).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.mpnn_layer.in_dim
        out_dim = cfg.mpnn_layer.out_dim
        num_layers = int(cfg.mpnn_layer.num_layers)

        if self.cfg.enc.use:
            # Encoder for node features only (no edge encoder)
            self.X_encoder = MLP(
                input_dim=self.cfg.enc.in_dim,
                hidden_dims=(self.cfg.enc.num_layers - 1) * [self.cfg.enc.mid_dim],
                output_dim=self.cfg.enc.out_dim,
                dropout_p=self.cfg.enc.enc_dropout,
            )
            assert self.cfg.enc.out_dim == in_dim, (
                f"Encoder out_dim={self.cfg.enc.out_dim} must equal " f"layer in_dim={in_dim}"
            )

        # ----- MPNN stack -----
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else out_dim
            layers.append(MPNNBaselineLayer(cfg, d_in, out_dim))
        self.mpnn_layers = nn.Sequential(*layers)

        # Node-only head — uses mpnn_layer.out_dim, no edge pooling
        self.head = PooledNodeOnlyHead(out_dim, cfg)
        self.output_positive = bool(getattr(self.cfg.head_layer, "output_positive", True))

    def forward(self, Batch):
        model_dtype = next(self.parameters()).dtype
        Batch.X1 = Batch.X1.to(model_dtype)
        Batch.X2 = Batch.X2.to(model_dtype)

        # 1) Encode node features (if config requires it)
        if self.cfg.enc.use:
            Batch.X1 = self.X_encoder(Batch.X1, mask=Batch.m1_x)
            Batch.X2 = self.X_encoder(Batch.X2, mask=Batch.m2_x)

        # 2) MPNN layers
        Batch = self.mpnn_layers(Batch)

        out = self.head(Batch)  # [B, 1]

        if self.output_positive:
            return F.softplus(out)
        return out
