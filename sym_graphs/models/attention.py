import math

import torch
import torch.nn as nn

from sym_graphs.utils.data_utils import PairBatch
from sym_graphs.utils.utils import masked_dropout


class MultiHeadAttention(nn.Module):
    """
    Symmetric, permutation-equivariant packed attention (no head mixing inside).

    Inputs via PairBatch:
      E1,E2: [B, P_E, T]  (upper-tri excl. diag; P_E=N_max*(N_max-1)//2, padded)
      X1,X2: [B, N_max, T] (diagonal, padded)
      m*_e : [B, P_E]
      m*_x : [B, N_max]

    Hyperparams:
      T        : number of "time" steps
      P        : P_sym = N_max*(N_max+1)//2 (packed rows)
      num_heads: H
      tau_temp : (kept for API compatibility; not used directly here)
      dropout  : dropout on attention weights

    Outputs:
      probs: [2, H, B, P_sym]
    """

    def __init__(
        self,
        T: int,
        P: int,  # must be P_sym = N_max*(N_max+1)//2
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert T >= 1 and P >= 1 and num_heads >= 1
        self.T = T
        self.P = P
        self.H = num_heads
        self.dropout = float(dropout)

        logit_std = 0.1
        w_std = logit_std / math.sqrt(self.T)
        self.w_sum = nn.Parameter(torch.randn(self.H, self.T) * w_std)
        self.w_diff = nn.Parameter(torch.randn(self.H, self.T) * (0.5 * w_std))  # smaller
        self.b_head = nn.Parameter(torch.zeros(self.H))

        self.eps = 1e-8

    def forward(self, Batch: PairBatch) -> torch.Tensor:
        # Shapes / checks
        _, P_E, T = Batch.E1.shape
        assert T == self.T, f"Temporal length mismatch: got {T}, expected {self.T}"
        N_max = Batch.X1.shape[1]
        P_sym = P_E + N_max
        assert self.P == P_sym, f"P={self.P} must equal P_sym={P_sym} (= {N_max}*(N_max+1)//2)."

        # Use module's dtype/device as reference
        dtype = self.w_sum.dtype
        device = self.w_sum.device

        # buffers / params
        w_sum = self.w_sum  # [H,T]
        w_diff = self.w_diff  # [H,T]
        b_head = self.b_head  # [H]

        # inputs (already harmonized upstream; avoid redundant .to() calls)
        def _td(t: torch.Tensor) -> torch.Tensor:
            if t.device != device or t.dtype != dtype:
                return t.to(device=device, dtype=dtype)
            return t

        def _dev(t: torch.Tensor) -> torch.Tensor:
            if t.device != device:
                return t.to(device=device)
            return t

        E1 = _td(Batch.E1)  # [B,P_E,T]
        X1 = _td(Batch.X1)  # [B,N_max,T]
        E2 = _td(Batch.E2)
        X2 = _td(Batch.X2)

        m1e = _dev(Batch.m1_e)  # [B,P_E]
        m1x = _dev(Batch.m1_x)  # [B,N_max]
        m2e = _dev(Batch.m2_e)
        m2x = _dev(Batch.m2_x)

        # Normalizers #
        N1 = (m1e.sum(1) + m1x.sum(1)).to(dtype)  # [B]
        N2 = (m2e.sum(1) + m2x.sum(1)).to(dtype)  # [B]
        den1 = N1.sqrt()[:, None, None]  # [B,1,1]
        den2 = N2.sqrt()[:, None, None]

        # ---- Normalize to reduce variance ----

        # Packed per-side spatial tensors
        S1 = torch.cat([E1, X1], dim=1) / den1  # [B,P_sym,T]
        S2 = torch.cat([E2, X2], dim=1) / den2

        # Symmetric & antisymmetric combinations (per position)
        S_sum = S1 + S2  # symmetric under 1↔2
        S_diff = S1 - S2  # flips sign under 1↔2

        # Per-head scores per side & per position (shared over positions → perm-equivariant)
        # S_*: [B,P,T], w_*: [H,T]  -> (S @ w.T): [B,P,H] -> permute: [B,H,P]
        base = torch.matmul(S_sum, w_sum.t()).permute(0, 2, 1) + b_head[None, :, None]
        delta = torch.matmul(S_diff, w_diff.t()).permute(0, 2, 1)
        logits1 = base + delta
        logits2 = base - delta

        # stack and permute to [2,H,B,P_sym]
        logits = torch.stack((logits1, logits2), dim=0)  # [2,B,H,P_sym]
        logits = logits.permute(0, 2, 1, 3).contiguous()  # [2,H,B,P_sym]

        # ---- masked softmax (mask BEFORE softmax) ----
        m1_sym = torch.cat([m1e, m1x], dim=1)  # [B,P_sym]
        m2_sym = torch.cat([m2e, m2x], dim=1)  # [B,P_sym]

        mask_valid = torch.stack([m1_sym, m2_sym], dim=0).to(
            dtype=dtype,
            device=device,
        )  # [2,B,P_sym]
        mask_valid = mask_valid.unsqueeze(1)  # [2,1,B,P_sym]
        mask_bool = mask_valid.bool()

        neg_large = torch.finfo(dtype).min
        masked_logits = logits.masked_fill(~mask_bool, neg_large)

        probs = torch.softmax(masked_logits, dim=-1)
        probs = probs * mask_valid  # zero-out padded
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # ---- dropout on attention weights (only on valid entries) ----
        if self.dropout > 0.0 and self.training:
            drop_mask = mask_bool.expand(-1, self.H, -1, -1)  # [2,H,B,P_sym]
            probs = masked_dropout(
                probs,
                p=self.dropout,
                mask=drop_mask,
                training=self.training,
            )

        return probs  # [2,H,B,P_sym]

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        device = getattr(self.w_sum, "device", "cpu")
        dtype = getattr(self.w_sum, "dtype", torch.float32)
        pdrop = self.dropout
        return (
            f"  {cls}(T={self.T}, P={self.P}, heads={self.H}, dim={self.T}, "
            f"  dropout={pdrop}, "
            f"  params : w_sum={tuple(self.w_sum.shape)}, "
            f"  w_diff={tuple(self.w_diff.shape)}, b_head={tuple(self.b_head.shape)}\n"
            f"  dtype={dtype}, device={device}"
        )
