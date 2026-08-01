import math
import os
import random
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pulser.backend import Results
from torch import nn


class DotDict(dict):
    """A dictionary that supports dot notation as attribute access."""

    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return DotDict(value)  # Recursively convert nested dicts
        return value

    def __setattr__(self, name, value):
        self[name] = value


class MaskIgnoringWrapper(nn.Module):
    """
    Wrap a module so it accepts a `mask=` keyword argument but ignores it.
    This lets us keep model source unchanged while satisfying calls like:
        self.X_encoder(x, mask=...)
    """

    def __init__(self, base_module: nn.Module):
        super().__init__()
        self.base_module = base_module

    def forward(self, *args, **kwargs):
        kwargs.pop("mask", None)
        return self.base_module(*args, **kwargs)


def bitstrings_to_array(bits: list[str]) -> np.ndarray:
    """transforms the computational basis into a np binary array
    Args:
        bits (list[str]): list of bitstrings in the computational basis

    Returns:
        np.ndarray: 2D binary np array
    """
    if not bits:
        return np.empty((0, 0), dtype=np.uint8)
    w = len(bits[0])
    joined = "".join(bits).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8) - 48  # '0' -> 0, '1' -> 1
    return arr.reshape(len(bits), w).astype(np.uint8)


def permuted_graph(G: nx.Graph, mapping: dict) -> nx.Graph:
    """Permutes a graph provided a mapping

    Args:
        G (nx.Graph): networkx graph
        mapping (dict):  isomorphism map

    Returns:
        nx.Graph:  output graph
    """
    edge_list = []
    for u, v in G.edges():
        edge = (mapping[u], mapping[v])
        edge_list.append((min(edge), max(edge)))
    H = nx.Graph()
    H.add_edges_from(edge_list)
    return H


def hamming_weight(bitstring: str) -> int:
    """computes the Hamming weight of a given bitstring

    Args:
        bitstring (str): a bitstring

    Returns:
        int: its hamming weight
    """
    return bitstring.count("1")


def get_index_groups(labels):
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)
    return list(groups.values())


def get_subcorr(res: Results, k: int) -> torch.Tensor:
    """Computes the correlation matrix dynamics on a definite subspace of
    bitstrings of hamming weight k

    Args:
        res (Results): the result object
        k (int): hamming weight of the bitstrings

    Returns:
        torch.Tensor: subspace correlation matrix
    """

    final_state = res._results["state"][list(res._results["state"].keys())[-1]]
    subspace_idx = [
        i
        for i in range(final_state.vector.shape[0])
        if hamming_weight(final_state._index_to_bitstring(i)) == k
    ]
    subspace_dyn = torch.stack(
        [
            torch.abs(res._results["state"][t].vector[subspace_idx]) ** 2
            for t in res._results["correlation_matrix"]
        ],
    ).T
    bitstrings = [final_state._index_to_bitstring(i) for i in subspace_idx]
    B = np.array([[int(b) for b in s] for s in bitstrings], dtype=np.float64)

    N = len(final_state._index_to_bitstring(0))
    T = list(res._results["state"].keys())
    sub_corr = torch.zeros((len(T), N, N))
    for it, _ in enumerate(T):
        P = subspace_dyn[:, it].cpu().numpy()
        C = B[:, :, None] * B[:, None, :]
        C = np.transpose(C, (1, 2, 0))
        C_weighted = C * P[None, None, :]
        sub_corr[it] += np.sum(C_weighted, axis=2)
    return sub_corr


def SRG_loader(root: str, sizes: list[int]) -> dict:
    """reads the SRGs stored in files and returns their adjacency matrices

    Args:
        root (str): the root of the datasets

    Returns:
        dict: dict where keys allow to identifiate the SRG.
    """
    Adj_dict = {}
    for s in sizes:
        Adj_dict[s] = {}
        Root = os.path.join(root, f"SRG_{s:d}")
        for i, file in enumerate(os.listdir(Root)):
            path = os.path.join(Root, file)
            with open(path, "rb") as f:  # read as binary
                raw_lines = f.read().splitlines()

            adj_list = []
            for raw in raw_lines:
                # decode safely, strip junk, keep only 0/1
                line = raw.decode("ascii", "ignore").strip()
                clean = "".join(ch for ch in line if ch in "01")
                if clean:
                    adj_list.append(clean)

            # validate square shape
            nrows = len(adj_list)
            if any(len(row) != nrows for row in adj_list):
                print(
                    f"⚠️ Skipping malformed file {path} "
                    f"(rows={nrows}, lengths={[len(r) for r in adj_list]})",
                )
                continue

            adj = np.array([list(map(int, row)) for row in adj_list], dtype=np.int8)
            Adj_dict[s][i] = adj
    return Adj_dict


def yaml_to_dotdict(file_path: str) -> DotDict:
    """Takes a config file and returns a DotDict obj

    Args:
        file_path (str): yaml file path

    Returns:
        DotDict: Dict that we can easily querry
    """
    with open(file_path) as file:
        data = yaml.safe_load(file)
        return DotDict(data)


def group_by_dyn(C, round_level):
    vals = set(C.sum(0).round(round_level).ravel().tolist())
    tuple_list = {}
    for i, v in enumerate(vals):
        tuple_list[i] = []
        idxs = np.where(C.sum(0).round(round_level) == v)
        idxs = [e.tolist() for e in idxs]
        N = len(idxs[0])
        K = len(idxs)
        for j in range(N):
            tuple_list[i].append([idxs[k][j] for k in range(K)])
    return tuple_list


def factors(n: int) -> tuple[int, int]:
    """
    Retourne deux entiers (a, b) tels que a*b = n et que |a - b| soit minimal.
    Si n est premier, retourne (n, 1).
    Précondition: n >= 1.
    """
    if n < 1:
        raise ValueError("n doit être un entier positif (>= 1).")

    r = int(math.isqrt(n))
    for d in range(r, 0, -1):
        if n % d == 0:
            return (n // d, d)

    return (n, 1)


def all_no_repeats(lists):
    return all(len(lst) == len(set(lst)) for lst in lists)


def compute_off_diag(C: np.array, approx_lvl: int) -> np.array:
    """Reduces the dynamics of a certain correlator to those of its
    equivalent sub-components

    Args:
        C (np.array): input tensor
        approx_lvl (int): the decimal after which to consider dynamics as equivalent

    Returns:
        np.array: result as a squeezed array
    """
    groups = group_by_dyn(C, approx_lvl)
    topop = [k for k, l in groups.items() if not all_no_repeats(l)]
    for e in topop:
        groups.pop(e, None)
    off_diag_dict = {}
    for i, remain in enumerate(list(groups.values())):
        off_diag_dict[i] = np.array([C[:, i, j] for i, j in np.array(remain)]).transpose().mean(1)
    return np.array(list(off_diag_dict.values()))


def signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """The signed square-root activation

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """
    relu = torch.nn.functional.relu
    return relu(x).sqrt() - relu(-x).sqrt()


def sample_gumbel_like(x, eps=1e-20):
    U = torch.rand_like(x).clamp_min(eps).clamp_max(1.0 - eps)
    return -torch.log(-torch.log(U))


def gumbel_sinkhorn(Batch, noise_factor=0.1, n_iters=20, tau=0.1):
    X1, X2 = Batch.X1, Batch.X2  # [B, N, D]
    m1 = Batch.m1_x > 0  # [B, N]  True = real
    m2 = Batch.m2_x > 0  # [B, N]
    N = X1.shape[1]
    # mask[b, i, j] = True iff row i and col j are both real
    mask = m1[:, :, None] & m2[:, None, :]  # [B, N, N]

    # base scores (≤ 0), add Gumbel noise, scale by tau
    scores = -torch.cdist(X1, X2, p=1)  # [B, N, N]
    noise = sample_gumbel_like(scores) * noise_factor
    scores = (scores + noise) / tau

    for _ in range(n_iters):
        scores = scores - (torch.logsumexp(scores, dim=2, keepdim=True)).view(-1, N, 1)
        scores = scores - (torch.logsumexp(scores, dim=1, keepdim=True)).view(-1, 1, N)

    return torch.exp(scores) * mask


def masked_batch_norm_scatter(
    X: torch.Tensor,
    mask: torch.Tensor,
    bn: nn.BatchNorm1d,
    zero_masked: bool = True,
) -> torch.Tensor:
    """
    Masked BatchNorm over valid (mask==1) positions only, then scatter back.

    X:    (B, T, C)
    mask: (B, T) 0/1 or bool
    bn:   nn.BatchNorm1d(C)

    Returns:
        X_bn: (B, T, C) normalized on valid entries only.
    """
    B, T, C = X.shape
    mask_flat = mask.bool().reshape(-1)  # (B*T,)
    X_flat = X.reshape(-1, C)  # (B*T, C)

    n_valid = int(mask_flat.sum())
    if n_valid == 0:
        return X.new_zeros(X.shape) if zero_masked else X.clone()

    valid = X_flat[mask_flat]  # (Nvalid, C)

    # ---- dtype/device harmonization (local cast only) ----
    if hasattr(bn, "weight") and bn.weight is not None:
        anchor = bn.weight
    elif hasattr(bn, "running_mean") and bn.running_mean is not None:
        anchor = bn.running_mean
    else:
        anchor = X

    valid_cast = valid.to(dtype=anchor.dtype, device=anchor.device)
    valid_bn = bn(valid_cast).to(dtype=X.dtype, device=X.device)
    # -----------------------------------------------------

    out_flat = X_flat.clone()
    out_flat[mask_flat] = valid_bn
    if zero_masked:
        out_flat[~mask_flat] = 0

    return out_flat.reshape(B, T, C)


def masked_layer_norm_scatter(
    X: torch.Tensor,
    mask: torch.Tensor,
    ln: nn.LayerNorm,
    zero_masked: bool = True,
) -> torch.Tensor:
    """
    Apply LayerNorm only on valid (mask==1) positions, then scatter back.

    X:    (B, T, C)
    mask: (B, T) 0/1 or bool
    ln:   nn.LayerNorm(C)

    Returns:
        X_ln: (B, T, C) layer-normalized on valid entries only.
    """
    B, T, C = X.shape
    mask_flat = mask.bool().reshape(-1)  # (B*T,)
    X_flat = X.reshape(-1, C)  # (B*T, C)

    n_valid = int(mask_flat.sum())
    if n_valid == 0:
        return X.new_zeros(X.shape) if zero_masked else X.clone()

    valid = X_flat[mask_flat]  # (Nvalid, C)

    # ---- dtype/device harmonization (local cast only) ----
    if hasattr(ln, "weight") and ln.weight is not None:
        anchor = ln.weight
    else:
        anchor = X

    valid_cast = valid.to(dtype=anchor.dtype, device=anchor.device)
    valid_ln = ln(valid_cast).to(dtype=X.dtype, device=X.device)
    # -----------------------------------------------------

    out_flat = X_flat.clone()
    out_flat[mask_flat] = valid_ln
    if zero_masked:
        out_flat[~mask_flat] = 0

    return out_flat.reshape(B, T, C)


def masked_dropout(x, p: float, mask: torch.Tensor, training: bool = True):
    """
    Dropout that is only applied where mask == True (valid positions).

    Args:
        x:      input tensor
        p:      dropout probability (0 <= p < 1)
        mask:   boolean or 0/1 tensor, broadcastable to x.
                True  -> dropout is applied here
                False -> keep as-is (no dropout, no scaling)
        training: if False, returns x unchanged (like nn.Dropout)

    Returns:
        Tensor with dropout applied only on masked positions.
    """
    if not training or p == 0.0:
        return x

    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability has to be in [0, 1), got {p}")

    mask_bool = mask.to(dtype=torch.bool)

    # sample where dropout is allowed
    drop = (torch.rand_like(x) < p) & mask_bool
    keep = ~drop

    # inverted dropout scaling only on masked (valid) positions
    scale = torch.ones_like(x, dtype=x.dtype)
    scale[mask_bool] = scale[mask_bool] / (1.0 - p)

    return x * keep.to(x.dtype) * scale


def asymm_embed_mat_l1_dist(X, Y, T, ins_cost: float, del_cost: float):
    """
    Generic asymmetric L1 distance:
      sum_{i,j} T_{ij} [ del_cost * relu(X_i - Y_j) + ins_cost * relu(Y_j - X_i) ]
    where X: [B, M, D], Y: [B, N, D], T: [B, M, N].
    """
    diff = X[:, :, None, :] - Y[:, None, :, :]  # [B, M, N, D]
    del_term = F.relu(diff).sum(dim=-1)  # [B, M, N]
    ins_term = F.relu(-diff).sum(dim=-1)  # [B, M, N]
    return (T * (del_cost * del_term + ins_cost * ins_term)).sum(dim=(1, 2))


def masked_count(mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    mask: [B, N] bool or {0,1}
    returns: [B] counts (clamped)
    """
    cnt = mask.to(torch.float32).sum(dim=1)
    return cnt.clamp_min(eps)


def pooled_tokens(
    x: torch.Tensor,
    mode: str,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    x:    [B, N, D]
    mask: [B, N] with 1/True for valid, 0/False for padded.

    Modes:
      - "mean"/"sum"/"max": EXACTLY the behavior (ignores mask) for benchmarking.
      - "norm_mean": masked mean, divides by valid count.
      - "sqrt_sum": masked sum / sqrt(count) (variance-stabilized aggregate).
    """
    if mode == "mean":
        return x.mean(dim=1)  # benchmark mode: ignores mask
    if mode == "sum":
        return x.sum(dim=1)  # benchmark mode: ignores mask
    if mode == "max":
        return x.max(dim=1).values  # benchmark mode: ignores mask

    if mask is None:
        raise ValueError(f"Pooling mode '{mode}' requires a mask, but mask is None.")

    if mode == "norm_mean":
        m = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B,N,1]
        cnt = masked_count(mask, eps=eps).to(dtype=x.dtype).unsqueeze(-1)  # [B,1]
        return (x * m).sum(dim=1) / cnt

    if mode == "sqrt_sum":
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        cnt = masked_count(mask, eps=eps).to(dtype=x.dtype).unsqueeze(-1)
        return (x * m).sum(dim=1) / cnt.sqrt()

    raise ValueError(f"Pooling mode '{mode}' not implemented.")


def _round_nearest(x: Any, clock_period: int = 4) -> Any:
    """
    To be used for rounding durations to the nearest clock period.

    Args:
        x: duration in ns
        clock period: clock period of device channel in ns

    Returns:
        duration rounded to nearest multiple of period
    """
    return round(x / clock_period) * clock_period


def seed_everything(seed: int, *, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Strong determinism (can be slower, may error if an op has no deterministic kernel)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # Still more stable than default without being strict
        torch.backends.cudnn.benchmark = False


def _call_seed_everything(cfg, rank: int):
    """
    Calls existing seed_everything() in a way that stays compatible
    regardless of its current signature.
    Preference order:
      1) seed_everything(seed, deterministic=...)
      2) seed_everything(seed)
      3) seed_everything()
    """
    seed = int(getattr(cfg, "seed", getattr(getattr(cfg, "optim", {}), "seed", 42)))
    deterministic = bool(getattr(cfg, "deterministic", False))

    # In DDP, per-rank seed offsets avoid identical RNG streams across ranks.
    seed_rank = seed + int(rank)

    try:
        seed_everything(seed_rank, deterministic=deterministic)
        return
    except TypeError:
        pass

    try:
        seed_everything(seed_rank)
        return
    except TypeError:
        pass

    seed_everything()


def _masked_softmax_1d(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    logits: [B, L]
    mask:   [B, L] bool or {0,1}
    returns probs [B, L] with probs.sum(-1)=1 over valid entries, 0 on padded.
    """
    if mask.dtype != torch.bool:
        mask_bool = mask != 0
    else:
        mask_bool = mask

    neg_large = -1e9 if logits.dtype not in (torch.float16, torch.bfloat16) else -1e4
    masked_logits = logits.masked_fill(~mask_bool, neg_large)

    probs = torch.softmax(masked_logits, dim=-1)
    probs = probs * mask_bool.to(dtype=probs.dtype)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def group_by_epsilon(A: np.ndarray, eps: float) -> list[list[tuple[int, ...]]]:
    """
    Cluster the multi-indices of an N-dimensional array into groups
    where array values differ by less than a given tolerance `eps`.

    Parameters
    ----------
    A : np.ndarray
        Input array of shape (N, ..., N), i.e., k-dimensional with all
        sides of length N.
    eps : float
        Tolerance for grouping values. Two indices belong to the same
        cluster if their corresponding values differ from the cluster
        reference value by less than eps.

    Returns
    -------
    clusters : List[List[Tuple[int, ...]]]
        A list of clusters, where each cluster is a list of index tuples
        (i1, ..., ik). Each tuple corresponds to an element of `A`.
    """
    # Flatten values with their multi-indices
    coords = np.array(np.unravel_index(np.arange(A.size), A.shape)).T
    values = A.ravel()

    # Sort by values for efficient grouping
    order = np.argsort(values)
    coords, values = coords[order], values[order]

    clusters: list[list[tuple[int, ...]]] = []
    current: list[tuple[int, ...]] = [tuple(coords[0])]
    ref = values[0]

    for c, v in zip(coords[1:], values[1:]):
        if abs(v - ref) < eps:
            current.append(tuple(c))
        else:
            clusters.append(current)
            current = [tuple(c)]
            ref = v
    clusters.append(current)
    return clusters
