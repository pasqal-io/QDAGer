import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class PairBatch:
    E1: torch.Tensor
    E2: torch.Tensor
    X1: torch.Tensor
    X2: torch.Tensor
    A1: torch.Tensor
    A2: torch.Tensor
    m1_e: torch.Tensor
    m2_e: torch.Tensor
    m1_x: torch.Tensor
    m2_x: torch.Tensor

    def to(self, device, move_adj: bool = False):
        # Move only what the model actually uses by default.
        self.E1 = self.E1.to(device)
        self.E2 = self.E2.to(device)
        self.X1 = self.X1.to(device)
        self.X2 = self.X2.to(device)

        self.m1_e = self.m1_e.to(device)
        self.m2_e = self.m2_e.to(device)
        self.m1_x = self.m1_x.to(device)
        self.m2_x = self.m2_x.to(device)

        # Adjacencies are currently unused in forward; keep on CPU unless explicitly requested.
        if move_adj:
            self.A1 = self.A1.to(device)
            self.A2 = self.A2.to(device)

        return self


def equally_spaced_indices(m: int, N: int, device=None) -> torch.Tensor:
    if m <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if m == 1:
        return torch.tensor([0], dtype=torch.long, device=device)  # or [N//2]

    idx = torch.linspace(0, N - 1, steps=m, device=device)  # float
    return idx.round().long()  # 1D LongTensor of indices


def padded_undirected_adj(data: Data, N_max: int) -> torch.Tensor:
    """
    Cached padded adjacency on CPU. Safe to call repeatedly.
    Stores result on the Data object as _A_pad_{N_max}.
    """
    key = f"_A_pad_{int(N_max)}"
    if hasattr(data, key):
        return getattr(data, key)

    num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
    if num_nodes > N_max:
        raise ValueError(f"num_nodes ({num_nodes}) > N_max ({N_max})")

    row, col = data.edge_index.cpu()
    adj = torch.zeros((N_max, N_max), dtype=torch.float32, device="cpu")
    adj[row, col] = 1.0
    adj[col, row] = 1.0

    setattr(data, key, adj)
    return adj


def padded_undirected_adj_ut(data: Data, N_max: int) -> torch.Tensor:
    """
    Upper-tri adjacency vector (i<j) padded to N_max, cached on the Data object.
    Returns CPU uint8 tensor of shape [K] where K=N_max*(N_max-1)//2.
    """
    key = f"_A_ut_{int(N_max)}"
    if hasattr(data, key):
        return getattr(data, key)

    A = padded_undirected_adj(data, N_max)  # [N_max,N_max] CPU float32
    u, v = get_triu_indices(N_max, device=torch.device("cpu"))
    A_ut = (A[u, v] > 0).to(torch.uint8).contiguous()  # [K] uint8

    setattr(data, key, A_ut)
    return A_ut


def make_batch_pairs(
    dataset,
    pair_indices,
    N_max: int,
    dim_in: int,
    features: str,
    device=None,
    need_degree: bool = False,
):
    """
    Efficient version for DDP:
      - caches *dataset-wide* tensors on CPU pinned memory once per (N_max, dim_in)
      - per batch: index_select on CPU, then move only the batch to GPU (non_blocking)
      - stores adjacency as upper-tri vector A_ut (uint8) rather than full [N,N]
      - Edge features E are padded/scattered in the SAME indexing as get_triu_indices(N_max),
        so that QDAGerLayer's cached u,v (for N_max) matches E and m_e and A_ut.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    want_cuda = (device is not None) and (dev.type == "cuda")

    N_max = int(N_max)
    dim_in = int(dim_in)
    P_E = (N_max * (N_max - 1)) // 2
    G = len(dataset)

    # ---- cache keys: only depend on shape now (CPU cache) ----
    key = (N_max, dim_in)

    if not hasattr(dataset, "_packed_cache"):
        dataset._packed_cache = {}

    # ---- dim_mask cache (CPU) ----
    if not hasattr(dataset, "_dim_mask_cache"):
        dataset._dim_mask_cache = {}
    dm_key = (N_max, dim_in)
    if dm_key not in dataset._dim_mask_cache:
        dim_max = int(getattr(dataset[0], features).shape[1])
        dataset._dim_mask_cache[dm_key] = equally_spaced_indices(dim_in, dim_max, device="cpu")
    dim_mask = dataset._dim_mask_cache[dm_key]

    # ---- build full per-graph tensors ONCE on CPU pinned ----
    if key not in dataset._packed_cache:
        # CPU tensors
        E_all = torch.empty((G, P_E, dim_in), dtype=torch.float32, device="cpu")
        X_all = torch.empty((G, N_max, dim_in), dtype=torch.float32, device="cpu")
        m_e_all = torch.empty((G, P_E), dtype=torch.float32, device="cpu")
        m_x_all = torch.empty((G, N_max), dtype=torch.float32, device="cpu")

        # adjacency upper-tri vector, uint8
        A_ut_all = torch.empty((G, P_E), dtype=torch.uint8, device="cpu")

        # Precompute full N_max upper-tri indices once (CPU)
        u_full, v_full = get_triu_indices(N_max, device=torch.device("cpu"))  # [P_E]

        # Precompute per-graph padded tensors
        for gidx in range(G):
            K_flat = getattr(dataset[gidx], features)[:, dim_mask]
            K_flat = torch.as_tensor(K_flat, dtype=torch.float32, device="cpu")  # [N^2, dim_in]

            N = int(math.isqrt(K_flat.shape[0]))
            C = K_flat.reshape(N, N, dim_in)

            # ----------------------------
            # Nodes: diagonal (contiguous padding is fine)
            # ----------------------------
            diag = torch.arange(N, device=torch.device("cpu"))
            C_x = C[diag, diag, :]  # [N, dim_in]
            X_pad, m_x = pad_to_PT(C_x, N_max)

            # ----------------------------
            # Edges: scatter into N_max upper-tri ordering
            # ----------------------------
            # Mask which N_max-edges are "real" edges among the first N nodes
            edge_mask = (u_full < N) & (v_full < N)  # [P_E] bool

            # Extract the true edges using the SAME ordering as triu_indices(N)
            uN, vN = get_triu_indices(N, device=torch.device("cpu"))
            C_e = C[uN, vN, :]  # [P_in, dim_in] where P_in = N*(N-1)//2

            # Scatter these into the N_max indexing
            E_pad = torch.zeros((P_E, dim_in), dtype=torch.float32, device="cpu")
            E_pad[edge_mask] = C_e

            # Edge mask aligned with E_pad and with QDAGerLayer's u,v
            m_e = edge_mask.to(torch.float32)  # [P_E]

            # Store
            E_all[gidx] = E_pad
            X_all[gidx] = X_pad
            m_e_all[gidx] = m_e
            m_x_all[gidx] = m_x

            # store only upper-tri adjacency vector (already aligned with u_full,v_full)
            A_ut_all[gidx] = padded_undirected_adj_ut(dataset[gidx], N_max)

        # Pin everything for fast H2D transfers
        if torch.cuda.is_available():
            E_all = E_all.pin_memory()
            X_all = X_all.pin_memory()
            m_e_all = m_e_all.pin_memory()
            m_x_all = m_x_all.pin_memory()
            A_ut_all = A_ut_all.pin_memory()

        dataset._packed_cache[key] = (E_all, X_all, A_ut_all, m_e_all, m_x_all)

    E_all, X_all, A_ut_all, m_e_all, m_x_all = dataset._packed_cache[key]

    # ---- parse pair indices ----
    if isinstance(pair_indices, tuple) and len(pair_indices) == 2:
        idx_i, idx_j = pair_indices  # can be pinned already
    else:
        idx_i = torch.tensor([p[0] for p in pair_indices], dtype=torch.long)
        idx_j = torch.tensor([p[1] for p in pair_indices], dtype=torch.long)

    # Keep CPU index tensors (pin helps if present)
    idx_i_cpu = idx_i
    idx_j_cpu = idx_j

    # ---- gather on CPU ----
    E1 = E_all.index_select(0, idx_i_cpu)
    E2 = E_all.index_select(0, idx_j_cpu)
    X1 = X_all.index_select(0, idx_i_cpu)
    X2 = X_all.index_select(0, idx_j_cpu)

    m1_e = m_e_all.index_select(0, idx_i_cpu)
    m2_e = m_e_all.index_select(0, idx_j_cpu)
    m1_x = m_x_all.index_select(0, idx_i_cpu)
    m2_x = m_x_all.index_select(0, idx_j_cpu)

    # Upper-tri adjacency vectors [B, K] (uint8 on CPU)
    A1 = A_ut_all.index_select(0, idx_i_cpu)
    A2 = A_ut_all.index_select(0, idx_j_cpu)

    # ---- move to GPU only if requested ----
    if want_cuda:
        E1 = E1.to(device=dev, non_blocking=True)
        E2 = E2.to(device=dev, non_blocking=True)
        X1 = X1.to(device=dev, non_blocking=True)
        X2 = X2.to(device=dev, non_blocking=True)

        m1_e = m1_e.to(device=dev, non_blocking=True)
        m2_e = m2_e.to(device=dev, non_blocking=True)
        m1_x = m1_x.to(device=dev, non_blocking=True)
        m2_x = m2_x.to(device=dev, non_blocking=True)

        # adjacency vectors for XOR are tiny; move them too
        A1 = A1.to(device=dev, non_blocking=True)
        A2 = A2.to(device=dev, non_blocking=True)

    # Build PairBatch (A1/A2 are upper-tri vectors, not full matrices)
    Batch = PairBatch(E1, E2, X1, X2, A1, A2, m1_e, m2_e, m1_x, m2_x)

    # ---- OPTIONAL degree payload ----
    if need_degree:
        if not hasattr(dataset, "_deg_cache"):
            dataset._deg_cache = {}

        deg_key = int(N_max)
        if deg_key not in dataset._deg_cache:
            deg_all = torch.empty((G, N_max), dtype=torch.float32, device="cpu")
            for gidx in range(G):
                deg_all[gidx] = padded_undirected_degree(dataset[gidx], N_max)
            if torch.cuda.is_available():
                deg_all = deg_all.pin_memory()
            dataset._deg_cache[deg_key] = deg_all
        else:
            deg_all = dataset._deg_cache[deg_key]

        deg1 = deg_all.index_select(0, idx_i_cpu)
        deg2 = deg_all.index_select(0, idx_j_cpu)

        if want_cuda:
            deg1 = deg1.to(device=dev, non_blocking=True)
            deg2 = deg2.to(device=dev, non_blocking=True)

        # keep degrees strictly zero on padded nodes
        deg1 = deg1 * (Batch.m1_x != 0).to(deg1.dtype)
        deg2 = deg2 * (Batch.m2_x != 0).to(deg2.dtype)

        Batch.deg1 = deg1
        Batch.deg2 = deg2

    return Batch


def connected_from_flat(Gflat, *, inplace=False, block_t=None):
    """
    Input:
      Gflat[p, t] with p = i*N + j  (row-major), so G[i,j,:] == Gflat[i*N + j, :]
      shape (N*N, T)

    Output:
      same layout/shape: Gcflat[p, t] = <n_i n_j>(t) - <n_i>(t)<n_j>(t),
      with <n_i>(t) = <n_i n_i>(t).

    Parameters
    ----------
    Gflat : ndarray, shape (N*N, T)
    inplace : bool
        If True, overwrite Gflat.
    block_t : int or None
        If set, process time in blocks (helps memory for huge T).

    Returns
    -------
    Gcflat : ndarray, shape (N*N, T)
    """
    Gflat = np.asarray(Gflat)
    if Gflat.ndim != 2:
        raise ValueError(f"Expected (N*N, T); got {Gflat.shape}")

    M, T = Gflat.shape
    N = int(np.sqrt(M))
    if N * N != M:
        raise ValueError(f"First dim must be a perfect square (N*N). Got {M}.")

    out = Gflat if inplace else Gflat.copy()

    # View as (N, N, T) without copying
    G = out.reshape(N, N, T)

    # m[i,t] = <n_i>(t) = <n_i n_i>(t) = G[i,i,t]
    m = np.diagonal(G, axis1=0, axis2=1).T  # (N, T)

    if block_t is None:
        # Fast vectorized subtraction
        G -= m[:, None, :] * m[None, :, :]
    else:
        block_t = int(block_t)
        if block_t <= 0:
            raise ValueError("block_t must be a positive int or None.")
        for t0 in range(0, T, block_t):
            t1 = min(T, t0 + block_t)
            G[:, :, t0:t1] -= m[:, None, t0:t1] * m[None, :, t0:t1]

    return out


def pad_to_PT(K_flat: torch.Tensor, P_target: int):
    """
    K_flat: [P_in, T_in] where P_in = N^2
    -> K_pad: [P_target, T_target], m: [P_target] (1 for valid rows)
    """
    P_in, T_in = K_flat.shape
    K_pad = F.pad(K_flat, (0, 0, 0, P_target - P_in))  # pad rows
    # Mask (valid rows)
    m = K_flat.new_zeros(P_target)
    m[: min(P_in, P_target)] = 1
    return K_pad, m


def reconstruct_flat_from_components_masked(E, X, m_x, N_max: int):
    """
    Reconstruction consistent with the *scattered* N_max triu ordering:
      - E is stored in the ordering of get_triu_indices(N_max)
      - valid edges are those with both endpoints < N_b
    """
    B, P_E, T = E.shape
    device, dtype = E.device, E.dtype

    K = torch.zeros(B, N_max, N_max, T, device=device, dtype=dtype)

    m_x_f = (m_x != 0).to(torch.int64)  # [B, N_max]
    u_full, v_full = get_triu_indices(N_max, device=device)  # [P_E]

    # Precompute N_max upper-tri indices once
    u_full, v_full = get_triu_indices(N_max, device=device)  # [P_E]

    for b in range(B):
        N_b = int(m_x_f[b].sum().item())
        if N_b == 0:
            continue

        # place diagonal (first N_b entries of X[b])
        idx = torch.arange(N_b, device=device)
        K[b, idx, idx, :] = X[b, :N_b, :]

        # Edges are stored in N_max-triu ordering already (scattered layout).
        # So for visualization we must select exactly those triu positions that
        # lie inside the top-left N_b x N_b block.
        edge_mask_b = (u_full < N_b) & (v_full < N_b)  # [P_E] bool

        # gather and place
        K[b, u_full[edge_mask_b], v_full[edge_mask_b], :] = E[b, edge_mask_b, :]
        K[b, v_full[edge_mask_b], u_full[edge_mask_b], :] = E[b, edge_mask_b, :]

    return K.reshape(B, N_max * N_max, T)


def reconstruct_matrix_from_components_masked(E, X, m_x, N_max: int):
    """
    Wrapper around reconstruct_flat_from_components_masked.
    E   : [B, N_max*(N_max-1)//2, D]
    X   : [B, N_max, D]
    m_x : [B, N_max]
    -> K : [B, N_max, N_max, D]
    """
    K_flat = reconstruct_flat_from_components_masked(E, X, m_x, N_max)  # [B, N*N, D]
    B, L, D = K_flat.shape
    assert L == N_max * N_max
    return K_flat.view(B, N_max, N_max, D)


def compute_score_matrix_from_K(X1, X2, K1, K2, m1_x, lam_pair=1.0):
    """
    X1, X2 : [B, N, D]
    K1, K2 : [B, N, N, D] (node+edge tensors)
    m1_x   : [B, N]       node mask for graph 1
    """
    # node-node similarity
    S_node = torch.matmul(X1, X2.transpose(1, 2))  # [B, N, N]

    # structural similarity
    S_pair = torch.einsum("bikd,bjkd->bij", K1, K2)  # [B, N, N]

    # normalize by number of valid nodes in G1
    deg1 = m1_x.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
    S_pair = S_pair / deg1.unsqueeze(2)

    return S_node + lam_pair * S_pair  # [B, N, N]


_triu_cache = {}


def get_triu_indices(n: int, device: torch.device):
    key = (n, device)
    if key not in _triu_cache:
        _triu_cache[key] = torch.triu_indices(n, n, offset=1, device=device)
    return _triu_cache[key]


def padded_undirected_degree(data: Data, N_max: int) -> torch.Tensor:
    """
    Robust undirected degree (counts each undirected edge once),
    padded to [N_max] on CPU and cached on the Data object.
    """
    key = f"_deg_pad_{int(N_max)}"
    if hasattr(data, key):
        return getattr(data, key)

    num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
    if num_nodes > N_max:
        raise ValueError(f"num_nodes ({num_nodes}) > N_max ({N_max})")

    # edge_index might be directed, undirected, or contain both directions.
    # We canonicalize edges as undirected pairs and unique them to avoid double counting.
    row, col = data.edge_index.cpu()
    a = torch.minimum(row, col)
    b = torch.maximum(row, col)
    pairs = torch.stack([a, b], dim=1)
    pairs = torch.unique(pairs, dim=0)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device="cpu")
    if pairs.numel() > 0:
        deg.index_add_(0, pairs[:, 0], torch.ones(pairs.size(0), dtype=torch.float32))
        deg.index_add_(0, pairs[:, 1], torch.ones(pairs.size(0), dtype=torch.float32))

    deg_pad = torch.zeros(N_max, dtype=torch.float32, device="cpu")
    deg_pad[:num_nodes] = deg

    setattr(data, key, deg_pad)
    return deg_pad


def _to_undirected_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    ei = edge_index.detach().cpu().numpy()
    row, col = ei[0], ei[1]
    data = np.ones(len(row), dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    # symmetrize (keeps it robust even if edge_index is directed)
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def rw_landing_probs(A: sp.csr_matrix, steps: int, eps: float = 1e-8) -> np.ndarray:
    n = A.shape[0]
    steps = int(steps)
    if steps <= 0:
        return np.zeros((n * n, 0), dtype=np.float32)

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    inv_deg = 1.0 / np.maximum(deg, eps)
    D_inv = sp.diags(inv_deg, format="csr")
    P = (D_inv @ A).tocsr()

    feats = []
    Pp = P.copy()
    for _ in range(steps):

        feats.append(Pp.toarray().reshape(-1).astype(np.float32))
        Pp = (Pp @ P).tocsr()
        Pp.eliminate_zeros()

    return np.stack(feats, axis=1)  # (n*n, steps)


def heat_kernel_vecs(
    A: sp.csr_matrix,
    T: int,
    t_min: float = 0,
    t_max: float = 10.0,
    k_eigs: int = 64,
    normalized: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Returns K where K[:, j] = vec(H_{t_j}) and shape(K) = (n*n, T).

    H_t ≈ U diag(exp(-t * λ)) U^T using k_eigs eigenpairs of Laplacian.

    Times t_j are log-spaced between t_min and t_max (common choice).
    """
    n = A.shape[0]

    # --- Laplacian choice ---
    deg = np.asarray(A.sum(axis=1)).reshape(-1)

    if normalized:
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
        D_inv_sqrt = sp.diags(inv_sqrt, format="csr")
        L = sp.eye(n, format="csr") - (D_inv_sqrt @ A @ D_inv_sqrt)  # L_sym
    else:
        D = sp.diags(deg, format="csr")
        L = D - A  # unnormalized

    # --- time grid ---
    # logspace avoids wasting resolution at large t
    ts = np.linspace(t_min, t_max, T).astype(np.float32)

    # --- eigendecomposition once ---
    kk = min(k_eigs, n)
    if n <= 200 or kk == n:
        w, U = np.linalg.eigh(L.toarray())
        idx = np.argsort(w)
        w, U = w[idx], U[:, idx]
        w, U = w[:kk].astype(np.float32), U[:, :kk].astype(np.float32)
    else:
        w, U = spla.eigsh(L, k=kk, which="SM")
        idx = np.argsort(w)
        w, U = w[idx].astype(np.float32), U[:, idx].astype(np.float32)

    # --- build (n*n, T) ---
    out = np.empty((n * n, T), dtype=np.float32)

    # Efficient: for each t, scale columns of U and do matrix product
    for j, t in enumerate(ts):
        weights = np.exp(-t * w).astype(np.float32)  # [kk]
        H = (U * weights) @ U.T  # [n, n]
        out[:, j] = H.reshape(n * n)

    return out  # (n*n, T), float32
