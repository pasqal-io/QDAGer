import itertools
import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.data import InMemoryDataset

from sym_graphs.utils.data_utils import (
    PairBatch,
    _to_undirected_csr,
    connected_from_flat,
    heat_kernel_vecs,
    make_batch_pairs,
    rw_landing_probs,
)


def has_json_extension(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".json"


def load_tensor_from_json(path, dtype=torch.float32, device="cpu"):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)  # list of lists
    return torch.tensor(data, dtype=dtype, device=device)


class MyGEDDataset(InMemoryDataset):
    """Generates the dataset compatible with the tasks of this work."""

    def __init__(
        self,
        root,
        name: Literal[
            "aids",
            "linux",
            "mutagenicity",
            "ogbg-code2",
            "ogbg-molhiv",
            "ogbg-molpcba",
            "yeast",
        ],
        mode: Literal["train", "val", "test"],
        dataset_type: Literal["equal", "unequal", "label"],
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ):
        type_mapper = {
            "equal": "no_attr_data",
            "unequal": "no_attr_asymm_data",
            "label": "label_symm_data",
        }
        self.type_mapper = type_mapper
        self.name = name
        self.mode = mode
        self.root = os.path.join(root, type_mapper[dataset_type], name)

        super().__init__(
            self.root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["file_{:s}".format(name.split(".")[0]) for name in os.listdir(self.root)]

    @property
    def processed_file_names(self):
        return [f"data_{self.mode}.pt"]

    def make_labels(self):
        labels_file = os.path.join(self.root, f"{self.mode:s}_result.pt")
        labels_list = torch.tensor(torch.load(labels_file)).transpose(1, 0)
        graphs = torch.load(
            os.path.join(self.root, f"{self.mode:s}_graphs.pt"),
            weights_only=False,
        )
        query_indices, target_indices = list(
            zip(*list(itertools.combinations_with_replacement(range(len(graphs)), 2))),
        )
        query_indices = torch.tensor(query_indices)
        target_indices = torch.tensor(target_indices)

        self.labels = torch.cat(
            (labels_list, query_indices.unsqueeze(0), target_indices.unsqueeze(0)),
            dim=0,
        )
        # order : LB,UB,_,Q,K

    def connect_corrs(self, *, block_t=None):
        """
        Replace data.corr_dyn for every element in the dataset by its connected correlator.
        This overwrites each array in-place (fast + memory-friendly).
        """
        for data in self:
            connected_from_flat(data.corr_dyn, inplace=True, block_t=block_t)
            data.corr_dyn = np.asarray(data.corr_dyn)

    def add_classical(self, dim: int = 20):

        hk_list, rw_list = [], []
        lap_slices = [0]
        rw_slices = [0]

        for i in range(len(self)):
            d = self.get(i)

            n = int(d.num_nodes) if d.num_nodes is not None else int(d.x.size(0))
            A = _to_undirected_csr(d.edge_index, n)

            hk = torch.from_numpy(heat_kernel_vecs(A, dim)).to(torch.float32)  # (n, dim)
            rw = torch.from_numpy(rw_landing_probs(A, dim)).to(
                torch.float32,
            )  # (n*n, dim)

            hk_list.append(hk)
            rw_list.append(rw)

            lap_slices.append(lap_slices[-1] + hk.size(0))  # == n
            rw_slices.append(rw_slices[-1] + rw.size(0))  # == n*n (or whatever is returned)

        self.data.hk_pe = torch.cat(hk_list, dim=0).contiguous()
        self.data.rw_pe = torch.cat(rw_list, dim=0).contiguous()

        self.slices["hk_pe"] = torch.tensor(lap_slices, dtype=torch.long)
        self.slices["rw_pe"] = torch.tensor(rw_slices, dtype=torch.long)

    def process(self):
        data_file = os.path.join(self.root, f"{self.mode:s}.pt")
        data_list = torch.load(data_file, weights_only=False)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self):
        return f"{self.name.capitalize()}-{self.mode}({len(self)})"


class DataLoader:
    """Data loader for batches containing pairs of data points.

    DDP-aware: for mode="train" and world_size>1, it shards the batches
    across ranks so that each rank sees a disjoint subset, with the same
    number of batches on every rank (we drop at most world_size-1 batches
    globally to enforce that).
    """

    def __init__(
        self,
        root,
        name,
        dataset_type,
        mode,
        batch_size,
        dim_in=100,
        shuffle_idx=True,
        normalize_labels=False,
        N_max=20,
        rank: int = 0,
        world_size: int = 1,
        connect_correlator: bool = False,
        need_degree: bool = False,
        label_mean: float | None = None,
        label_std: float | None = None,
        label_eps: float = 1e-8,
        features: Literal["corr_dyn", "hk_pe", "rw_pe"] = "corr_dyn",
    ):

        dataset = MyGEDDataset(
            root=root,
            name=name,
            mode=mode,
            dataset_type=dataset_type,
        )
        dataset.make_labels()
        if connect_correlator:
            dataset.connect_corrs()

        if features in ["hk_pe", "rw_pe"]:
            dataset.add_classical(dim_in)

        # --- optional label normalization (STRICTLY gated by normalize_labels) ---
        self.normalize_labels = bool(normalize_labels)
        self.label_mean = None
        self.label_std = None

        if self.normalize_labels:
            # labels are positive integers -> compute stats in float32
            y = dataset.labels[0, :].to(torch.float32)

            # Use provided stats if given, else compute from this split
            if (label_mean is None) or (label_std is None):
                mean = y.mean()
                std = y.std(unbiased=False).clamp_min(label_eps)
                self.label_mean = float(mean.item())
                self.label_std = float(std.item())
            else:
                self.label_mean = float(label_mean)
                self.label_std = float(max(float(label_std), label_eps))

            # store normalized labels in float32
            self.norm_labels = (y - self.label_mean) / self.label_std
        else:
            # If normalization is off, ignore any externally passed stats
            # (keeps behavior clean and reproducible)
            pass
        # ---------------------------------------------------------------------

        # ------------------------------------------------------------
        # FIX A: DDP-correct shuffle (rank-0 shuffle, broadcast order)
        # ------------------------------------------------------------
        A = dataset.labels[3:, :].detach().cpu().numpy()  # rows: Q,K ; shape [2, num_pairs]
        num_pairs = int(dataset.labels.shape[1])

        if shuffle_idx:
            if (
                mode == "train"
                and world_size > 1
                and dist.is_available()
                and dist.is_initialized()
            ):
                # NCCL cannot broadcast CPU tensors, so broadcast a CUDA tensor.
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "DDP shuffle broadcast requires CUDA when using NCCL backend, "
                        "but torch.cuda.is_available() is False.",
                    )

                if rank == 0:
                    idx = list(range(num_pairs))
                    random.shuffle(idx)
                    perm = torch.tensor(idx, dtype=torch.long, device=torch.device("cuda"))
                else:
                    perm = torch.empty(num_pairs, dtype=torch.long, device=torch.device("cuda"))

                dist.broadcast(perm, src=0)
                idx = perm.cpu().tolist()
            else:
                idx = list(range(num_pairs))
                random.shuffle(idx)
        else:
            idx = list(range(num_pairs))
        # ------------------------------------------------------------

        pair_indices = [(int(A[0, i]), int(A[1, i])) for i in idx]
        L = idx  # these are the actual (possibly shuffled) dataset indices

        self.N_max = N_max
        self.dataset = dataset
        self.dim_in = dim_in
        self.need_degree = bool(need_degree)
        self.features = features

        # Build full (global) batch chunks
        full_idx_chunks = [L[i : i + batch_size] for i in range(0, len(L), batch_size)]
        full_pair_chunks = [
            pair_indices[i : i + batch_size] for i in range(0, len(pair_indices), batch_size)
        ]

        assert len(full_idx_chunks) == len(full_pair_chunks)
        num_full_batches = len(full_idx_chunks)

        # DDP sharding only for training mode
        if mode == "train" and world_size > 1:
            usable_batches = (num_full_batches // world_size) * world_size
            if usable_batches == 0:
                raise RuntimeError(
                    f"Not enough batches ({num_full_batches}) to split across "
                    f"world_size={world_size}. Reduce world_size or batch_size.",
                )

            full_idx_chunks = full_idx_chunks[:usable_batches]
            full_pair_chunks = full_pair_chunks[:usable_batches]

            # Strided split: each rank gets 1/world_size of the batches
            self.idx_chunks = full_idx_chunks[rank::world_size]
            self.pair_chunks = full_pair_chunks[rank::world_size]
        else:
            self.idx_chunks = full_idx_chunks
            self.pair_chunks = full_pair_chunks

        assert len(self.idx_chunks) == len(self.pair_chunks)
        self.num_batches = len(self.idx_chunks)

        # Prebuild per-batch index tensors once (saves per-batch Python work).
        self.i_chunks = [
            torch.tensor([p[0] for p in chunk], dtype=torch.long) for chunk in self.pair_chunks
        ]
        self.j_chunks = [
            torch.tensor([p[1] for p in chunk], dtype=torch.long) for chunk in self.pair_chunks
        ]
        self.idx_chunks_t = [torch.tensor(chunk, dtype=torch.long) for chunk in self.idx_chunks]

        # Keep labels in a pinned, contiguous tensor for fast per-batch gathers
        self.labels_all = dataset.labels.contiguous()
        self.lb_all = self.labels_all[0].contiguous().to(dtype=torch.float32)  # [num_pairs]
        self.ub_all = self.labels_all[1].contiguous().to(dtype=torch.float32)  # [num_pairs]

        # Pin memory to speed CPU->GPU transfer with non_blocking=True
        if torch.cuda.is_available():
            self.i_chunks = [t.pin_memory() for t in self.i_chunks]
            self.j_chunks = [t.pin_memory() for t in self.j_chunks]
            self.idx_chunks_t = [t.pin_memory() for t in self.idx_chunks_t]
            self.labels_all = self.labels_all.pin_memory()
            self.lb_all = self.lb_all.pin_memory()
            self.ub_all = self.ub_all.pin_memory()

    def get_batch(self, i: int, device=None) -> PairBatch:

        Batch = make_batch_pairs(
            self.dataset,
            (self.i_chunks[i], self.j_chunks[i]),
            self.N_max,
            self.dim_in,
            self.features,
            device=device,
            need_degree=self.need_degree,
        )

        idx_t = self.idx_chunks_t[i]  # [B] long (pinned if CUDA available)

        # Full labels (CPU pinned) for debugging/alternate losses
        Batch.labels = self.labels_all.index_select(1, idx_t)  # [5, B] on CPU (pinned)

        # Fast LB/UB fetch (CPU pinned -> GPU non_blocking)
        if device is not None:
            dev = torch.device(device)
            Batch.lb = (
                self.lb_all.index_select(0, idx_t)
                .view(-1, 1)
                .to(
                    device=dev,
                    non_blocking=True,
                )
            )
            Batch.ub = (
                self.ub_all.index_select(0, idx_t)
                .view(-1, 1)
                .to(
                    device=dev,
                    non_blocking=True,
                )
            )
        else:
            Batch.lb = self.lb_all.index_select(0, idx_t).view(-1, 1)
            Batch.ub = self.ub_all.index_select(0, idx_t).view(-1, 1)

        # Normalized labels (only if enabled)
        if self.normalize_labels:
            nl_cpu = self.norm_labels.index_select(0, idx_t)  # [B] float32 on CPU
            Batch.norm_labels = (
                nl_cpu.to(device=device, non_blocking=True) if device is not None else nl_cpu
            )
            Batch.label_mean = self.label_mean  # python float
            Batch.label_std = self.label_std  # python float

        return Batch
