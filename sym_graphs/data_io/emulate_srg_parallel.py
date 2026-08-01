#!/usr/bin/env python
"""
Parallel emulation of SRG graphs + k-body correlation matrices.

Handles multiple datasets, e.g.:
  - SRG(25): 15 graphs, ~128 GB RAM peak each
  - SRG(26): 10 graphs  ~256 GB RAM peak each

Concurrency is limited by MEMORY, not CPU count, and is computed PER DATASET
because the per-task footprint differs with vertex count:
   n_workers = min(user_limit, floor(available_RAM / MEM_PER_TASK)).
Cores are then divided across workers to avoid oversubscription.

Usage:
    python emulate_srg_parallel.py                    # all datasets, auto workers
    python emulate_srg_parallel.py --datasets srg26   # only SRG(26)
    python emulate_srg_parallel.py --workers 2        # cap concurrency
    python emulate_srg_parallel.py --out-dir /path/to/results

Results are written as C_srg<N>_<key>_k2.npy, one per graph. Existing files
are skipped, so the run is resumable after a crash or preemption.
"""

import argparse
import os
import traceback
from dataclasses import dataclass

import networkx as nx
import numpy as np

# ---- dataset registry -----------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# data_io sits TWO levels below the repo root (<repo>/sym_graphs/data_io), so
# climb two dirs to reach <repo>/dataset (matches the original absolute path).
BASE = os.environ.get("SRG_DATASET_DIR", os.path.join(SCRIPT_DIR, "..", "..", "dataset"))


@dataclass
class Dataset:
    name: str  # short id used for --datasets and logging
    path: str  # folder passed to SRG_loader
    n_vertices: int  # vertex count == size key returned by SRG_loader
    mem_per_task_gb: float  # peak RAM per emulation
    workers: int | None = None  # desired concurrent workers (None = auto from RAM)


# mem_per_task_gb doubles per extra qubit: 25 -> 128 GB, 26 -> 256 GB, ...
# workers = how many to run at once on a 512 GB node (512/128=4, 512/256=2).
# NOTE: SRG_loader(path, [n]) reads <path>/SRG_<n>/, so BOTH datasets point at
# the SAME "SRG" folder and are told apart by n_vertices:
#   dataset/SRG/SRG_25/ (15 graphs)   dataset/SRG/SRG_26/ (10 graphs)
DATASETS = [
    Dataset("srg25", os.path.join(BASE, "SRG"), 25, 128, workers=4),
    Dataset("srg26", os.path.join(BASE, "SRG"), 26, 256, workers=2),
]


# ---- global emulation parameters ------------------------------------------
CFG_FILE = os.environ.get(
    "EMU_CFG",
    os.path.join(SCRIPT_DIR, "..", "..", "configs", "emu_cfg", "config_light.yaml"),
)
K = 2
CHUNK_S = 4096
OUT_DIR = "./results_k2"


def process_one(key, adj, out_path, threads):
    """Emulate a single graph and atomically save its k-body correlation matrix."""
    # Keep each worker single-node-friendly: bound its own thread pool.
    import torch

    torch.set_num_threads(threads)

    from sym_graphs.correlators.k_correlators import Correlator
    from sym_graphs.utils.utils import yaml_to_dotdict

    if os.path.exists(out_path):
        return key, out_path, "skipped (exists)"

    cfg = yaml_to_dotdict(CFG_FILE)
    graph = nx.from_numpy_array(adj)

    corr = Correlator(graph, cfg)
    res = corr.emulate_non_UD("SV", True)
    C = corr.k_body_full_chunked(res, K, chunk_s=CHUNK_S).cpu().numpy()

    tmp = out_path + ".tmp.npy"
    np.save(tmp, C)
    os.replace(tmp, out_path)  # atomic: no half-written files on crash
    return key, out_path, "done"


def decide_workers(desired, mem_per_task_gb, force=False):
    """Pick the concurrent-worker count for one dataset.

    `desired` is the requested concurrency (CLI --workers, else the dataset's
    own `workers`, else None to auto-fill from RAM). By default the result is
    capped by a memory-safety estimate so we never oversubscribe RAM. Pass
    force=True to use `desired` verbatim (you accept the OOM risk).
    """
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        avail_gb = float(os.environ.get("AVAIL_RAM_GB", mem_per_task_gb))
    mem_cap = max(1, int(avail_gb // mem_per_task_gb))

    if desired and force:
        return max(1, desired)  # pin exactly, skip the safety cap
    if desired:
        return max(1, min(desired, mem_cap))  # honor request, but stay safe
    return mem_cap  # fully automatic


def run_dataset(ds, out_dir, user_workers, force, n_cores):
    """Load and emulate every graph in one dataset."""
    from sym_graphs.utils.utils import SRG_loader

    srg = SRG_loader(ds.path, [ds.n_vertices])[ds.n_vertices]
    items = list(srg.items())  # [(key, adj), ...]

    # CLI --workers overrides the dataset's built-in `workers`; either is
    # capped for safety unless --force is passed.
    desired = user_workers or ds.workers
    n_workers = decide_workers(desired, ds.mem_per_task_gb, force=force)
    threads = max(1, n_cores // n_workers)

    # Cap BLAS/OMP threads (inherited by spawned workers created below).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)

    print(
        f"\n=== {ds.name}: SRG({ds.n_vertices}) | {len(items)} graphs | "
        f"{n_workers} workers x {threads} threads "
        f"(~{n_workers * ds.mem_per_task_gb:.0f} GB peak, "
        f"{ds.mem_per_task_gb:.0f} GB/task) ===",
    )

    tasks = [
        (k, adj, os.path.join(out_dir, f"C_srg{ds.n_vertices}_{k}_k{K}.npy")) for k, adj in items
    ]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Fresh spawned process per task so the large state vector is reclaimed on exit.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(process_one, k, adj, out, threads): k for (k, adj, out) in tasks}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                key, path, status = fut.result()
                print(f"[{status}] {ds.name} graph {key} -> {path}")
            except Exception:
                print(f"[FAILED] {ds.name} graph {k}\n{traceback.format_exc()}")


def main():
    names = [d.name for d in DATASETS]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=names,
        default=names,
        help="Which datasets to run (default: all).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Override concurrent emulations per dataset (0 = use "
        "each dataset's built-in value / auto from RAM).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Use the requested worker count verbatim, skipping the "
        "memory-safety cap (risk of OOM).",
    )
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n_cores = os.cpu_count() or 1

    for ds in DATASETS:
        if ds.name in args.datasets:
            run_dataset(ds, args.out_dir, args.workers, args.force, n_cores)


if __name__ == "__main__":
    main()
