import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from sym_graphs.data_io.dataset import DataLoader
from sym_graphs.models.simple_model import DirectQDAGerModel, MPNNBaselineModel
from sym_graphs.models.sinkhorn_model import SinkhornMPNNModel, SinkhornQDAGerModel
from sym_graphs.train.train import adjust_learning_rate, eval_one_epoch, train_one_epoch_DDP
from sym_graphs.utils.utils import MaskIgnoringWrapper, _call_seed_everything, yaml_to_dotdict


def _get_loss_space(cfg_data, key: str, default: str = "raw") -> str:
    """
    DotDict-safe getter. cfg_data is a dict-like (DotDict inherits dict).
    """
    v = cfg_data.get(key)
    if v is None:
        return default
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to YAML config file (relative or absolute)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The concerned dataset",
    )

    parser.add_argument(
        "--features",
        type=str,
        choices=["corr_dyn", "hk_pe", "rw_pe"],
        default="corr_dyn",
        required=False,
        help="The features used in the input (default: corr_dyn for quantum features)",
    )

    parser.add_argument(
        "--mpnn",
        action="store_true",
        default=False,
        help="If set, train baseline MPNNs instead of QDAGer",
    )

    args = parser.parse_args()

    # -------------------- resolve config path --------------------
    cfg_path = args.cfg
    dataset_name = args.dataset

    if not os.path.isabs(cfg_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, cfg_path)

    # Logging early (all ranks will print once DDP spawns; that's normal)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info(f"Using config file: {cfg_path}")

    cfg = yaml_to_dotdict(cfg_path)

    root = cfg.data.root
    dataset_type = cfg.data.dataset_type
    batch_size = cfg.data.batch_size
    dyn_features = args.features
    mpnn = args.mpnn
    layer = cfg.transformer_layer if not mpnn else cfg.mpnn_layer

    N_max = layer.N_max

    # -------------------- distributed init / device --------------------
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device(cfg.optim.device)
        rank = 0
        world_size = 1

    try:
        # -------------------- reproducibility (BEFORE loaders/model/opt) --------------------
        _call_seed_everything(cfg, rank=rank)
        torch.set_float32_matmul_precision("high")

        # -------------------- input dimension --------------------
        in_dim = cfg.enc.in_dim if cfg.enc.use else layer.in_dim

        # -------------------- label-space policy (DotDict-safe) --------------------
        # Support BOTH:
        #   cfg.data.loss_space_val (preferred)
        #   cfg.data.loss_space_eval (legacy)
        loss_space_train = _get_loss_space(cfg.data, "loss_space_train", default="raw")
        loss_space_val = _get_loss_space(
            cfg.data,
            "loss_space_val",
            default=_get_loss_space(cfg.data, "loss_space_eval", default="raw"),
        )
        loss_space_test = _get_loss_space(cfg.data, "loss_space_test", default="raw")

        allowed = {"raw", "normalized"}
        if loss_space_train not in allowed:
            raise ValueError(
                f"cfg.data.loss_space_train must be one of {allowed}, got {loss_space_train}",
            )
        if loss_space_val not in allowed:
            raise ValueError(
                f"cfg.data.loss_space_val must be one of {allowed}, got {loss_space_val}",
            )
        if loss_space_test not in allowed:
            raise ValueError(
                f"cfg.data.loss_space_test must be one of {allowed}, got {loss_space_test}",
            )

        need_norm = (
            (loss_space_train == "normalized")
            or (loss_space_val == "normalized")
            or (loss_space_test == "normalized")
        )

        # -------------------- DataLoaders --------------------
        connect_corrs = bool(cfg.data.get("connect_corrs", False))
        need_degree = bool(getattr(cfg.transformer_layer, "degree_scaler", False))

        # Train loader (sharded if DDP)
        train_loader = DataLoader(
            root,
            dataset_name,
            dataset_type,
            "train",
            batch_size,
            dim_in=in_dim,
            N_max=N_max,
            rank=rank,
            world_size=world_size,
            connect_correlator=connect_corrs,
            normalize_labels=need_norm,
            need_degree=need_degree,
            features=dyn_features,
        )

        # Share stats from train to val/test (only meaningful if need_norm=True)
        shared_mean = getattr(train_loader, "label_mean", None)
        shared_std = getattr(train_loader, "label_std", None)

        # Only rank 0 needs val/test loaders (saves time + avoids duplicated warnings)
        if rank == 0:
            val_loader = DataLoader(
                root,
                dataset_name,
                dataset_type,
                "val",
                batch_size,
                dim_in=in_dim,
                N_max=N_max,
                rank=0,
                world_size=1,
                connect_correlator=connect_corrs,
                normalize_labels=need_norm,
                label_mean=shared_mean,
                label_std=shared_std,
                need_degree=need_degree,
                features=dyn_features,
            )
            test_loader = DataLoader(
                root,
                dataset_name,
                dataset_type,
                "test",
                batch_size,
                dim_in=in_dim,
                N_max=N_max,
                rank=0,
                world_size=1,
                connect_correlator=connect_corrs,
                normalize_labels=need_norm,
                label_mean=shared_mean,
                label_std=shared_std,
                need_degree=need_degree,
                features=dyn_features,
            )
        else:
            val_loader = None
            test_loader = None

        # -------------------- Model --------------------
        use_sink = cfg.sinkhorn.get("use", False)
        if use_sink:
            schedule_tau = cfg.sinkhorn.get("schedule_tau", False)
            model = (
                SinkhornMPNNModel(cfg).to(device) if mpnn else SinkhornQDAGerModel(cfg).to(device)
            )
        else:
            schedule_tau = False
            model = (
                MPNNBaselineModel(cfg).to(device) if mpnn else DirectQDAGerModel(cfg).to(device)
            )

        # Only wrap encoders if they do NOT accept `mask=`.
        # head.MLP already supports mask correctly, so do NOT override it.
        import inspect

        def _accepts_mask(mod: nn.Module) -> bool:
            try:
                sig = inspect.signature(mod.forward)
                return "mask" in sig.parameters
            except (TypeError, ValueError):
                # If signature can't be inspected, assume it does NOT accept mask
                return False

        if (
            hasattr(model, "X_encoder")
            and isinstance(model.X_encoder, nn.Module)
            and not _accepts_mask(model.X_encoder)
        ):
            model.X_encoder = MaskIgnoringWrapper(model.X_encoder)

        if (
            hasattr(model, "E_encoder")
            and isinstance(model.E_encoder, nn.Module)
            and not _accepts_mask(model.E_encoder)
        ):
            model.E_encoder = MaskIgnoringWrapper(model.E_encoder)

        # torch.compile (optional)
        if bool(cfg.optim.get("compile", False)):
            model = torch.compile(model, mode="reduce-overhead")

        # Synchronize batch norm through parallel runs : but makes the full run much slower
        # So I leave this "impure" batch norm routine
        # if is_distributed and bool(cfg.transformer_layer.get("batch_norm", False)):
        #    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap with DDP if distributed
        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        # -------------------- Optimizer --------------------
        base_lr = float(cfg.optim.base_lr)
        weight_decay = float(cfg.optim.weight_decay)

        opt_name = cfg.optim.optimizer.lower()
        if opt_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optim.optimizer}")

        # -------------------- Training loop --------------------
        best_val_loss = float("inf")
        best_state_dict = None

        early_stop_patience = int(cfg.optim.get("early_stop_patience", 0))
        epochs_no_improve = 0
        stop_training = False

        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        save_best_every = int(cfg.data.get("save_best_every", 0))
        eval_every = int(cfg.data.get("eval_every", cfg.data.get("save_every", 1)))
        eval_every = max(1, eval_every)

        for epoch in range(int(cfg.optim.max_epoch)):
            t0 = time.time()

            # Sinkhorn tau schedule
            if use_sink and schedule_tau:
                model_to_control = model.module if isinstance(model, DDP) else model
                model_to_control.set_tau_for_epoch(epoch, cfg)

            # LR schedule
            if cfg.optim.scheduler == "cosine_with_warmup":
                lr = adjust_learning_rate(optimizer, cfg, epoch)
            else:
                lr = optimizer.param_groups[0]["lr"]

            # ---------------------- TRAIN ----------------------
            train_loss = train_one_epoch_DDP(model, train_loader, optimizer, device, cfg)

            # ---------------------- EVAL (rank 0 only) ----------------------
            if rank == 0:
                eval_model = model.module if isinstance(model, DDP) else model

                # Grad norm
                grad_sq_sum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq_sum += p.grad.detach().pow(2).sum().item()
                grad_norm = grad_sq_sum**0.5

                # Param norm
                with torch.no_grad():
                    total = 0.0
                    for p in eval_model.parameters():
                        total += p.data.pow(2).sum().item()
                    param_norm = total**0.5

                do_eval = ((epoch + 1) % eval_every == 0) or (epoch == 0)
                val_loss = None
                test_loss = None
                improved = False

                if do_eval:
                    val_loss = eval_one_epoch(
                        eval_model,
                        val_loader,
                        device,
                        label_space=loss_space_val,
                    )
                    test_loss = eval_one_epoch(
                        eval_model,
                        test_loader,
                        device,
                        label_space=loss_space_test,
                    )

                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                        sd = eval_model.state_dict()
                        best_state_dict = {k: v.cpu().clone() for k, v in sd.items()}

                    # Early stopping (eval epochs only)
                    if early_stop_patience > 0:
                        if improved:
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if epochs_no_improve >= early_stop_patience:
                            logging.info(
                                f"Early stopping triggered at epoch {epoch+1} "
                                f"(no val improvement for {early_stop_patience} eval steps).",
                            )
                            stop_training = True

                    # Save best-so-far periodically
                    if (
                        save_best_every > 0
                        and best_state_dict is not None
                        and ((epoch + 1) % save_best_every == 0)
                    ):
                        ckpt_path = os.path.join(
                            models_dir,
                            f"{dataset_name}_best_epoch{epoch+1:04d}_valloss{best_val_loss:.6f}.pt",
                        )
                        torch.save(best_state_dict, ckpt_path)
                        logging.info(f"Saved best checkpoint (so far) to {ckpt_path}")

                dt = time.time() - t0
                if do_eval:
                    logging.info(
                        f"Epoch {epoch+1:4d}/{cfg.optim.max_epoch} "
                        f"| lr={lr:.3e} "
                        f"| param_norm={param_norm:.6f} "
                        f"| grad_norm={grad_norm:.6f} "
                        f"| train_loss={train_loss:.6f} "
                        f"| val_loss={val_loss:.6f} "
                        f"| test_loss={test_loss:.6f} "
                        f"| time={dt:.2f}s",
                    )
                else:
                    logging.info(
                        f"Epoch {epoch+1:4d}/{cfg.optim.max_epoch} "
                        f"| lr={lr:.3e} "
                        f"| param_norm={param_norm:.6f} "
                        f"| grad_norm={grad_norm:.6f} "
                        f"| train_loss={train_loss:.6f} "
                        f"| time={dt:.2f}s",
                    )

            # Broadcast stop flag
            if is_distributed and early_stop_patience > 0:
                stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

        # -------------------- Final test + save (rank 0 only) --------------------
        if rank == 0:
            eval_model = model.module if isinstance(model, DDP) else model

            if best_state_dict is not None:
                eval_model.load_state_dict(best_state_dict)

            final_test_loss = eval_one_epoch(
                eval_model,
                test_loader,
                device,
                label_space=loss_space_test,
            )
            logging.info(f"Final test loss: {final_test_loss:.6f}")
            suffix = (
                ("eq_" if cfg.data.dataset_type == "equal" else "uneq_")
                + ("SH" if cfg.sinkhorn.use else "dir")
                + f"_{dyn_features}"
                + ("_mpnn" if mpnn else "")
            )
            ckpt_path = os.path.join(models_dir, f"{dataset_name}_{suffix}.pt")
            torch.save(eval_model.state_dict(), ckpt_path)
            logging.info(f"Saved model checkpoint to {ckpt_path}")

    finally:
        # Always cleanup NCCL properly
        if is_distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
