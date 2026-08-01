import math

import torch
import torch.distributed as dist

# -------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------


def compute_loss(model, batch, label_space: str = "raw"):
    """
    label_space:
      - "raw": interval loss in raw label space (LB/UB)
      - "normalized": MSE in z-space using batch.label_mean / batch.label_std
                      (model still outputs pred_raw >= 0)
    """
    pred_raw = model(batch)
    if pred_raw.dim() == 1:
        pred_raw = pred_raw.view(-1, 1)

    dev, dt = pred_raw.device, pred_raw.dtype

    # Get raw targets (LB/UB). In our case LB==UB.
    if hasattr(batch, "lb") and hasattr(batch, "ub"):
        lb = batch.lb
        ub = batch.ub
    else:
        lb = batch.labels[0, :].reshape(-1, 1)
        ub = batch.labels[1, :].reshape(-1, 1)

    lb = lb.to(device=dev, dtype=dt)
    ub = ub.to(device=dev, dtype=dt)

    if label_space == "raw":
        return (torch.relu(lb - pred_raw) ** 2 + torch.relu(pred_raw - ub) ** 2).mean()

    if label_space == "normalized":
        # Require stats (python floats) provided by DataLoader *only if normalize_labels=True*
        if not (hasattr(batch, "label_mean") and hasattr(batch, "label_std")):
            raise RuntimeError(
                "label_space='normalized' but batch has no label_mean/label_std. "
                "Enable data.normalize_labels and ensure DataLoader attaches stats to the batch.",
            )

        mean = torch.tensor(batch.label_mean, device=dev, dtype=dt)
        std = torch.tensor(batch.label_std, device=dev, dtype=dt)

        # Use *raw* label as target (LB).
        y_raw = lb

        pred_z = (pred_raw - mean) / std
        y_z = (y_raw - mean) / std

        return ((y_z - pred_z) ** 2).mean()

    raise ValueError(f"Unknown label_space='{label_space}' (use 'raw' or 'normalized').")


# -------------------------------------------------------------------
# LR schedule
# -------------------------------------------------------------------


def adjust_learning_rate(optimizer, cfg, epoch):
    """
    Cosine LR with linear warmup over epochs.

    base_lr -> warmup -> cosine decay to min_lr across max_epoch.
    """
    base_lr = float(cfg.optim.base_lr)
    min_lr = float(cfg.optim.min_lr)
    max_epoch = int(cfg.optim.max_epoch)
    warmup_epochs = int(cfg.optim.num_warmup_epochs)

    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        # Cosine decay from base_lr to min_lr
        progress = float(epoch - warmup_epochs) / max(
            1.0,
            float(max_epoch - warmup_epochs - 1),
        )
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


# -------------------------------------------------------------------
# Single-process training (no DDP)
# -------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, cfg):
    """
    Plain train loop (single process / single GPU).

    Uses the custom loader interface:
      - loader.num_batches
      - loader.get_batch(i)
    """
    model.train()
    total_loss = 0.0
    num_batches = loader.num_batches

    for i in range(num_batches):
        batch = loader.get_batch(i, device=device)

        optimizer.zero_grad(set_to_none=True)
        label_space = getattr(cfg.data, "loss_space_train", "raw")
        loss = compute_loss(model, batch, label_space=label_space)

        loss.backward()

        if getattr(cfg.optim, "clip_grad_norm", False):
            # This matches the single-GPU code
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=getattr(cfg.optim, "max_norm", 10.0),
            )

        optimizer.step()

        total_loss += loss.detach().item()

    return total_loss / max(1, num_batches)


# -------------------------------------------------------------------
# DDP training — **no manual collectives**
# -------------------------------------------------------------------


def train_one_epoch_DDP(model, loader, optimizer, device, cfg):
    """
    DDP train loop with correct GLOBAL loss reporting (weighted by batch size).
    Training itself is unchanged; only the returned scalar is now the true global mean.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return train_one_epoch(model, loader, optimizer, device, cfg)

    model.train()

    local_loss_sum = 0.0  # sum of (loss_mean * batch_size)
    local_count = 0.0  # sum of batch sizes

    num_batches = loader.num_batches  # per-rank, already sharded

    for i in range(num_batches):
        batch = loader.get_batch(i, device=device)

        optimizer.zero_grad(set_to_none=True)
        label_space = getattr(cfg.data, "loss_space_train", "raw")
        loss = compute_loss(model, batch, label_space=label_space)  # mean over batch

        loss.backward()

        if getattr(cfg.optim, "clip_grad_norm", False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=getattr(cfg.optim, "max_norm", 10.0),
            )

        optimizer.step()

        bsz = (
            float(batch.lb.shape[0])
            if hasattr(batch, "lb")
            else float(loss.shape[0] if loss.dim() else 1.0)
        )
        local_loss_sum += loss.detach().item() * bsz
        local_count += bsz

    # --- all-reduce to get global mean ---
    t_sum = torch.tensor(local_loss_sum, device=device, dtype=torch.float32)
    t_cnt = torch.tensor(local_count, device=device, dtype=torch.float32)

    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

    return (t_sum / t_cnt.clamp_min(1.0)).item()


# -------------------------------------------------------------------
# Evaluation (no DDP collectives)
# -------------------------------------------------------------------


def eval_one_epoch(model, loader, device, label_space: str = "raw"):
    """
    Evaluation loop.

    label_space:
      - "raw": interval loss on [LB,UB]
      - "normalized": MSE on batch.norm_labels

    Returns: mean loss per sample (weights batches by batch size).
    """
    model.eval()

    local_loss_sum = 0.0  # sum(loss_mean * batch_size)
    local_count = 0.0  # sum(batch_size)

    num_batches = loader.num_batches

    with torch.no_grad():
        for i in range(num_batches):
            batch = loader.get_batch(i, device=device)
            loss = compute_loss(model, batch, label_space=label_space)  # mean over batch

            bsz = float(batch.lb.shape[0])  # batch size
            local_loss_sum += loss.detach().item() * bsz
            local_count += bsz

    return local_loss_sum / max(1.0, local_count)
