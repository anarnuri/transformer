import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DistributedSampler, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

# 🔁 STAGE 1 MODEL: mech_type + joints only
from stage_1_model import Stage1LLaMA
from dataset import BarLinkageDataset


# =========================================================
# CoordinateBinner (same as before)
# =========================================================
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def bin_to_value_torch(self, bin_index_tensor: torch.Tensor) -> torch.Tensor:
        bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
        bin_centers_tensor = torch.tensor(
            self.bin_centers,
            device=bin_index_tensor.device,
            dtype=torch.float32,
        )
        return bin_centers_tensor[bin_index_tensor]


# =========================================================
# Loss (same structure as before)
# =========================================================
def cross_entropy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_token: int = 2,
    ignore_index: int = -100,
    label_smoothing: float = 0.10,
) -> torch.Tensor:
    """
    predictions: (B, T, V)
    targets:     (B, T)
    """
    B, T, V = predictions.shape
    mask = targets != pad_token  # (B, T)

    predictions_flat = predictions.view(-1, V)  # (B*T, V)
    targets_flat = targets.view(-1)            # (B*T,)
    targets_flat[~mask.view(-1)] = ignore_index

    loss_unreduced = F.cross_entropy(
        predictions_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )  # (B*T,)

    loss_unreduced = loss_unreduced.view(B, T)  # (B, T)
    return loss_unreduced[mask].mean()


# =========================================================
# DDP utils
# =========================================================
def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# =========================================================
# Stringify config for wandb name
# =========================================================
def stringify_config(cfg: dict) -> str:
    """
    Convert model_config into a short wandb-safe string.
    Example: {"d_model":512, "h":8, "N":6}
    -> "d_model512_h8_N6_..."
    """
    parts = []
    for k, v in cfg.items():
        if isinstance(v, float):
            v = f"{v:.3g}"
        parts.append(f"{k}{v}")
    return "_".join(parts)


# =========================================================
# Checkpoint
# =========================================================
def save_best_checkpoint(
    model,
    optimizer,
    epoch,
    best_loss,
    batch_size,
    lr,
    model_config,
    save_dir="./weights",
):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config["d_model"]
    n_heads = model_config["h"]
    n_layers = model_config["N"]

    checkpoint_path = os.path.join(
        save_dir,
        f"STAGE1_LLAMA_d{d_model}_h{n_heads}_n{n_layers}_bs{batch_size}_lr{lr}_best.pth",
    )
    torch.save(
        {
            "model_state_dict": (
                model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": model_config,
        },
        checkpoint_path,
    )
    print(
        f"[Rank {get_rank()}] ✅ Saved best model at {checkpoint_path} "
        f"(Val Loss: {best_loss:.6f})"
    )


# =========================================================
# Training Loop
# =========================================================
def train(checkpoint_path=None, use_strict_resume: bool = False):
    local_rank = setup_ddp()
    rank = get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Using device {device}")
    torch.set_float32_matmul_precision("medium")

    # ------------------ Config ------------------
    batch_size = 1024
    num_epochs = 1000
    lr = 1e-3

    seq_len = 17
    NUM_BINS = 201
    NUM_MECH_TYPES = 17

    # 🔁 UPDATED SPECIAL TOKENS
    SOS_TOKEN = 0
    EOS_TOKEN = 1
    PAD_TOKEN = 2
    MASK_TOKEN = 3          # NEW dedicated mask token
    NUM_SPECIAL_TOKENS = 4  # SOS, EOS, PAD, MASK
    BIN_OFFSET = NUM_SPECIAL_TOKENS  # bins start at token 4

    # ------------------ Dataset ------------------
    dataset = BarLinkageDataset(
        data_dir="/home/anurizada/Documents/processed_dataset_17"
    )
    vocab_size = NUM_BINS + NUM_SPECIAL_TOKENS

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------ Model ------------------
    model_config = {
        "tgt_seq_len": seq_len,
        "d_model": 512,
        "h": 8,
        "N": 6,
        "num_labels": NUM_MECH_TYPES,
        "vocab_size": vocab_size,
        "dropout": 0.1,
        "pad_token_id": PAD_TOKEN,
        "debug": True,
    }

    model = Stage1LLaMA(
        tgt_seq_len=model_config["tgt_seq_len"],
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        h=model_config["h"],
        N=model_config["N"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
        pad_token_id=model_config["pad_token_id"],
        debug=model_config["debug"],
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # For continuous error metric
    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # ------------------ Optimizer + Cosine Scheduler ------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    warmup_epochs = 5
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    best_loss = float("inf")
    start_epoch = 0

    # ------------------ WandB ------------------
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name="STAGE1_LLAMA_" + stringify_config(model_config),
            config=model_config,
        )

    # ------------------ Resume (optional) ------------------
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=use_strict_resume
        )
        if (not use_strict_resume) and rank == 0:
            print(
                f"[Rank {rank}] 🔁 Non-strict load. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Rank {rank}] 🔁 Resumed from {checkpoint_path} (epoch {start_epoch})")

    # =========================================================
    # Training Epochs
    # =========================================================
    for epoch in range(start_epoch, num_epochs):
        # ------------------ Training ------------------
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_mse = 0.0

        pbar = tqdm(
            total=len(train_loader),
            desc=f"[Rank {rank}] Train {epoch}",
            disable=(rank != 0),
        )

        for batch in train_loader:
            decoder_input = batch["decoder_input_discrete"].to(device)  # (B, T)
            decoder_mask = batch["causal_mask"].to(device).bool()       # (B, T, T) - unused but passed
            mech_labels = batch["encoded_labels"].to(device)            # (B,)
            target_tokens = batch["labels_discrete"].to(device)         # (B, T)

            # =======================================================
            # OPTION A — MASK SOME INPUT TOKENS
            # =======================================================
            p_mask = 0.15   # 15% chance to mask each token
            mask_mask = (torch.rand_like(decoder_input.float()) < p_mask)
            mask_mask &= (decoder_input != PAD_TOKEN)   # never mask PAD
            # (no mech token inside decoder_input, it's separate)

            decoder_input_masked = decoder_input.clone()
            decoder_input_masked[mask_mask] = MASK_TOKEN

            # =======================================================
            # OPTION B — TARGET BIN JITTERING (±5) ON BIN TOKENS
            # =======================================================
            p_perturb = 0.40
            Δ = 5

            target_tokens_jittered = target_tokens.clone()

            jitter_mask = (torch.rand_like(target_tokens.float()) < p_perturb)
            jitter_mask &= (target_tokens != PAD_TOKEN)
            # don't perturb non-bin tokens (specials)
            jitter_mask &= (target_tokens >= BIN_OFFSET)
            jitter_mask &= (target_tokens < BIN_OFFSET + NUM_BINS)
            # optionally, do not jitter where we masked input
            jitter_mask &= ~mask_mask

            noise = torch.randint(
                low=-Δ,
                high=Δ + 1,
                size=target_tokens.shape,
                device=device,
            )

            target_tokens_jittered = torch.where(
                jitter_mask,
                target_tokens_jittered + noise,
                target_tokens_jittered,
            )

            lower = BIN_OFFSET
            upper = BIN_OFFSET + NUM_BINS - 1
            target_tokens_jittered = target_tokens_jittered.clamp(lower, upper)

            optimizer.zero_grad(set_to_none=True)

            # predictions: (B, T, V)
            predictions = model(decoder_input_masked, decoder_mask, mech_labels)

            # CE loss on jittered targets
            ce_loss = cross_entropy_loss(
                predictions, target_tokens_jittered, pad_token=PAD_TOKEN
            )

            # ----- MSE in continuous space (for logging) -----
            pred_tokens = predictions.argmax(dim=-1)         # (B, T)
            pred_bin_rel = pred_tokens - BIN_OFFSET          # relative to first bin
            target_bin_rel = target_tokens - BIN_OFFSET      # ORIGINAL GT, not jittered

            coord_mask = (
                (target_bin_rel >= 0)
                & (target_bin_rel < NUM_BINS)
                & (target_tokens != PAD_TOKEN)
            )

            if coord_mask.any():
                pred_cont = binner.bin_to_value_torch(
                    pred_bin_rel.clamp(0, NUM_BINS - 1)
                )
                target_cont = binner.bin_to_value_torch(
                    target_bin_rel.clamp(0, NUM_BINS - 1)
                )
                mse_loss = F.mse_loss(
                    pred_cont[coord_mask], target_cont[coord_mask]
                )
            else:
                mse_loss = torch.tensor(0.0, device=device)

            total_loss = ce_loss
            total_loss.backward()
            optimizer.step()

            # accuracy over discrete bins (vs ORIGINAL target_tokens)
            valid_mask = target_tokens != PAD_TOKEN
            correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
            acc = correct / (valid_mask.sum().float() + 1e-12)

            epoch_loss += total_loss.item()
            epoch_acc += acc.item()
            epoch_mse += mse_loss.item()

            if rank == 0:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/accuracy": acc.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            pbar.set_postfix({"loss": total_loss.item(), "acc": acc.item()})
            pbar.update(1)

        pbar.close()
        scheduler.step()

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_mse_total = 0.0

        with torch.no_grad():
            pbar = tqdm(
                total=len(val_loader),
                desc=f"[Rank {rank}] Val {epoch}",
                disable=(rank != 0),
            )

            for batch in val_loader:
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device).bool()
                mech_labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                # 🔍 no masking / jittering in validation
                predictions = model(decoder_input, decoder_mask, mech_labels)
                ce_loss = cross_entropy_loss(
                    predictions, target_tokens, pad_token=PAD_TOKEN
                )

                pred_tokens = predictions.argmax(dim=-1)
                pred_bin_rel = pred_tokens - BIN_OFFSET
                target_bin_rel = target_tokens - BIN_OFFSET

                coord_mask = (
                    (target_bin_rel >= 0)
                    & (target_bin_rel < NUM_BINS)
                    & (target_tokens != PAD_TOKEN)
                )

                if coord_mask.any():
                    pred_cont = binner.bin_to_value_torch(
                        pred_bin_rel.clamp(0, NUM_BINS - 1)
                    )
                    target_cont = binner.bin_to_value_torch(
                        target_bin_rel.clamp(0, NUM_BINS - 1)
                    )
                    mse_loss = F.mse_loss(
                        pred_cont[coord_mask], target_cont[coord_mask]
                    )
                else:
                    mse_loss = torch.tensor(0.0, device=device)

                valid_mask = target_tokens != PAD_TOKEN
                correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                acc = correct / (valid_mask.sum().float() + 1e-12)

                val_loss += ce_loss.item()
                val_acc += acc.item()
                val_mse_total += mse_loss.item()

                if rank == 0:
                    wandb.log(
                        {
                            "val/ce_loss": ce_loss.item(),
                            "val/mse_loss": mse_loss.item(),
                            "val/accuracy": acc.item(),
                            "val/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        }
                    )

                pbar.set_postfix({"val_loss": ce_loss.item(), "val_acc": acc.item()})
                pbar.update(1)

            pbar.close()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)
        avg_train_mse = epoch_mse / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_mse = val_mse_total / len(val_loader)

        if rank == 0:
            print(
                f"[Epoch {epoch}] "
                f"TrainLoss={avg_train_loss:.4f} | TrainAcc={avg_train_acc:.4f} | TrainMSE={avg_train_mse:.4f} || "
                f"ValLoss={avg_val_loss:.4f} | ValAcc={avg_val_acc:.4f} | ValMSE={avg_val_mse:.4f}"
            )

            wandb.log(
                {
                    "epoch_summary/train_loss": avg_train_loss,
                    "epoch_summary/train_acc": avg_train_acc,
                    "epoch_summary/train_mse": avg_train_mse,
                    "epoch_summary/val_loss": avg_val_loss,
                    "epoch_summary/val_acc": avg_val_acc,
                    "epoch_summary/val_mse": avg_val_mse,
                    "epoch": epoch,
                }
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    batch_size,
                    lr,
                    model_config,
                )

    cleanup_ddp()


if __name__ == "__main__":
    train(checkpoint_path=None, use_strict_resume=False)
