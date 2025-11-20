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

# 🔁 latent model
from vit_llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset


# =========================================================
# Latent normalization (unchanged)
# =========================================================
class LatentTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x = (x - x_min) / (x_max - x_min + 1e-8)
        x = x * 2 - 1
        return torch.clamp(x, -1.0, 1.0)


# =========================================================
# CoordinateBinner (unchanged)
# =========================================================
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def bin_to_value_torch(self, bin_index_tensor):
        bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
        bin_centers_tensor = torch.tensor(
            self.bin_centers, device=bin_index_tensor.device, dtype=torch.float32
        )
        return bin_centers_tensor[bin_index_tensor]


# =========================================================
# Loss (unchanged)
# =========================================================
def cross_entropy_loss(predictions, targets, pad_token=2,
                       ignore_index=-100, label_smoothing=0.05):
    B, T, V = predictions.shape
    mask = targets != pad_token
    predictions_flat = predictions.view(-1, V)
    targets_flat = targets.view(-1)
    targets_flat[~mask.view(-1)] = ignore_index

    loss_unreduced = F.cross_entropy(
        predictions_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    loss_unreduced = loss_unreduced.view(B, T)
    return loss_unreduced[mask].mean()


# =========================================================
# DDP utils (unchanged)
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

def stringify_config(cfg: dict) -> str:
    """
    Convert model_config into a short wandb-safe string.
    Example:
        {"d_model":512, "h":8, "N":6}
    becomes:
        d512_h8_n6_vocab204_lat50_drop0.1_pad2
    """
    parts = []
    for k, v in cfg.items():
        if isinstance(v, float):
            v = f"{v:.3g}"  # short float
        parts.append(f"{k}{v}")
    return "_".join(parts)


# =========================================================
# Checkpoint (unchanged)
# =========================================================
def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, model_config,
                         save_dir="./weights"):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config["d_model"]
    n_heads = model_config["h"]
    n_layers = model_config["N"]

    checkpoint_path = os.path.join(
        save_dir,
        f"LATENT_LLAMA_d{d_model}_h{n_heads}_n{n_layers}_bs{batch_size}_lr{lr}_best.pth"
    )
    torch.save(
        {
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": model_config,
        },
        checkpoint_path,
    )
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} (Val Loss: {best_loss:.6f})")


# =========================================================
# Training Loop
# =========================================================
def train(checkpoint_path=None, use_strict_resume=False):
    local_rank = setup_ddp()
    rank = get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Using device {device}")
    torch.set_float32_matmul_precision("medium")

    # ------------------ Config ------------------
    batch_size = 512
    num_epochs = 400
    lr = 1e-4
    seq_len = 17
    NUM_BINS = 201
    NUM_MECH_TYPES = 17
    NUM_SPECIAL_TOKENS = 3
    PAD_TOKEN = 2
    BIN_OFFSET = NUM_SPECIAL_TOKENS
    LATENT_DIM = 50
    LATENT_NOISE_STD = 0.00   # <--- added

    # ------------------ Dataset ------------------
    dataset = BarLinkageDataset(data_dir="/home/anurizada/Documents/processed_dataset_17")
    vocab_size = NUM_BINS + NUM_SPECIAL_TOKENS

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler,
                            batch_size=batch_size, num_workers=4, pin_memory=True)


    # ------------------ Model ------------------
    model_config = {
        "tgt_seq_len": seq_len,
        "d_model": 512,
        "h": 8,
        "N": 6,
        "num_labels": NUM_MECH_TYPES,
        "vocab_size": vocab_size,
        "latent_dim": LATENT_DIM,
        "dropout": 0.1,
        "pad_token_id": PAD_TOKEN,
        "debug": False,
    }

    model = LatentLLaMA_SingleToken(
        latent_dim=model_config["latent_dim"],
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

    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS-1)

    # ------------------ Optimizer + Cosine Scheduler ------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    warmup_epochs = 5
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_epochs])

    best_loss = float("inf")
    start_epoch = 0

    # ------------------ WandB ------------------
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name="LMBD_LATENT_LLaMA_" + stringify_config(model_config),
            config=model_config,
        )

    # =========================================================
    # Training Epochs
    # =========================================================
    for epoch in range(start_epoch, num_epochs):

        # ------------------ Training ------------------
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(total=len(train_loader), desc=f"[Rank {rank}] Train {epoch}",
                    disable=(rank != 0))

        for batch in train_loader:

            # ---------- Latents ----------
            latents = batch["vae_mu"].to(device).squeeze(-1)  # (B, 50)

            # 🔥 Add noise during training
            latents = latents + torch.randn_like(latents) * LATENT_NOISE_STD
            latents = torch.clamp(latents, -1, 1)

            # ---------- Forward ----------
            decoder_input = batch["decoder_input_discrete"].to(device)
            decoder_mask = batch["causal_mask"].to(device).bool()
            mech_labels = batch["encoded_labels"].to(device)
            target_tokens = batch["labels_discrete"].to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(decoder_input, decoder_mask, latents, mech_labels)

            ce_loss = cross_entropy_loss(predictions, target_tokens)

            # ---------- MSE ----------
            pred_tokens = predictions.argmax(dim=-1)
            pred_bin_rel = pred_tokens - BIN_OFFSET
            target_bin_rel = target_tokens - BIN_OFFSET
            coord_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS) & (target_tokens != PAD_TOKEN)

            if coord_mask.any():
                pred_cont = binner.bin_to_value_torch(pred_bin_rel.clamp(0, NUM_BINS - 1))
                target_cont = binner.bin_to_value_torch(target_bin_rel.clamp(0, NUM_BINS - 1))
                mse_loss = F.mse_loss(pred_cont[coord_mask], target_cont[coord_mask])
            else:
                mse_loss = torch.tensor(0.0, device=device)

            # ---------- Backprop ----------
            total_loss = ce_loss
            total_loss.backward()
            optimizer.step()

            # ---------- Metrics ----------
            valid_mask = target_tokens != PAD_TOKEN
            correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
            acc = correct / (valid_mask.sum().float() + 1e-12)

            epoch_loss += total_loss.item()
            epoch_acc += acc.item()

            # ---------- Logging ----------
            if rank == 0:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/ce_loss": ce_loss.item(),
                    "train/mse_loss": mse_loss.item(),
                    "train/accuracy": acc.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/noise_std": LATENT_NOISE_STD,
                })

            pbar.set_postfix({"loss": total_loss.item(), "acc": acc.item()})
            pbar.update(1)

        pbar.close()

        # Step LR scheduler
        scheduler.step()

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_mse_total = 0.0

        with torch.no_grad():
            pbar = tqdm(total=len(val_loader), desc=f"[Rank {rank}] Val {epoch}",
                        disable=(rank != 0))

            for batch in val_loader:

                # ---------- Latents ----------
                latents = batch["vae_mu"].to(device).squeeze(-1)

                # 🎯 OPTIONAL: noise in validation too  
                latents = latents + torch.randn_like(latents) * LATENT_NOISE_STD
                latents = torch.clamp(latents, -1, 1)

                # ---------- Forward ----------
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device).bool()
                mech_labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                predictions = model(decoder_input, decoder_mask, latents, mech_labels)
                ce_loss = cross_entropy_loss(predictions, target_tokens)

                # MSE
                pred_tokens = predictions.argmax(dim=-1)
                pred_bin_rel = pred_tokens - BIN_OFFSET
                target_bin_rel = target_tokens - BIN_OFFSET
                coord_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS) & (target_tokens != PAD_TOKEN)

                if coord_mask.any():
                    pred_cont = binner.bin_to_value_torch(pred_bin_rel.clamp(0, NUM_BINS - 1))
                    target_cont = binner.bin_to_value_torch(target_bin_rel.clamp(0, NUM_BINS - 1))
                    mse_loss = F.mse_loss(pred_cont[coord_mask], target_cont[coord_mask])
                else:
                    mse_loss = torch.tensor(0.0, device=device)

                # Accuracy
                valid_mask = target_tokens != PAD_TOKEN
                correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                acc = correct / (valid_mask.sum().float() + 1e-12)

                val_loss += ce_loss.item()
                val_acc += acc.item()
                val_mse_total += mse_loss.item()

                if rank == 0:
                    wandb.log({
                        "val/ce_loss": ce_loss.item(),
                        "val/mse_loss": mse_loss.item(),
                        "val/accuracy": acc.item(),
                        "val/noise_std": LATENT_NOISE_STD,
                        "val/lr": optimizer.param_groups[0]["lr"],
                    })

                pbar.set_postfix({"val_loss": ce_loss.item(), "val_acc": acc.item()})
                pbar.update(1)

            pbar.close()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_mse = val_mse_total / len(val_loader)

        if rank == 0:
            print(f"[Epoch {epoch}] TrainLoss={epoch_loss/len(train_loader):.4f} | "
                  f"ValLoss={avg_val_loss:.4f} | ValAcc={avg_val_acc:.4f} | ValMSE={avg_val_mse:.4f}")

            wandb.log({
                "val/loss": avg_val_loss,
                "val/acc": avg_val_acc,
                "val/mse": avg_val_mse,
                "epoch": epoch,
            })

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(model, optimizer, epoch, best_loss,
                                     batch_size, lr, model_config)

    cleanup_ddp()


if __name__ == "__main__":
    train(checkpoint_path=None, use_strict_resume=False)
