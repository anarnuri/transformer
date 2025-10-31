# train_coords_only_no_clip.py
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DistributedSampler, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from vit_model import SingleImageTransformerCLIP   # still handles mech_labels
from dataset import BarLinkageDataset


# -------------------------------------------------------
# CoordinateBinner
# -------------------------------------------------------
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


# -------------------------------------------------------
# Losses
# -------------------------------------------------------
def cross_entropy_loss(predictions, targets, pad_token=2,
                       ignore_index=-100, label_smoothing=0.1):
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
    loss = loss_unreduced[mask].mean()
    return loss


# -------------------------------------------------------
# DDP helpers
# -------------------------------------------------------
def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# -------------------------------------------------------
# Scheduler
# -------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        progress = float(current_epoch - num_warmup_epochs) / float(
            max(1, num_training_epochs - num_warmup_epochs)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
def train(checkpoint_path=None, use_strict_resume=False):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    setup_ddp()
    print(f"[Rank {rank}] Using device {device}")
    torch.set_float32_matmul_precision("medium")

    # ----------------- Hyperparams -----------------
    batch_size = 2
    num_epochs = 200
    lr = 1e-4
    mse_weight = 1.0
    seq_len = 25

    NUM_BINS = 201
    NUM_SPECIAL_TOKENS = 3   # e.g. SOS, EOS, PAD
    PAD_TOKEN = 2
    BIN_OFFSET = NUM_SPECIAL_TOKENS
    vocab_size = NUM_SPECIAL_TOKENS + NUM_BINS

    # ----------------- Dataset -----------------
    dataset = BarLinkageDataset(data_dir="/home/anurizada/Documents/processed_dataset_17")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # ----------------- Model -----------------
    model_config = {
        "tgt_seq_len": seq_len,
        "output_size": vocab_size,
        "d_model": 256,
        "h": 8,
        "N": 1,
        "num_labels": 17,   # mech types
        "vocab_size": vocab_size,
    }

    model = SingleImageTransformerCLIP(  # accepts mech_labels argument
        tgt_seq_len=model_config["tgt_seq_len"],
        d_model=model_config["d_model"],
        h=model_config["h"],
        N=model_config["N"],
        num_labels=model_config["num_labels"],
        vocab_size=model_config["vocab_size"],
    ).to(device)

    model = DDP(model, device_ids=[local_rank])
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 10, num_epochs)

    # ----------------- W&B -----------------
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name="coords_only_no_clip",
            config=model_config,
        )

    best_loss = float("inf")

    # ----------------- Training Loop -----------------
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"[Rank {rank}] Train {epoch}", leave=False) as pbar:
            for batch in train_loader:
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device)
                images = batch["images"].to(device)
                mech_labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                optimizer.zero_grad()

                # model now returns only logits
                predictions = model(decoder_input, decoder_mask, images, mech_labels)

                ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

                # --- numeric MSE ---
                pred_tokens = predictions.argmax(dim=-1)
                pred_bin_rel = pred_tokens - BIN_OFFSET
                target_bin_rel = target_tokens - BIN_OFFSET
                coord_mask = (
                    (target_bin_rel >= 0)
                    & (target_bin_rel < NUM_BINS)
                    & (target_tokens != PAD_TOKEN)
                )

                if coord_mask.any():
                    pred_cont = binner.bin_to_value_torch(torch.clamp(pred_bin_rel, 0, NUM_BINS - 1))
                    target_cont = binner.bin_to_value_torch(torch.clamp(target_bin_rel, 0, NUM_BINS - 1))
                    mse_loss = F.mse_loss(pred_cont[coord_mask], target_cont[coord_mask])
                else:
                    mse_loss = torch.tensor(0.0, device=device)

                total_loss = ce_loss + mse_weight * mse_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                pbar.set_postfix({"Loss": total_loss.item()})
                pbar.update(1)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device)
                images = batch["images"].to(device)
                mech_labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                predictions = model(decoder_input, decoder_mask, images, mech_labels)
                ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

                pred_tokens = predictions.argmax(dim=-1)
                pred_bin_rel = pred_tokens - BIN_OFFSET
                target_bin_rel = target_tokens - BIN_OFFSET
                coord_mask = (
                    (target_bin_rel >= 0)
                    & (target_bin_rel < NUM_BINS)
                    & (target_tokens != PAD_TOKEN)
                )

                if coord_mask.any():
                    pred_cont = binner.bin_to_value_torch(torch.clamp(pred_bin_rel, 0, NUM_BINS - 1))
                    target_cont = binner.bin_to_value_torch(torch.clamp(target_bin_rel, 0, NUM_BINS - 1))
                    mse_loss = F.mse_loss(pred_cont[coord_mask], target_cont[coord_mask])
                else:
                    mse_loss = torch.tensor(0.0, device=device)

                total_loss = ce_loss + mse_weight * mse_loss
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # ----------------- Logging & checkpoint -----------------
        if rank == 0:
            wandb.log({
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "epoch": epoch,
            })
            print(f"[Epoch {epoch}] Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                os.makedirs("./weights", exist_ok=True)
                torch.save(model.module.state_dict(), "./weights/best_coords_only_no_clip.pth")
                print(f"✅ Saved best checkpoint (Val {best_loss:.4f})")

    cleanup_ddp()


if __name__ == "__main__":
    train()
