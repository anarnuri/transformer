import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DistributedSampler, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from vit_llama_model import SingleImageTransformerCLIP_LLaMA  # your model
from dataset import BarLinkageDataset


# =========================================================
# CoordinateBinner
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
# Loss
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
# Checkpoint
# =========================================================
def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, model_config,
                         save_dir="./weights"):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config["d_model"]
    n_heads = model_config["h"]
    n_layers = model_config["N"]
    checkpoint_path = os.path.join(
        save_dir,
        f"LLAMA_d{d_model}_h{n_heads}_n{n_layers}_bs{batch_size}_lr{lr}_best.pth"
    )
    torch.save(
        {
            "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": model_config,
        },
        checkpoint_path,
    )
    print(f"[Rank {get_rank()}] ✅ Saved best model at {checkpoint_path} (Val Loss: {best_loss:.6f})")


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
    lr = 1e-3
    seq_len = 17
    NUM_BINS = 201
    NUM_MECH_TYPES = 17
    NUM_SPECIAL_TOKENS = 3
    PAD_TOKEN = 2
    BIN_OFFSET = NUM_SPECIAL_TOKENS

    # ------------------ Dataset ------------------
    dataset = BarLinkageDataset(data_dir="/home/anurizada/Documents/processed_dataset_17")
    vocab_size = NUM_BINS + NUM_SPECIAL_TOKENS

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

    # ------------------ Model ------------------
    model_config = {
        "tgt_seq_len": seq_len,
        "d_model": 512,
        "h": 8,
        "N": 6,
        "num_labels": NUM_MECH_TYPES,
        "vocab_size": vocab_size,
        "img_patch": 8,
        "dropout": 0.1,
        "pad_token_id": PAD_TOKEN,
        "debug": False,
    }

    model = SingleImageTransformerCLIP_LLaMA(
        tgt_seq_len=model_config["tgt_seq_len"],
        d_model=model_config["d_model"],
        h=model_config["h"],
        N=model_config["N"],
        num_labels=model_config["num_labels"],
        vocab_size=model_config["vocab_size"],
        dropout=model_config["dropout"],
        pad_token_id=model_config["pad_token_id"],
        debug=model_config["debug"],
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # ------------------ Optimizer + Scheduler ------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_loss = float("inf")
    start_epoch = 0

    # ------------------ WandB ------------------
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name=f"LLaMA_d{model_config['d_model']}_h{model_config['h']}_n{model_config['N']}_bs{batch_size}_lr{lr}",
            config=model_config,
        )

    # ------------------ Resume ------------------
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Rank {rank}] 🔁 Resumed from {checkpoint_path}")

    # =========================================================
    # Training Epochs
    # =========================================================
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss, epoch_acc = 0.0, 0.0

        pbar = tqdm(total=len(train_loader), desc=f"[Rank {rank}] Train {epoch}", disable=rank != 0)
        for step, batch in enumerate(train_loader):
            decoder_input = batch["decoder_input_discrete"].to(device)
            decoder_mask = batch["causal_mask"].to(device).bool()
            images = batch["images"].to(device)
            mech_labels = batch["encoded_labels"].to(device)
            target_tokens = batch["labels_discrete"].to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(decoder_input, decoder_mask, images, mech_labels)
            ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

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

            total_loss = ce_loss
            total_loss.backward()
            optimizer.step()

            valid_mask = target_tokens != PAD_TOKEN
            correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
            acc = correct / (valid_mask.sum().float() + 1e-12)

            epoch_loss += total_loss.item()
            epoch_acc += acc.item()

            if rank == 0:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/ce_loss": ce_loss.item(),
                    "train/mse_loss": mse_loss.item(),
                    "train/accuracy": acc.item(),
                    "epoch": epoch,
                })

            pbar.set_postfix({"loss": total_loss.item(), "acc": acc.item()})
            pbar.update(1)
        pbar.close()

        # ------------------ Validation ------------------
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            pbar = tqdm(total=len(val_loader), desc=f"[Rank {rank}] Val {epoch}", disable=rank != 0)
            for batch in val_loader:
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device).bool()
                images = batch["images"].to(device)
                mech_labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                predictions = model(decoder_input, decoder_mask, images, mech_labels)
                ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

                pred_tokens = predictions.argmax(dim=-1)
                valid_mask = target_tokens != PAD_TOKEN
                correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                acc = correct / (valid_mask.sum().float() + 1e-12)
                total_loss = ce_loss

                val_loss += total_loss.item()
                val_acc += acc.item()

                pbar.set_postfix({"val_loss": total_loss.item(), "val_acc": acc.item()})
                pbar.update(1)
            pbar.close()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        if rank == 0:
            wandb.log({"val/loss": avg_val_loss, "val/acc": avg_val_acc, "epoch": epoch})
            print(f"[Epoch {epoch}] TrainLoss={epoch_loss/len(train_loader):.4f} | ValLoss={avg_val_loss:.4f} | ValAcc={avg_val_acc:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, model_config)

    cleanup_ddp()


if __name__ == "__main__":
    train(checkpoint_path=None, use_strict_resume=False)
