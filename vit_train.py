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
import random 

from vit_model import SingleImageTransformer
from dataset import BarLinkageDataset


# -------------------------
# CoordinateBinner
# -------------------------
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


# -------------------------
# Loss & metrics
# -------------------------
def cross_entropy_loss(predictions, targets, pad_token=2, ignore_index=-100, label_smoothing=0.1):
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


# -------------------------
# DDP helpers
# -------------------------
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


# -------------------------
# Checkpointing
# -------------------------
def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, model_config,
                         save_dir="./weights"):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config["d_model"]
    n_heads = model_config["h"]
    n_layers = model_config["N"]
    checkpoint_path = os.path.join(
        save_dir,
        f"d{d_model}_h{n_heads}_n{n_layers}_bs{batch_size}_lr{lr}_best.pth"
    )
    torch.save(
        {
            "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_config": model_config,
        },
        checkpoint_path,
    )
    print(f"[Rank {get_rank()}] ✅ Saved best model at {checkpoint_path} (Val Loss: {best_loss:.6f})")


# -------------------------
# Cosine LR schedule
# -------------------------
def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        progress = float(current_epoch - num_warmup_epochs) / float(
            max(1, num_training_epochs - num_warmup_epochs)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# -------------------------
# Helper: infer vocab size
# -------------------------
def compute_vocab_from_dataset(dataset):
    max_token = -1
    for i in range(min(len(dataset), 5000)):  # partial scan for speed
        s = dataset[i]
        max_token = max(
            max_token,
            int(s["decoder_input_discrete"].max().item()),
            int(s["labels_discrete"].max().item()),
        )
    vocab_size = max_token + 1
    print(f"✅ Detected vocab_size from dataset: {vocab_size}")
    return vocab_size


# -------------------------
# Training
# -------------------------
def train(checkpoint_path=None, use_strict_resume=False):
    local_rank = setup_ddp()
    rank = get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Using device: {device}")
    torch.set_float32_matmul_precision("medium")

    # ----------------- Hyperparams -----------------
    batch_size = 64
    num_epochs = 200
    lr = 1e-4
    mse_weight = 0.0
    seq_len = 18

    NUM_BINS = 201
    NUM_MECH_TYPES = 17
    NUM_SPECIAL_TOKENS = 3  # SOS, EOS, PAD

    SOS_TOKEN = 0
    EOS_TOKEN = 1
    PAD_TOKEN = 2
    BIN_OFFSET = NUM_SPECIAL_TOKENS + NUM_MECH_TYPES  # e.g. 20

    # ----------------- Dataset -----------------
    dataset = BarLinkageDataset(data_dir="/home/anurizada/Documents/processed_dataset")
    vocab_size = compute_vocab_from_dataset(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

    # ----------------- Model -----------------
    model_config = {
        "tgt_seq_len": seq_len,
        "output_size": vocab_size,
        "d_model": 512,
        "h": 8,
        "N": 1,
        "num_labels": NUM_MECH_TYPES,
        "vocab_size": vocab_size,
    }

    model = SingleImageTransformer(
        tgt_seq_len=model_config["tgt_seq_len"],
        d_model=model_config["d_model"],
        h=model_config["h"],
        N=model_config["N"],
        num_labels=model_config["num_labels"],
        vocab_size=model_config["vocab_size"],
        img_in_channels=1,
        img_size=64,
        pad_token_id=PAD_TOKEN,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # ----------------- Optimizer -----------------
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_loss = float("inf")
    start_epoch = 0

    # ----------------- WandB -----------------
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name=f"d{model_config['d_model']}_h{model_config['h']}_n{model_config['N']}_ViT_no_clip",
            config=model_config,
        )

    # ----------------- Resume -----------------
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Rank {rank}] 🔁 Resumed from {checkpoint_path} (epoch {start_epoch})")

    # ----------------- Training Loop -----------------
    for epoch in range(start_epoch, num_epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss, epoch_ce, epoch_mse, epoch_acc = 0.0, 0.0, 0.0, 0.0

        with tqdm(total=len(train_loader), desc=f"[Rank {rank}] Train {epoch}", disable=rank != 0) as pbar:
            for batch in train_loader:
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device).bool()
                images = batch["images"].to(device)
                labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                # ----- Scheduled Sampling -----
                if epoch > 400:
                    ss_prob = min(0.2, 0.01 * (epoch - 10))
                    if random.random() < ss_prob:
                        with torch.no_grad():
                            logits_ss = model(decoder_input, decoder_mask, images, labels)
                            preds_ss = logits_ss.argmax(dim=-1)
                        mask = (torch.rand_like(decoder_input.float()) < ss_prob)
                        decoder_input = torch.where(mask, preds_ss, decoder_input)
                # -----------------------------
                
                optimizer.zero_grad()
                predictions = model(decoder_input, decoder_mask, images, labels)

                ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

                total_loss = ce_loss
                total_loss.backward()
                optimizer.step()

                # --- Optional MSE reconstruction loss ---
                pred_tokens = predictions.argmax(dim=-1)
                pred_bin_rel = pred_tokens - BIN_OFFSET
                target_bin_rel = target_tokens - BIN_OFFSET
                target_numeric_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS)
                nonpad_mask = target_tokens != PAD_TOKEN
                mse_mask = target_numeric_mask & nonpad_mask

                if mse_mask.any():
                    pred_clamped = torch.clamp(pred_bin_rel, 0, NUM_BINS - 1).long()
                    target_clamped = torch.clamp(target_bin_rel, 0, NUM_BINS - 1).long()
                    pred_cont = binner.bin_to_value_torch(pred_clamped)
                    target_cont = binner.bin_to_value_torch(target_clamped)
                    mse_loss = F.mse_loss(pred_cont[mse_mask], target_cont[mse_mask])
                else:
                    mse_loss = torch.tensor(0.0, device=device)

                # Accuracy
                valid_mask = target_tokens != PAD_TOKEN
                correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                batch_acc = correct / (valid_mask.sum().float() + 1e-12)

                # Accumulate
                epoch_loss += total_loss.item()
                epoch_ce += ce_loss.item()
                epoch_mse += mse_loss.item()
                epoch_acc += batch_acc.item()

                if rank == 0:
                    wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/accuracy": batch_acc.item(),
                        "epoch": epoch,
                    })
                pbar.set_postfix({"Loss": total_loss.item(), "Acc": batch_acc.item()})
                pbar.update(1)

        # ----------------- Validation -----------------
        model.eval()
        val_loss, val_ce, val_mse, val_acc = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"[Rank {rank}] Val {epoch}", disable=rank != 0) as pbar:
                for batch in val_loader:
                    decoder_input = batch["decoder_input_discrete"].to(device)
                    decoder_mask = batch["causal_mask"].to(device).bool()
                    images = batch["images"].to(device)
                    labels = batch["encoded_labels"].to(device)
                    target_tokens = batch["labels_discrete"].to(device)

                    predictions = model(decoder_input, decoder_mask, images, labels)
                    ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

                    pred_tokens = predictions.argmax(dim=-1)
                    pred_bin_rel = pred_tokens - BIN_OFFSET
                    target_bin_rel = target_tokens - BIN_OFFSET
                    target_numeric_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS)
                    nonpad_mask = target_tokens != PAD_TOKEN
                    mse_mask = target_numeric_mask & nonpad_mask

                    if mse_mask.any():
                        pred_clamped = torch.clamp(pred_bin_rel, 0, NUM_BINS - 1).long()
                        target_clamped = torch.clamp(target_bin_rel, 0, NUM_BINS - 1).long()
                        pred_cont = binner.bin_to_value_torch(pred_clamped)
                        target_cont = binner.bin_to_value_torch(target_clamped)
                        mse_loss = F.mse_loss(pred_cont[mse_mask], target_cont[mse_mask])
                    else:
                        mse_loss = torch.tensor(0.0, device=device)

                    total_loss = ce_loss + mse_weight * mse_loss
                    valid_mask = target_tokens != PAD_TOKEN
                    correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                    batch_acc = correct / (valid_mask.sum().float() + 1e-12)

                    val_loss += total_loss.item()
                    val_ce += ce_loss.item()
                    val_mse += mse_loss.item()
                    val_acc += batch_acc.item()

                    if rank == 0:
                        wandb.log({
                            "val/total_loss": total_loss.item(),
                            "val/ce_loss": ce_loss.item(),
                            "val/mse_loss": mse_loss.item(),
                            "val/accuracy": batch_acc.item(),
                            "epoch": epoch,
                        })
                    pbar.set_postfix({"Val Loss": total_loss.item(), "Val Acc": batch_acc.item()})
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)

        if rank == 0:
            print(f"[Epoch {epoch}] TrainLoss {epoch_loss/len(train_loader):.4f} | ValLoss {avg_val_loss:.4f}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, model_config)

    cleanup_ddp()


if __name__ == "__main__":
    train(checkpoint_path=None, use_strict_resume=False)
