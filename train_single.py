# train_with_ce_mse.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DistributedSampler, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
import wandb

from model import SingleImageTransformer
from dataset import BarLinkageDataset  # your dataset (unchanged)

# -------------------------
# CoordinateBinner (your provided implementation)
# -------------------------
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def value_to_bin(self, value):
        clipped_value = np.clip(value, -self.kappa, self.kappa)
        bin_index = np.floor((clipped_value + self.kappa) * (self.num_bins - 1) / (2 * self.kappa))
        return bin_index.astype(int)

    def bin_to_value(self, bin_index):
        bin_index = np.clip(bin_index, 0, self.num_bins - 1)
        return self.bin_centers[bin_index]

    def value_to_bin_torch(self, value_tensor):
        clipped_value = torch.clamp(value_tensor, -self.kappa, self.kappa)
        bin_index = torch.floor((clipped_value + self.kappa) * (self.num_bins - 1) / (2 * self.kappa))
        return bin_index.long()

    def bin_to_value_torch(self, bin_index_tensor):
        bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
        bin_centers_tensor = torch.tensor(self.bin_centers, device=bin_index_tensor.device, dtype=torch.float32)
        return bin_centers_tensor[bin_index_tensor]


# -------------------------
# Loss / metrics (mostly unchanged)
# -------------------------
def cross_entropy_loss(predictions, targets, pad_token=2, ignore_index=-100, label_smoothing=0.1, debug=False):
    """
    Cross entropy loss with optional label smoothing.
    """
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
        label_smoothing=label_smoothing
    )
    loss_unreduced = loss_unreduced.view(B, T)
    loss = loss_unreduced[mask].mean()
    return loss


def accuracy(predictions, targets, pad_token=2, debug=False):
    pred_tokens = predictions.argmax(dim=-1)  # [B, T]
    mask = targets != pad_token
    correct = (pred_tokens == targets) & mask
    # avoid division by zero
    denom = mask.sum().float()
    acc = correct.sum().float() / (denom + 1e-12)
    if debug:
        print("\n=== Accuracy Debug ===")
        print(f"Pred tokens shape: {pred_tokens.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"#non-pad tokens: {mask.sum().item()}")
        print(f"Accuracy: {acc.item():.4f}")
    return acc

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, init_scale=1/0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_scale)))

    def forward(self, image_embeddings, label_embeddings):
        image_embeddings = F.normalize(image_embeddings.squeeze(), p=2, dim=1)
        label_embeddings = F.normalize(label_embeddings.squeeze(), p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeddings @ label_embeddings.t()

        N = logits.shape[0]
        targets = torch.arange(N, device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        return (loss_i2t + loss_t2i) / 2


# -------------------------
# DDP helpers, checkpointing
# -------------------------
def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, clip_loss_fn, model_config, save_dir="./weights"):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config['d_model']
    n_heads = model_config['h']
    n = model_config['N']
    checkpoint_path = os.path.join(save_dir, f"d{d_model}_h{n_heads}_n{n}_bs{batch_size}_lr{lr}_best.pth")
    torch.save({
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'clip_loss_state_dict': clip_loss_fn.state_dict() if clip_loss_fn is not None else None,
        'epoch': epoch,
        'best_loss': best_loss,
        'batch_size': batch_size,
        'learning_rate': lr,
        'model_config': model_config,
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} with loss {best_loss:.6f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_gradient_flow(model, epoch):
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            wandb.log({f"grad/{name}": grad_norm})
            total_grad_norm += grad_norm ** 2
    total_grad_norm = total_grad_norm ** 0.5
    wandb.log({"grad/total": total_grad_norm, "epoch": epoch})
    return

def log_learning_rates(optimizer, epoch, prefix="train"):
    for i, group in enumerate(optimizer.param_groups):
        wandb.log({f"{prefix}/lr/group_{i}": group['lr'], "epoch": epoch})

def load_model_weights(checkpoint_path, model, clip_loss_fn=None, map_location="cpu"):
    """Load only matching weights from checkpoint (skip missing/unexpected)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    ckpt_state = checkpoint["model_state_dict"]

    # Handle DDP wrapping: strip "module." prefix if needed
    new_ckpt_state = {}
    for k, v in ckpt_state.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        new_ckpt_state[new_key] = v

    # Load partially: only matching keys
    model_state = model.state_dict()
    matched = {k: v for k, v in new_ckpt_state.items() if k in model_state and v.shape == model_state[k].shape}

    missing_keys = set(model_state.keys()) - set(matched.keys())
    unexpected_keys = set(new_ckpt_state.keys()) - set(matched.keys())

    model.load_state_dict(matched, strict=False)

    if clip_loss_fn is not None and checkpoint.get("clip_loss_state_dict") is not None:
        clip_loss_fn.load_state_dict(checkpoint["clip_loss_state_dict"], strict=False)

    print(f"[Rank {get_rank()}] Loaded weights from {checkpoint_path}")
    print(f"Matched keys: {len(matched)} | Missing: {len(missing_keys)} | Unexpected: {len(unexpected_keys)}")
    return missing_keys, unexpected_keys

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        # cosine decay after warmup
        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

# -------------------------
# Training loop (modified to include MSE via binner)
# -------------------------
def train(checkpoint_path=None, use_strict_resume=False):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    device = torch.device(f'cuda:{local_rank}')
    print(f"Using device: {device}")

    setup_ddp()
    torch.set_float32_matmul_precision('medium')

    # Hyperparams
    batch_size = 512
    num_epochs = 200
    lr = 1e-4
    clip_loss_weight = 1.0
    mse_weight = 1.0       # weight on MSE component
    seq_len = 25

    # Binner config
    NUM_BINS = 200
    NUM_SPECIAL_TOKENS = 3    # adjust if different in your vocab (e.g., 3 special tokens)
    PAD_TOKEN = 2             # same as your code
    BIN_OFFSET = NUM_SPECIAL_TOKENS  # the vocab index where bin 0 maps to

    # Dataset (unchanged)
    dataset = BarLinkageDataset(data_dir='/home/anurizada/Documents/processed_dataset_17')
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    # Binner instance
    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # Model config
    vocab_size = NUM_BINS + NUM_SPECIAL_TOKENS
    num_labels = 105 # len(dataset.label_mapping['label_to_index'])
    model_config = {
        'tgt_seq_len': seq_len,
        'output_size': vocab_size,
        'd_model': 512,
        'h': 8,
        'N': 1,
        'num_labels': num_labels,
        'vocab_size': vocab_size
    }

    model = SingleImageTransformer(
        tgt_seq_len=model_config['tgt_seq_len'],
        d_model=model_config['d_model'],
        h=model_config['h'],
        N=model_config['N'],
        num_labels=model_config['num_labels']
    ).to(device)

    total_params = count_parameters(model)
    print(f"[Rank {rank}] Model created with {total_params:,} trainable parameters")
    print(f"[Rank {rank}] Vocabulary size: {vocab_size}, Number of labels: {num_labels}")

    model = DDP(model, device_ids=[local_rank])
    clip_loss_fn = CLIPContrastiveLoss().to(device)

    # WandB
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name=f"discrete_d{model_config['d_model']}_h{model_config['h']}_n{model_config['N']}",
            config={
                "output_size": model_config['output_size'],
                "tgt_seq_len": model_config['tgt_seq_len'],
                "d_model": model_config['d_model'],
                "n_heads": model_config['h'],
                "n_layers": model_config['N'],
                "num_labels": num_labels,
                "vocab_size": vocab_size,
                "batch_size": batch_size,
                "lr": lr,
                "clip_loss_weight": clip_loss_weight,
                "mse_weight": mse_weight,
                "total_params": total_params
            }
        )

    optimizer = Adam([
        {'params': model.parameters()},
        {'params': clip_loss_fn.parameters()}
    ], lr=lr, weight_decay=1e-5)

    num_warmup_epochs = 10
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_epochs)
    best_loss = float("inf")

    if checkpoint_path is not None:
        if use_strict_resume:
            # full resume (for continuing same dataset)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("clip_loss_state_dict") is not None:
                clip_loss_fn.load_state_dict(checkpoint["clip_loss_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"[Rank {get_rank()}] Resumed training from {checkpoint_path} at epoch {start_epoch}")
        else:
            # lightweight load (only model + clip_loss weights)
            load_model_weights(checkpoint_path, model, clip_loss_fn, map_location=device)
            # start fresh optimizer/scheduler
            start_epoch = 0
            best_loss = float("inf")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_mse = 0.0
        epoch_acc = 0.0

        with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Debug prints first batch
                if epoch == 0 and batch_idx == 0 and rank == 0:
                    dec_in = batch["decoder_input_discrete"][0].cpu().numpy()
                    labels_discrete = batch["labels_discrete"][0].cpu().numpy()
                    print("\n=== Dataset Debug ===")
                    print("Decoder Input:", dec_in.tolist())
                    print("Target Labels:", labels_discrete.tolist())
                    pad_token = PAD_TOKEN
                    print("Pad tokens in decoder_input:", (dec_in == pad_token).sum())
                    print("Pad tokens in labels_discrete:", (labels_discrete == pad_token).sum())

                # Move batch to device
                decoder_input = batch["decoder_input_discrete"].to(device)   # (B, T)
                decoder_mask = batch["causal_mask"].to(device)
                images = batch["images"].to(device)
                labels = batch["encoded_labels"].to(device)                 # maybe used by model
                target_tokens = batch["labels_discrete"].to(device)        # (B, T)

                optimizer.zero_grad()
                # Model forward (keeps your original signature)
                predictions, image_emb, label_emb = model(decoder_input, decoder_mask, images, labels)
                # predictions: (B, T, V)

                # ---- Cross-entropy loss (unchanged) ----
                ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

                # ---- MSE: map predicted bins and target bins to continuous values ----
                # predicted vocab indices
                pred_tokens = predictions.argmax(dim=-1)  # (B, T)
                # relative bin index (0..NUM_BINS-1) = vocab_index - BIN_OFFSET
                pred_bin_rel = pred_tokens - BIN_OFFSET
                target_bin_rel = target_tokens - BIN_OFFSET

                # mask positions where target is a numeric bin (not special token) and not PAD
                target_numeric_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS)
                nonpad_mask = (target_tokens != PAD_TOKEN)
                mse_mask = (target_numeric_mask & nonpad_mask)

                if mse_mask.sum() > 0:
                    # clamp predictions into [0, NUM_BINS-1]
                    pred_bin_rel_clamped = torch.clamp(pred_bin_rel, 0, NUM_BINS - 1).long()
                    target_bin_rel_clamped = torch.clamp(target_bin_rel, 0, NUM_BINS - 1).long()

                    # convert to continuous via binner (on device)
                    pred_cont = binner.bin_to_value_torch(pred_bin_rel_clamped).to(device)    # (B, T) float
                    target_cont = binner.bin_to_value_torch(target_bin_rel_clamped).to(device)

                    mse_loss = F.mse_loss(pred_cont[mse_mask], target_cont[mse_mask])
                else:
                    mse_loss = torch.tensor(0.0, device=device, requires_grad=True)

                # ---- CLIP loss ----
                clip_loss = clip_loss_fn(image_emb, label_emb)

                # ---- Total loss ----
                total_loss = ce_loss + mse_weight * mse_loss + clip_loss_weight * clip_loss

                # Backprop
                total_loss.backward()
                optimizer.step()

                # Metrics (accuracy over non-pad positions)
                valid_mask = (target_tokens != PAD_TOKEN)
                if valid_mask.sum() > 0:
                    correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                    batch_acc = correct / valid_mask.sum().float()
                else:
                    batch_acc = torch.tensor(0.0, device=device)

                # Logging & accumulation
                epoch_loss += total_loss.item()
                epoch_ce += ce_loss.item()
                epoch_mse += mse_loss.item()
                epoch_acc += batch_acc.item()

                if rank == 0:
                    log_learning_rates(optimizer, epoch)
                    log_gradient_flow(model, epoch)

                    wandb.log({
                        "train/ce_loss": ce_loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/clip_loss": clip_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/accuracy": batch_acc.item(),
                        "epoch": epoch,
                        "batch": epoch * len(train_loader) + batch_idx
                    })

                pbar.set_postfix({"Loss": total_loss.item(), "Acc": batch_acc.item()})
                pbar.update(1)

        # scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ce = epoch_ce / len(train_loader)
        avg_train_mse = epoch_mse / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_ce = 0.0
        val_mse = 0.0
        val_acc = 0.0

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Rank {rank} Val Epoch {epoch}", leave=False) as pbar:
                for batch_idx, batch in enumerate(val_loader):
                    decoder_input = batch["decoder_input_discrete"].to(device)
                    decoder_mask = batch["causal_mask"].to(device)
                    images = batch["images"].to(device)
                    labels = batch["encoded_labels"].to(device)
                    target_tokens = batch["labels_discrete"].to(device)

                    predictions, image_emb, label_emb = model(decoder_input, decoder_mask, images, labels)

                    ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)
                    pred_tokens = predictions.argmax(dim=-1)
                    pred_bin_rel = pred_tokens - BIN_OFFSET
                    target_bin_rel = target_tokens - BIN_OFFSET

                    target_numeric_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS)
                    nonpad_mask = (target_tokens != PAD_TOKEN)
                    mse_mask = (target_numeric_mask & nonpad_mask)

                    if mse_mask.sum() > 0:
                        pred_bin_rel_clamped = torch.clamp(pred_bin_rel, 0, NUM_BINS - 1).long()
                        target_bin_rel_clamped = torch.clamp(target_bin_rel, 0, NUM_BINS - 1).long()
                        pred_cont = binner.bin_to_value_torch(pred_bin_rel_clamped).to(device)
                        target_cont = binner.bin_to_value_torch(target_bin_rel_clamped).to(device)
                        mse_loss = F.mse_loss(pred_cont[mse_mask], target_cont[mse_mask])
                    else:
                        mse_loss = torch.tensor(0.0, device=device)

                    clip_loss = clip_loss_fn(image_emb, label_emb)
                    total_loss = ce_loss + mse_weight * mse_loss + clip_loss_weight * clip_loss

                    # Accuracy
                    valid_mask = (target_tokens != PAD_TOKEN)
                    if valid_mask.sum() > 0:
                        correct = ((pred_tokens == target_tokens) & valid_mask).sum().float()
                        batch_acc = correct / valid_mask.sum().float()
                    else:
                        batch_acc = torch.tensor(0.0, device=device)

                    val_loss += total_loss.item()
                    val_ce += ce_loss.item()
                    val_mse += mse_loss.item()
                    val_acc += batch_acc.item()

                    if rank == 0:
                        wandb.log({
                            "val/ce_loss": ce_loss.item(),
                            "val/mse_loss": mse_loss.item(),
                            "val/clip_loss": clip_loss.item(),
                            "val/total_loss": total_loss.item(),
                            "val/accuracy": batch_acc.item(),
                            "epoch": epoch
                        })

                    pbar.set_postfix({"Val Loss": total_loss.item(), "Val Acc": batch_acc.item()})
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_ce = val_ce / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, CE: {avg_train_ce:.6f}, MSE: {avg_train_mse:.6f}, Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, CE: {avg_val_ce:.6f}, MSE: {avg_val_mse:.6f}, Acc: {avg_val_acc:.4f}")

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_loss=best_loss,
                    batch_size=batch_size,
                    lr=lr,
                    clip_loss_fn=clip_loss_fn,
                    model_config=model_config
                )

    cleanup_ddp()

if __name__ == "__main__":
    train(checkpoint_path=None, use_strict_resume=False)

# # ======================================================
# # train_with_ce_mse.py  (with differentiable + hard MSE)
# # ======================================================
# import os
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import random_split, DistributedSampler, DataLoader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import LambdaLR
# from tqdm import tqdm
# import wandb

# from model import SingleImageTransformer
# from dataset import BarLinkageDataset  # your dataset (unchanged)


# # -------------------------
# # CoordinateBinner
# # -------------------------
# class CoordinateBinner:
#     def __init__(self, kappa=1.0, num_bins=200):
#         self.kappa = kappa
#         self.num_bins = num_bins
#         self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
#         self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
#         self._bin_centers_t = None

#     def value_to_bin_torch(self, value_tensor):
#         clipped_value = torch.clamp(value_tensor, -self.kappa, self.kappa)
#         bin_index = torch.floor(
#             (clipped_value + self.kappa) * (self.num_bins - 1) / (2 * self.kappa)
#         )
#         return bin_index.long()

#     def bin_to_value_torch(self, bin_index_tensor):
#         if (self._bin_centers_t is None) or (self._bin_centers_t.device != bin_index_tensor.device):
#             self._bin_centers_t = torch.tensor(
#                 self.bin_centers, device=bin_index_tensor.device, dtype=torch.float32
#             )
#         bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
#         return self._bin_centers_t[bin_index_tensor]


# # -------------------------
# # Losses / metrics
# # -------------------------
# def cross_entropy_loss(predictions, targets, pad_token=2, ignore_index=-100, label_smoothing=0.1):
#     B, T, V = predictions.shape
#     mask = targets != pad_token
#     predictions_flat = predictions.view(-1, V)
#     targets_flat = targets.view(-1)
#     targets_flat[~mask.view(-1)] = ignore_index
#     loss_unreduced = F.cross_entropy(
#         predictions_flat,
#         targets_flat,
#         ignore_index=ignore_index,
#         reduction="none",
#         label_smoothing=label_smoothing,
#     )
#     loss_unreduced = loss_unreduced.view(B, T)
#     return loss_unreduced[mask].mean()


# def accuracy(predictions, targets, pad_token=2):
#     pred_tokens = predictions.argmax(dim=-1)
#     mask = targets != pad_token
#     correct = (pred_tokens == targets) & mask
#     return correct.sum().float() / (mask.sum().float() + 1e-12)


# class CLIPContrastiveLoss(nn.Module):
#     def __init__(self, init_scale=1 / 0.07):
#         super().__init__()
#         self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_scale)))

#     def forward(self, image_embeddings, label_embeddings):
#         image_embeddings = F.normalize(image_embeddings.squeeze(), p=2, dim=1)
#         label_embeddings = F.normalize(label_embeddings.squeeze(), p=2, dim=1)
#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_embeddings @ label_embeddings.t()
#         N = logits.shape[0]
#         targets = torch.arange(N, device=logits.device)
#         loss_i2t = F.cross_entropy(logits, targets)
#         loss_t2i = F.cross_entropy(logits.t(), targets)
#         return 0.5 * (loss_i2t + loss_t2i)


# # -------------------------
# # DDP helpers
# # -------------------------
# def setup_ddp():
#     dist.init_process_group("nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# def cleanup_ddp():
#     dist.destroy_process_group()


# def get_rank():
#     return dist.get_rank() if dist.is_initialized() else 0


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def save_best_checkpoint(
#     model, optimizer, epoch, best_loss, batch_size, lr, clip_loss_fn, model_config, save_dir="./weights"
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     d_model = model_config["d_model"]
#     n_heads = model_config["h"]
#     n = model_config["N"]
#     path = os.path.join(save_dir, f"d{d_model}_h{n_heads}_n{n}_bs{batch_size}_lr{lr}_best.pth")
#     torch.save(
#         {
#             "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "clip_loss_state_dict": clip_loss_fn.state_dict() if clip_loss_fn else None,
#             "epoch": epoch,
#             "best_loss": best_loss,
#             "batch_size": batch_size,
#             "learning_rate": lr,
#             "model_config": model_config,
#         },
#         path,
#     )
#     print(f"[Rank {get_rank()}] Saved best model to {path} (val loss {best_loss:.6f})")


# # -------------------------
# # Training loop
# # -------------------------
# def train(checkpoint_path=None, use_strict_resume=False):
#     rank = int(os.environ["RANK"])
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])

#     setup_ddp()
#     device = torch.device(f"cuda:{local_rank}")
#     torch.set_float32_matmul_precision("medium")

#     # ---------------- Hyperparameters ----------------
#     batch_size = 512
#     num_epochs = 200
#     lr = 1e-4
#     clip_loss_weight = 0.1
#     mse_weight = 0.1
#     seq_len = 25
#     NUM_BINS = 200
#     NUM_SPECIAL_TOKENS = 3
#     PAD_TOKEN = 2
#     BIN_OFFSET = NUM_SPECIAL_TOKENS

#     # ---------------- Dataset ----------------
#     dataset = BarLinkageDataset(data_dir="/home/anurizada/Documents/processed_dataset")
#     dataset = torch.utils.data.Subset(dataset, range(1000000))
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
#     train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
#     val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

#     # ---------------- Model setup ----------------
#     binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)
#     vocab_size = NUM_BINS + NUM_SPECIAL_TOKENS + 1
#     num_labels = 105
#     model_config = dict(
#         tgt_seq_len=seq_len, d_model=512, h=8, N=6, num_labels=num_labels, vocab_size=vocab_size
#     )
#     model = SingleImageTransformer(**model_config).to(device)
#     print(f"[Rank {rank}] Model parameters: {count_parameters(model):,}")
#     model = DDP(model, device_ids=[local_rank])
#     clip_loss_fn = CLIPContrastiveLoss().to(device)

#     if rank == 0:
#         wandb.init(
#             project="bar-linkage-transformer",
#             name=f"discrete_d{model_config['d_model']}_h{model_config['h']}_n{model_config['N']}",
#             config=dict(**model_config, batch_size=batch_size, lr=lr, clip_loss_weight=clip_loss_weight, mse_weight=mse_weight),
#         )

#     optimizer = Adam([{"params": model.parameters()}, {"params": clip_loss_fn.parameters()}], lr=lr, weight_decay=1e-5)
#     best_loss = float("inf")

#     # ---------------- Training ----------------
#     for epoch in range(num_epochs):
#         train_sampler.set_epoch(epoch)
#         model.train()
#         epoch_loss = epoch_ce = epoch_mse = epoch_acc = 0.0
#         with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
#             for batch_idx, batch in enumerate(train_loader):
#                 decoder_input = batch["decoder_input_discrete"].to(device)
#                 decoder_mask = batch["causal_mask"].to(device)
#                 images = batch["images"].to(device)
#                 labels = batch["encoded_labels"].to(device)
#                 target_tokens = batch["labels_discrete"].to(device)
#                 optimizer.zero_grad()

#                 predictions, image_emb, label_emb = model(decoder_input, decoder_mask, images, labels)
#                 ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

#                 # ===== Differentiable (soft) MSE =====
#                 logits_bins = predictions[..., BIN_OFFSET:BIN_OFFSET + NUM_BINS]
#                 probs_bins = F.softmax(logits_bins, dim=-1)
#                 if (not hasattr(binner, "_bin_centers_t")) or (binner._bin_centers_t is None) or (binner._bin_centers_t.device != predictions.device):
#                     binner._bin_centers_t = torch.tensor(binner.bin_centers, device=predictions.device, dtype=predictions.dtype)
#                 pred_cont_soft = (probs_bins * binner._bin_centers_t).sum(dim=-1)
#                 target_bin_rel = (target_tokens - BIN_OFFSET).clamp(0, NUM_BINS - 1)
#                 target_numeric_mask = ((target_tokens >= BIN_OFFSET) & (target_tokens < BIN_OFFSET + NUM_BINS))
#                 target_cont = binner._bin_centers_t[target_bin_rel]
#                 mse_mask = target_numeric_mask & (target_tokens != PAD_TOKEN)
#                 mse_loss_soft = F.mse_loss(pred_cont_soft[mse_mask], target_cont[mse_mask]) if mse_mask.any() else torch.tensor(0.0, device=device, requires_grad=True)

#                 # ===== Hard (argmax) MSE =====
#                 pred_tokens = predictions.argmax(dim=-1)
#                 pred_bin_rel = pred_tokens - BIN_OFFSET
#                 target_bin_rel = target_tokens - BIN_OFFSET
#                 target_numeric_mask = (target_bin_rel >= 0) & (target_bin_rel < NUM_BINS)
#                 nonpad_mask = (target_tokens != PAD_TOKEN)
#                 mse_mask_hard = (target_numeric_mask & nonpad_mask)
#                 if mse_mask_hard.sum() > 0:
#                     pred_cont_hard = binner.bin_to_value_torch(pred_bin_rel.clamp(0, NUM_BINS - 1)).to(device)
#                     target_cont_hard = binner.bin_to_value_torch(target_bin_rel.clamp(0, NUM_BINS - 1)).to(device)
#                     mse_loss_hard = F.mse_loss(pred_cont_hard[mse_mask_hard], target_cont_hard[mse_mask_hard])
#                 else:
#                     mse_loss_hard = torch.tensor(0.0, device=device)

#                 clip_loss = clip_loss_fn(image_emb, label_emb)
#                 total_loss = ce_loss + mse_weight * mse_loss_soft + clip_loss_weight * clip_loss
#                 total_loss.backward()
#                 optimizer.step()

#                 valid_mask = (target_tokens != PAD_TOKEN)
#                 batch_acc = (((pred_tokens == target_tokens) & valid_mask).sum().float() / (valid_mask.sum().float() + 1e-12))
#                 epoch_loss += total_loss.item()
#                 epoch_ce += ce_loss.item()
#                 epoch_mse += mse_loss_soft.item()
#                 epoch_acc += batch_acc.item()

#                 if rank == 0:
#                     wandb.log({
#                         "train/ce_loss": ce_loss.item(),
#                         "train/mse_loss_soft": mse_loss_soft.item(),
#                         "train/mse_loss_hard": mse_loss_hard.item(),
#                         "train/clip_loss": clip_loss.item(),
#                         "train/total_loss": total_loss.item(),
#                         "train/accuracy": batch_acc.item(),
#                         "epoch": epoch,
#                         "batch": epoch * len(train_loader) + batch_idx,
#                     })

#                 pbar.set_postfix({"Loss": total_loss.item(), "Acc": batch_acc.item()})
#                 pbar.update(1)

#         avg_train_loss = epoch_loss / len(train_loader)

#         # ---------------- Validation ----------------
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             with tqdm(total=len(val_loader), desc=f"Rank {rank} Val Epoch {epoch}", leave=False) as pbar:
#                 for batch_idx, batch in enumerate(val_loader):
#                     decoder_input = batch["decoder_input_discrete"].to(device)
#                     decoder_mask = batch["causal_mask"].to(device)
#                     images = batch["images"].to(device)
#                     labels = batch["encoded_labels"].to(device)
#                     target_tokens = batch["labels_discrete"].to(device)
#                     predictions, image_emb, label_emb = model(decoder_input, decoder_mask, images, labels)
#                     ce_loss = cross_entropy_loss(predictions, target_tokens, pad_token=PAD_TOKEN)

#                     logits_bins = predictions[..., BIN_OFFSET:BIN_OFFSET + NUM_BINS]
#                     probs_bins = F.softmax(logits_bins, dim=-1)
#                     pred_cont_soft = (probs_bins * binner._bin_centers_t).sum(dim=-1)
#                     target_bin_rel = (target_tokens - BIN_OFFSET).clamp(0, NUM_BINS - 1)
#                     target_numeric_mask = ((target_tokens >= BIN_OFFSET) & (target_tokens < BIN_OFFSET + NUM_BINS))
#                     target_cont = binner._bin_centers_t[target_bin_rel]
#                     mse_mask = target_numeric_mask & (target_tokens != PAD_TOKEN)
#                     mse_loss_soft = F.mse_loss(pred_cont_soft[mse_mask], target_cont[mse_mask]) if mse_mask.any() else torch.tensor(0.0, device=device)

#                     pred_tokens = predictions.argmax(dim=-1)
#                     pred_bin_rel = pred_tokens - BIN_OFFSET
#                     target_bin_rel = target_tokens - BIN_OFFSET
#                     mse_mask_hard = ((target_bin_rel >= 0) & (target_bin_rel < NUM_BINS)) & (target_tokens != PAD_TOKEN)
#                     mse_loss_hard = F.mse_loss(
#                         binner.bin_to_value_torch(pred_bin_rel.clamp(0, NUM_BINS - 1))[mse_mask_hard],
#                         binner.bin_to_value_torch(target_bin_rel.clamp(0, NUM_BINS - 1))[mse_mask_hard],
#                     ) if mse_mask_hard.sum() > 0 else torch.tensor(0.0, device=device)

#                     clip_loss = clip_loss_fn(image_emb, label_emb)
#                     total_loss = ce_loss + mse_weight * mse_loss_soft + clip_loss_weight * clip_loss
#                     val_loss += total_loss.item()

#                     if rank == 0:
#                         wandb.log({
#                             "val/ce_loss": ce_loss.item(),
#                             "val/mse_loss_soft": mse_loss_soft.item(),
#                             "val/mse_loss_hard": mse_loss_hard.item(),
#                             "val/clip_loss": clip_loss.item(),
#                             "val/total_loss": total_loss.item(),
#                             "epoch": epoch,
#                         })
#                     pbar.set_postfix({"Val Loss": total_loss.item()})
#                     pbar.update(1)

#         avg_val_loss = val_loss / len(val_loader)
#         if rank == 0:
#             print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
#             if avg_val_loss < best_loss:
#                 best_loss = avg_val_loss
#                 save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, clip_loss_fn, model_config)

#     cleanup_ddp()


# if __name__ == "__main__":
#     train(checkpoint_path=None, use_strict_resume=False)
