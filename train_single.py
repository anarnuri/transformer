import torch
import os
from torch.utils.data import random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import torch.nn as nn
import numpy as np

from model import SingleImageTransformer  # Using the new model

def cross_entropy_loss(predictions, targets, pad_token=2, ignore_index=-100):
    """
    Cross entropy loss for discrete token prediction
    predictions: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len) with token indices
    """
    # Create mask for padding tokens
    mask = targets != pad_token
    # Flatten for cross entropy
    predictions_flat = predictions.view(-1, predictions.size(-1))
    targets_flat = targets.view(-1)
    
    # Set padding tokens to ignore_index
    targets_flat[~mask.view(-1)] = ignore_index
    
    return F.cross_entropy(predictions_flat, targets_flat, ignore_index=ignore_index)

def accuracy(predictions, targets, pad_token=2):
    """
    Calculate accuracy for non-padding tokens
    predictions: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)
    """
    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=-1)
    
    # Create mask for non-padding tokens
    mask = targets != pad_token
    
    # Calculate accuracy only on non-padding positions
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy

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
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'clip_loss_state_dict': clip_loss_fn.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'batch_size': batch_size,
        'learning_rate': lr,
        'model_config': model_config,
        'vocab_size': model_config['vocab_size']
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} with loss {best_loss:.6f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_gradient_flow(model, epoch):
    """Log detailed gradient flow information"""
    total_grad_norm = 0.0
    layer_grads = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            layer_grads[name] = grad_norm
            total_grad_norm += grad_norm ** 2
            
            wandb.log({
                f"grad/{name}": grad_norm,
            })
    
    total_grad_norm = total_grad_norm ** 0.5
    wandb.log({
        f"grad/total": total_grad_norm,
        "epoch": epoch
    })
    
    return layer_grads

def log_learning_rates(optimizer, epoch, prefix="train"):
    """Log learning rates for all parameter groups"""
    for i, group in enumerate(optimizer.param_groups):
        wandb.log({
            f"{prefix}/lr/group_{i}": group['lr'],
            "epoch": epoch
        })

def train():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    device = torch.device(f'cuda:{local_rank}')
    print(f"Using device: {device}")

    setup_ddp()
    torch.set_float32_matmul_precision('medium')

    # Hyperparameters
    batch_size = 64  # Reduced for discrete prediction
    num_epochs = 200
    lr = 1e-4  # Lower learning rate for classification
    clip_loss_weight = 0.5  # Reduced contrastive loss weight
    seq_len = 25  # Maximum sequence length

    # Load Dataset - You'll need to create a new dataset class for the .npy files
    # dataset = YourNewDataset(data_dir='/path/to/processed_dataset')
    # For now, using a placeholder
    from dataset import BarLinkageDataset  # You need to create this
    
    dataset = BarLinkageDataset(data_dir='/home/anurizada/Documents/processed_dataset')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    # Get vocabulary size from dataset or config
    vocab_size = 203  # NUM_SPECIAL_TOKENS (3) + num_bins (200)
    num_labels = len(dataset.label_mapping['label_to_index'])  # Number of classes

    # Model Configuration
    model_config = {
        'tgt_seq_len': seq_len,
        'output_size': vocab_size,  # Now predicting vocabulary tokens
        'd_model': 512,
        'h': 8,
        'N': 6,
        'num_labels': num_labels,
        'vocab_size': vocab_size
    }

    # Initialize Model
    model = SingleImageTransformer(
        tgt_seq_len=model_config['tgt_seq_len'],
        # output_size=model_config['output_size'],
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

    # Initialize WandB
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
                "total_params": total_params
            }
        )

    optimizer = Adam([
        {'params': model.parameters()},
        {'params': clip_loss_fn.parameters()}
    ], lr=lr, weight_decay=1e-5)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Get batch data - adjust based on your new dataset
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask = batch["causal_mask"].to(device)
                images = batch["images"].to(device)
                labels = batch["encoded_labels"].to(device)
                target_tokens = batch["labels_discrete"].to(device)

                # Forward pass
                optimizer.zero_grad()
                predictions, image_emb, label_emb = model(decoder_input, decoder_mask, images, labels)

                # Calculate losses
                prediction_loss = cross_entropy_loss(predictions, target_tokens)
                batch_acc = accuracy(predictions, target_tokens)
                clip_loss = clip_loss_fn(image_emb, label_emb)
                total_loss = prediction_loss + clip_loss_weight * clip_loss

                # Backward pass and optimization
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Log gradients and learning rates
                if rank == 0:
                    log_gradient_flow(model, epoch)
                    log_learning_rates(optimizer, epoch)
                
                optimizer.step()
                epoch_loss += total_loss.item()
                epoch_acc += batch_acc.item()

                if rank == 0:
                    wandb.log({
                        "train/ce_loss": prediction_loss.item(),
                        "train/accuracy": batch_acc.item(),
                        "train/clip_loss": clip_loss.item(),
                        "train/clip_logit_scale": clip_loss_fn.logit_scale.exp().item(),
                        "train/total_loss": total_loss.item(),
                        "epoch": epoch,
                        "batch": epoch * len(train_loader) + batch_idx
                    })

                pbar.set_postfix({
                    "Loss": total_loss.item(),
                    "Acc": batch_acc.item()
                })
                pbar.update(1)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Rank {rank} Val Epoch {epoch}", leave=False) as pbar:
                for batch_idx, batch in enumerate(val_loader):
                    decoder_input = batch["decoder_input_discrete"].to(device)
                    decoder_mask = batch["causal_mask"].to(device)
                    images = batch["images"].to(device)
                    labels = batch["encoded_labels"].to(device)
                    target_tokens = batch["labels_discrete"].to(device)

                    # Forward pass
                    predictions, image_emb, label_emb = model(decoder_input, decoder_mask, images, labels)
                    
                    # Calculate losses
                    prediction_loss = cross_entropy_loss(predictions, target_tokens)
                    batch_acc = accuracy(predictions, target_tokens)
                    clip_loss = clip_loss_fn(image_emb, label_emb)
                    total_loss = prediction_loss + clip_loss_weight * clip_loss

                    val_loss += total_loss.item()
                    val_acc += batch_acc.item()

                    if rank == 0:
                        wandb.log({
                            "val/ce_loss": prediction_loss.item(),
                            "val/accuracy": batch_acc.item(),
                            "val/clip_loss": clip_loss.item(),
                            "val/total_loss": total_loss.item(),
                            "epoch": epoch,
                        })

                    pbar.set_postfix({
                        "Val Loss": total_loss.item(),
                        "Val Acc": batch_acc.item()
                    })
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

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
    train()