import torch
from torch.utils.data import DataLoader
from stage_1_model import Stage1LLaMA     # <-- Your Stage-1 model
from dataset import BarLinkageDataset

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "/home/anurizada/Documents/processed_dataset_17"
CHECKPOINT = "./weights/STAGE_1_WEIGHTS/STAGE1_LLAMA_d512_h8_n9_bs512_lr0.001_best.pth"      # <- change to your path
BATCH_SIZE = 32
DEVICE = "cuda:0"

# ------------------------------
# Load dataset
# ------------------------------
dataset = BarLinkageDataset(DATA_DIR)

# just use validation-like split
val_len = int(0.2 * len(dataset))
train_len = len(dataset) - val_len

_, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# Load model (same config as training!)
# ------------------------------
model_config = {
    "tgt_seq_len": 17,
    "d_model": 512,
    "h": 8,
    "N": 9,
    "num_labels": 17,
    "vocab_size": 201 + 3,
    "dropout": 0.1,
    "pad_token_id": 2,
    "debug": False,
}

model = Stage1LLaMA(
    tgt_seq_len=model_config["tgt_seq_len"],
    d_model=model_config["d_model"],
    h=model_config["h"],
    N=model_config["N"],
    num_labels=model_config["num_labels"],
    vocab_size=model_config["vocab_size"],
    dropout=model_config["dropout"],
    pad_token_id=model_config["pad_token_id"],
    debug=model_config["debug"],
).to(DEVICE)

# ------------------------------
# Load checkpoint weights
# ------------------------------
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("✅ Loaded model + checkpoint")

# ------------------------------
# Inference loop
# ------------------------------
PAD = 2
BIN_OFFSET = 3        # your special tokens: [SOS, EOS, PAD]

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):

        dec_in   = batch["decoder_input_discrete"].to(DEVICE)
        dec_mask = batch["causal_mask"].to(DEVICE).bool()
        mech     = batch["encoded_labels"].to(DEVICE)
        targets  = batch["labels_discrete"].to(DEVICE)
        print(dec_in)
        print(targets)
        # forward pass
        logits = model(dec_in, dec_mask, mech)
        pred_tokens = logits.argmax(dim=-1)    # (B, T)

        for i in range(dec_in.size(0)):
            print("\n=============================")
            print(f"Sample {batch_idx * BATCH_SIZE + i}")

            print("Mech type:", mech[i].item())

            # convert tensors to lists
            gt = targets[i].cpu().numpy().tolist()
            pr = pred_tokens[i].cpu().numpy().tolist()

            print("GT tokens:  ", gt)
            print("Predicted:  ", pr)

            # stop after ~20 samples
            if i > 20:
                break
        break  # remove this if you want full dataset
