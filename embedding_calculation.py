import torch
import torch.nn.functional as F

from vit_llama_latent_model import LatentLLaMA_Decoder_Simple  # <-- make sure this points to your file

# =========================================================
# CONFIG
# =========================================================
CHECKPOINT_PATH = "./weights/latent_d512_h8_n6_bs512_lr0.0001_vit_latent_llama.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD MODEL
# =========================================================
print(f"🔹 Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model_config = ckpt["model_config"]

# Some checkpoints may not have latent_seq_len explicitly stored
latent_seq_len = model_config.get("latent_seq_len", 50)
latent_dim = model_config.get("latent_dim", 50)  # if you store flat 50-D and reshape

model = LatentLLaMA_Decoder_Simple(
    latent_seq_len=latent_seq_len,
    latent_dim=latent_dim if latent_dim > 1 else 1,
    tgt_seq_len=model_config["tgt_seq_len"],
    vocab_size=model_config["vocab_size"],
    d_model=model_config["d_model"],
    h=model_config["h"],
    N=model_config["N"],
    num_labels=model_config["num_labels"],
    dropout=model_config.get("dropout", 0.1),
    pad_token_id=model_config.get("pad_token_id", 2),
    debug=False,
).to(DEVICE)

missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print("✅ Model loaded.")
if missing:
    print(f"⚠️ Missing keys: {len(missing)} (showing first 10): {missing[:10]}")
if unexpected:
    print(f"⚠️ Unexpected keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")

model.eval()
print()


# =========================================================
# BUILD DUMMY INPUTS
# =========================================================
B = 32
tgt_seq_len = model.tgt_seq_len
vocab_size = model.vocab_size
num_labels = model.num_labels

# Dummy latents: shape (B, latent_seq_len, latent_dim)
if latent_dim == 1:
    latents = torch.randn(B, latent_seq_len, 1, device=DEVICE)
else:
    # If you treat latent as flat (B, 50), reshape to (B, 50, 1) or (B, L, D)
    latents = torch.randn(B, latent_seq_len, latent_dim, device=DEVICE)

# Dummy tokens & mech labels
decoder_input_ids = torch.randint(0, vocab_size, (B, tgt_seq_len), device=DEVICE)
mech_labels_batch = torch.randint(0, num_labels, (B,), device=DEVICE)
mech_all = torch.arange(0, num_labels, device=DEVICE)


# =========================================================
# FORWARD THROUGH EMBEDDING PATHS
# =========================================================
with torch.no_grad():
    # 1) Latent tokens after projector
    latent_tokens = model.encode_latent(latents)           # (B, latent_seq_len, D)
    latent_mean = latent_tokens.mean().item()
    latent_std = latent_tokens.std().item()

    # 2) Token embeddings (tgt_embed)
    token_embs = model.tgt_embed.embedding(decoder_input_ids)  # (B, T, D)
    token_mean = token_embs.mean().item()
    token_std = token_embs.std().item()

    # 3) Mech embeddings
    mech_embs = model.mech_embedding(mech_all)             # (num_labels, D)
    mech_mean = mech_embs.mean().item()
    mech_std = mech_embs.std().item()

print("📊 Embedding statistics:")
print(f"  Latent tokens: mean = {latent_mean:.4f}, std = {latent_std:.4f}")
print(f"  Token embeddings: mean = {token_mean:.4f}, std = {token_std:.4f}")
print(f"  Mech embeddings:  mean = {mech_mean:.4f}, std = {mech_std:.4f}")
print("\n👉 If latent std is much smaller or larger than token/mech std, you have a scale imbalance.\n")


# =========================================================
# COSINE SIMILARITY BETWEEN AVERAGE DIRECTIONS
# =========================================================
with torch.no_grad():
    latent_vec = latent_tokens.mean(dim=(0, 1))   # (D,)
    token_vec = token_embs.mean(dim=(0, 1))       # (D,)
    mech_vec = mech_embs.mean(dim=0)              # (D,)

    sim_lat_tok = F.cosine_similarity(latent_vec, token_vec, dim=0).item()
    sim_lat_mech = F.cosine_similarity(latent_vec, mech_vec, dim=0).item()
    sim_tok_mech = F.cosine_similarity(token_vec, mech_vec, dim=0).item()

print("🔍 Mean cosine similarities:")
print(f"  Latent ↔ Token: {sim_lat_tok:.4f}")
print(f"  Latent ↔ Mech:  {sim_lat_mech:.4f}")
print(f"  Token  ↔ Mech:  {sim_tok_mech:.4f}")
print(
    "\n➡️ If Latent↔Token and Latent↔Mech are near 0 while Token↔Mech is > 0.3 or so,\n"
    "   your latent space is misaligned with the token/mech embedding spaces."
)
