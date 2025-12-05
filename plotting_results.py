# =========================================================
# IMPORTS
# =========================================================
import os
import json
import requests

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset
from dataset_generation.curve_plot import get_pca_inclination, rotate_curve

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# =========================================================
# SIMULATOR CONFIG
# =========================================================
API_ENDPOINT = "http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)

# =========================================================
# MODEL / DATASET LOADING
# =========================================================
checkpoint_path = "./weights/CE_GAUS/LATENT_LLAMA_d768_h8_n6_bs512_lr0.0005_best.pth"
data_dir = "/home/anurizada/Documents/processed_dataset_17"
batch_size = 1

dataset = BarLinkageDataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint["model_config"]

model = LatentLLaMA_SingleToken(
    tgt_seq_len=model_config["tgt_seq_len"],
    d_model=model_config["d_model"],
    h=model_config["h"],
    N=model_config["N"],
    num_labels=model_config["num_labels"],
    vocab_size=model_config["vocab_size"],
    latent_dim=model_config["latent_dim"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================================================
# CONFIG
# =========================================================
NUM_NEIGHBORS = 20            # hard-coded kNN neighbors
NUM_MECH_TYPES = 17
SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2
NUM_SPECIAL_TOKENS = 3
BIN_OFFSET = NUM_SPECIAL_TOKENS
NUM_BINS = 201
LATENT_DIM = 50
tgt_seq_len = model_config["tgt_seq_len"]

FIXED_TEMPERATURE = 0.0       # greedy decoding
TOP_K = None                  # no top-k

latent_path = "/home/anurizada/Documents/processed_dataset_17/vae_mu.npy"
labels_cont_path = "/home/anurizada/Documents/processed_dataset_17/labels_continuous.npy"
encoded_labels_path = "/home/anurizada/Documents/processed_dataset_17/encoded_labels.npy"

label_mapping_path = "/home/anurizada/Documents/processed_dataset_17/label_mapping.json"
coupler_mapping_path = "/home/anurizada/Documents/transformer/BSIdict.json"

# Load numpy arrays once
LATENTS_ALL = np.load(latent_path)          # (N, latent_dim)
LABELS_CONT_ALL = np.load(labels_cont_path) # (N, ?)
ENCODED_LABELS_ALL = np.load(encoded_labels_path)  # (N,)
N_SAMPLES = LATENTS_ALL.shape[0]

# =========================================================
# Load Label + Coupler Mapping
# =========================================================
with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)
index_to_label = label_mapping["index_to_label"]

with open(coupler_mapping_path, "r") as f:
    coupler_mapping = json.load(f)


def coupler_index_for(mech_type: str) -> int:
    if mech_type in coupler_mapping and "c" in coupler_mapping[mech_type]:
        cvec = coupler_mapping[mech_type]["c"]
        if isinstance(cvec, list) and 1 in cvec:
            return cvec.index(1)
    return -1


def safe_name(name: str, max_len=30):
    return "".join([(c if c.isalnum() else "_") for c in name])[:max_len] or "unk"


# =========================================================
# CoordinateBinner
# =========================================================
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def bin_to_value_torch(self, idx):
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        centers = torch.tensor(self.bin_centers, device=idx.device, dtype=torch.float32)
        return centers[idx]


binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS - 1)


# =========================================================
# Causal Mask
# =========================================================
def build_causal_mask(seq_len, device):
    m = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)


# =========================================================
# Autoregressive Latent Prediction
# (we will use greedy: temperature=0, top_k=None)
# =========================================================
def predict_autoregressive_latent(
    model,
    latent,
    mech_idx,
    max_seq_len,
    device,
    temperature=1.0,
    top_k=None,
    eos_token=EOS_TOKEN,
    sos_token=SOS_TOKEN,
):
    model.eval()

    latent = latent.unsqueeze(0) if latent.dim() == 1 else latent
    latent = latent.to(device)

    mech_labels = torch.tensor([mech_idx], device=device, dtype=torch.long)
    decoder_input = torch.tensor([[sos_token]], device=device, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_seq_len):
            mask = build_causal_mask(decoder_input.size(1), device)
            logits = model(decoder_input, mask, latent, mech_labels)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            if top_k:
                k = min(top_k, probs.size(-1))
                topk_probs, topk_idx = torch.topk(probs, k=k)
                next_token = topk_idx.gather(-1, torch.multinomial(topk_probs, 1))
            elif temperature == 0:
                # greedy
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            token = int(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if token == eos_token:
                break

    return decoder_input.squeeze(0).cpu().numpy()


# =========================================================
# Clean joint label rows (for GT)
# =========================================================
def clean_and_reshape_label(row):
    mask = (row != 2.0) & (row != -1.0)
    row = row[mask]
    if row.size % 2:
        row = row[:-1]
    return row.reshape(-1, 2) if row.size else np.zeros((0, 2))


# =========================================================
# Simulate curve through API
# =========================================================
def simulate_curve(params, mech_type):
    ex = {
        "params": params.tolist(),
        "type": mech_type,
        "speedScale": speedscale,
        "steps": steps,
        "relativeTolerance": 0.1,
    }

    try:
        r = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps([ex])).json()
        if isinstance(r, list) and "poses" in r[0]:
            return np.array(r[0]["poses"])
    except Exception:
        return None

    return None


# =========================================================
# Joint connections from B (skip first link)
# =========================================================
def get_joint_connections(mech_name: str):
    """
    Return list of (a,b) joint indices to connect with bars,
    skipping the first link (first row of B).
    """
    if mech_name not in coupler_mapping:
        return []

    mech_entry = coupler_mapping[mech_name]
    if "B" not in mech_entry:
        return []

    B = mech_entry["B"]
    connections = []

    # Skip first row
    for row in B[1:]:
        joints = [j for j, val in enumerate(row) if val == 1]
        if len(joints) >= 2:
            for a in range(len(joints)):
                for b in range(a + 1, len(joints)):
                    connections.append((joints[a], joints[b]))

    return connections


# =========================================================
# Predict and ALIGN coupler + joints to GT coupler
# =========================================================
def predict_and_align_for_latent(
    latent_vec,
    mech_idx,
    mech_name,
    orig_phi,
    orig_denom,
    ox_mean,
    oy_mean,
):
    """
    latent_vec: np.array (latent_dim,)
    Returns:
        aligned_curve: (T, 2) coupler curve aligned to GT
        aligned_joints: (J, 2) joints aligned to GT
    or (None, None) if something fails.
    """
    latent_t = torch.tensor(latent_vec, dtype=torch.float32, device=device)

    # Predict tokens
    pred_tokens = predict_autoregressive_latent(
        model,
        latent_t,
        mech_idx,
        tgt_seq_len,
        device,
        temperature=FIXED_TEMPERATURE,
        top_k=TOP_K,
    )

    coord_tokens = [t for t in pred_tokens if t >= BIN_OFFSET]
    if len(coord_tokens) < 4:
        return None, None

    coords = binner.bin_to_value_torch(
        torch.tensor(coord_tokens, device=device) - BIN_OFFSET
    ).cpu().numpy()

    if coords.size % 2:
        coords = coords[:-1]

    pred_points = coords.reshape(-1, 2)

    # Simulate predicted mechanism
    Pp = simulate_curve(pred_points, mech_name)
    if Pp is None or Pp.shape[0] < minsteps:
        return None, None

    coup_idx_pred = coupler_index_for(mech_name)
    if coup_idx_pred < 0:
        return None, None

    gen_x = Pp[:, coup_idx_pred, 0]
    gen_y = Pp[:, coup_idx_pred, 1]

    # Align predicted coupler to GT coupler
    gen_phi = -get_pca_inclination(gen_x, gen_y)
    rotation = gen_phi - orig_phi
    gen_x, gen_y = rotate_curve(gen_x, gen_y, rotation)

    gen_denom = np.sqrt(np.var(gen_x) + np.var(gen_y)) + 1e-8
    if gen_denom < 1e-12:
        return None, None

    scale = orig_denom / gen_denom
    gen_x *= scale
    gen_y *= scale

    # translate to GT center
    gen_x -= (np.mean(gen_x) - ox_mean)
    gen_y -= (np.mean(gen_y) - oy_mean)

    aligned_curve = np.column_stack([gen_x, gen_y])

    # Align joints by same transform
    px = pred_points[:, 0]
    py = pred_points[:, 1]
    px, py = rotate_curve(px, py, rotation)
    px *= scale
    py *= scale
    px -= (np.mean(px) - ox_mean)
    py -= (np.mean(py) - oy_mean)
    aligned_joints = np.column_stack([px, py])

    return aligned_curve, aligned_joints


# =========================================================
# kNN setup
# =========================================================
knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1)
knn.fit(LATENTS_ALL)


# =========================================================
# PLOTTING HELPERS
# =========================================================
def plot_prediction(
    gt_x,
    gt_y,
    curve,
    joints,
    mech_name,
    title,
    save_path,
    color_curve="g",
):
    """
    Plot GT coupler + one predicted coupler + joints + linkage.
    GT joints are NOT plotted (only predicted joints).
    """
    plt.figure(figsize=(6, 6))

    # GT coupler curve
    plt.plot(gt_x, gt_y, "r-", linewidth=2, label="GT coupler")

    # predicted coupler
    if curve is not None:
        plt.plot(curve[:, 0], curve[:, 1], color_curve + "--", linewidth=2, label="Pred coupler")

    # predicted joints + linkage
    if joints is not None and joints.shape[0] > 0:
        for j, (xj, yj) in enumerate(joints):
            plt.scatter(xj, yj, c=color_curve, s=40)
            plt.text(xj + 0.003, yj + 0.003, f"{j}", fontsize=8, color=color_curve)

        connections = get_joint_connections(mech_name)
        for a, b in connections:
            if a < joints.shape[0] and b < joints.shape[0]:
                x1, y1 = joints[a]
                x2, y2 = joints[b]
                plt.plot([x1, x2], [y1, y2], color_curve + "-", linewidth=1)

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# PROCESS ONE QUERY CURVE
# =========================================================
def process_one_query(base_dir="results_knn"):

    # --------------------------
    # Pick random query index
    # --------------------------
    query_idx = np.random.randint(0, N_SAMPLES)
    query_latent = LATENTS_ALL[query_idx]
    gt_mech_idx = int(ENCODED_LABELS_ALL[query_idx])
    gt_mech_name = index_to_label[str(gt_mech_idx)]

    # ground truth mechanism params
    gt_points = clean_and_reshape_label(LABELS_CONT_ALL[query_idx])
    if gt_points.shape[0] == 0:
        print(f"[{query_idx}] No valid GT points, skipping.")
        return

    # simulate GT coupler
    P_gt = simulate_curve(gt_points, gt_mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        print(f"[{query_idx}] GT simulation failed, skipping.")
        return

    coup_idx_gt = coupler_index_for(gt_mech_name)
    if coup_idx_gt < 0:
        print(f"[{query_idx}] No coupler index for GT mech, skipping.")
        return

    gx, gy = P_gt[:, coup_idx_gt, 0], P_gt[:, coup_idx_gt, 1]
    gt_x, gt_y = gx.copy(), gy.copy()

    # alignment reference from GT coupler
    orig_phi = -get_pca_inclination(gt_x, gt_y)
    orig_denom = np.sqrt(np.var(gt_x) + np.var(gt_y)) + 1e-8
    ox_mean, oy_mean = np.mean(gt_x), np.mean(gt_y)

    # sample directory (flat)
    sample_dir = os.path.join(base_dir, f"sample_{query_idx:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    # --------------------------
    # kNN neighbors
    # --------------------------
    _, idxs = knn.kneighbors(query_latent.reshape(1, -1))
    neighbor_idxs = idxs[0][1:]  # skip self

    print(f"Query idx={query_idx}, GT mech={gt_mech_name}, neighbors={neighbor_idxs.tolist()}")

    # --------------------------
    # For each mechanism type
    # --------------------------
    for mech_idx in range(NUM_MECH_TYPES):
        mech_name = index_to_label[str(mech_idx)]
        mech_safe = safe_name(mech_name)

        # ======================================================
        # ORIGINAL LATENT prediction
        # ======================================================
        curve_orig, joints_orig = predict_and_align_for_latent(
            query_latent,
            mech_idx,
            mech_name,
            orig_phi,
            orig_denom,
            ox_mean,
            oy_mean,
        )

        if curve_orig is not None and joints_orig is not None:
            save_path = os.path.join(sample_dir, f"{mech_safe}_original.png")
            title = f"Query latent — mech={mech_name}, idx={query_idx}"
            plot_prediction(
                gt_x,
                gt_y,
                curve_orig,
                joints_orig,
                mech_name,
                title,
                save_path,
                color_curve="b",
            )

        # ======================================================
        # NEIGHBOR LATENTS prediction
        # ======================================================
        for nidx in neighbor_idxs:
            neigh_latent = LATENTS_ALL[nidx]

            curve_n, joints_n = predict_and_align_for_latent(
                neigh_latent,
                mech_idx,
                mech_name,
                orig_phi,
                orig_denom,
                ox_mean,
                oy_mean,
            )

            if curve_n is None or joints_n is None:
                continue

            save_path = os.path.join(sample_dir, f"{mech_safe}_neighbor_{int(nidx)}.png")
            title = f"kNN latent — mech={mech_name}, query={query_idx}, neigh={int(nidx)}"
            plot_prediction(
                gt_x,
                gt_y,
                curve_n,
                joints_n,
                mech_name,
                title,
                save_path,
                color_curve="g",
            )


# =========================================================
# MAIN
# =========================================================
def main():
    NUM_QUERIES = 100  # how many random queries to visualize

    base_dir = "results_knn"
    os.makedirs(base_dir, exist_ok=True)

    for _ in range(NUM_QUERIES):
        process_one_query(base_dir=base_dir)

    print("DONE.")


if __name__ == "__main__":
    main()
