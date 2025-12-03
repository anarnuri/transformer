# =========================================================
# IMPORTS
# =========================================================
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset 
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision('medium')
from sklearn.neighbors import NearestNeighbors

from dataset_generation.curve_plot import get_pca_inclination, rotate_curve
import scipy.spatial.distance as sciDist
from tqdm import tqdm
import requests
import time
import matplotlib.pyplot as plt
import os
import json
import torch.nn.functional as F
from tslearn.metrics import dtw_path

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Headless simulator version
index = 0 # local server index 
API_ENDPOINT = f"http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps*20/360)

checkpoint_path = "./weights/CE_GAUS/LATENT_LLAMA_d768_h8_n6_bs512_lr0.0005_best.pth"
data_dir = "/home/anurizada/Documents/processed_dataset_17"
batch_size = 1

dataset = BarLinkageDataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint['model_config']

# Initialize model
model = LatentLLaMA_SingleToken(
    tgt_seq_len=model_config['tgt_seq_len'],
    d_model=model_config['d_model'],
    h=model_config['h'],
    N=model_config['N'],
    num_labels=model_config['num_labels'],
    vocab_size=model_config['vocab_size'],
    latent_dim=model_config['latent_dim']).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# =========================================================
# CONFIG
# =========================================================
NUM_NEIGHBORS = 10
NUM_MECH_TYPES = 17
SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2
NUM_SPECIAL_TOKENS = 3
BIN_OFFSET = NUM_SPECIAL_TOKENS
NUM_BINS = 201
LATENT_DIM = 50
tgt_seq_len = 17

FIXED_TEMPERATURE = 0.0
TOP_K = None

latent_path = "/home/anurizada/Documents/processed_dataset_17/vae_mu.npy"
labels_cont_path = "/home/anurizada/Documents/processed_dataset_17/labels_continuous.npy"
encoded_labels_path = "/home/anurizada/Documents/processed_dataset_17/encoded_labels.npy"

label_mapping_path = "/home/anurizada/Documents/processed_dataset_17/label_mapping.json"
coupler_mapping_path = "/home/anurizada/Documents/transformer/BSIdict.json"

# GLOBAL statistics
GLOBAL_DTW_VALUES = []
GLOBAL_DTW_BELOW_2 = 0   # <---- NEW


# =========================================================
# Load Label Mapping
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
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            token = int(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if token == eos_token:
                break

    return decoder_input.squeeze(0).cpu().numpy()


# =========================================================
# Normalization
# =========================================================
def get_pca_inclination(x, y):
    cx, cy = np.mean(x), np.mean(y)
    cov = np.cov(x - cx, y - cy)
    eigvals, eigvecs = np.linalg.eig(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(major[1], major[0])


def rotate_curve(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return x * c - y * s, x * s + y * c


def normalize_curve(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.var(x) + np.var(y))
    x /= denom
    y /= denom
    phi = -get_pca_inclination(x, y)
    return rotate_curve(x, y, phi)


# =========================================================
# Clean joint label rows
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
    except:
        return None

    return None


# =========================================================
# Compute best DTW among 6 transformations
# =========================================================
def best_variant_dtw(gt_curve, pred_curve):
    global GLOBAL_DTW_BELOW_2

    variants = [
        pred_curve,
        pred_curve[::-1],
        np.column_stack([ pred_curve[:,0], -pred_curve[:,1] ]),
        np.column_stack([ pred_curve[::-1,0], -pred_curve[::-1,1] ]),
        np.column_stack([ -pred_curve[:,0], pred_curve[:,1] ]),
        np.column_stack([ -pred_curve[::-1,0], pred_curve[::-1,1] ])
    ]

    best = 1e18

    for V in variants:
        path, dist = dtw_path(gt_curve, V)
        scale = np.sqrt(np.var(gt_curve) + np.var(V)) + 1e-12
        dtw_val = dist / (len(path) * scale)

        if dtw_val < best:
            best = dtw_val

    best *= 100.0

    # Track DTW < 2
    if best < 2.0:
        GLOBAL_DTW_BELOW_2 += 1

    return best


# =========================================================
# PROCESS ONE CURVE
# =========================================================
def process_one_curve():

    latents = np.load(latent_path)
    labels_cont = np.load(labels_cont_path)
    encoded_labels = np.load(encoded_labels_path)
    N = latents.shape[0]

    query_idx = np.random.randint(0, N)
    query_latent = latents[query_idx : query_idx + 1]

    # ground truth
    gt_points = clean_and_reshape_label(labels_cont[query_idx])
    gt_mech_idx = int(encoded_labels[query_idx])
    gt_mech_name = index_to_label[str(gt_mech_idx)]

    P_gt = simulate_curve(gt_points, gt_mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        return

    coup_idx_gt = coupler_index_for(gt_mech_name)
    if coup_idx_gt < 0:
        return

    gx, gy = P_gt[:, coup_idx_gt, 0], P_gt[:, coup_idx_gt, 1]
    gt_x, gt_y = normalize_curve(gx, gy)
    gt_curve = np.column_stack([gt_x, gt_y])

    orig_latent_tensor = torch.tensor(query_latent[0], dtype=torch.float32, device=device)

    # -----------------------------------------------
    # ORIGINAL LATENT
    # -----------------------------------------------
    for mech_idx in range(NUM_MECH_TYPES):

        mech_name = index_to_label[str(mech_idx)]

        pred_tokens = predict_autoregressive_latent(
            model, orig_latent_tensor, mech_idx,
            tgt_seq_len, device,
            temperature=FIXED_TEMPERATURE, top_k=TOP_K
        )

        coord_tokens = [t for t in pred_tokens if t >= BIN_OFFSET]
        if len(coord_tokens) < 4:
            continue

        coords = binner.bin_to_value_torch(
            torch.tensor(coord_tokens, device=device) - BIN_OFFSET
        ).cpu().numpy()

        if coords.size % 2:
            coords = coords[:-1]

        pred_points = coords.reshape(-1, 2)

        # simulate predicted curve
        Pp = simulate_curve(pred_points, mech_name)
        if Pp is None or Pp.shape[0] < minsteps:
            continue

        coup_idx_pred = coupler_index_for(mech_name)
        if coup_idx_pred < 0:
            continue

        px, py = Pp[:, coup_idx_pred, 0], Pp[:, coup_idx_pred, 1]
        px_n, py_n = normalize_curve(px, py)
        pred_curve = np.column_stack([px_n, py_n])

        best_dtw = best_variant_dtw(gt_curve, pred_curve)
        GLOBAL_DTW_VALUES.append(best_dtw)

    # -----------------------------------------------
    # NEIGHBOR LATENTS
    # -----------------------------------------------
    full_latents = np.load(latent_path)
    knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1)
    knn.fit(full_latents)
    _, idxs = knn.kneighbors(query_latent)

    neighbor_idxs = idxs[0][1:]

    for ng_idx in neighbor_idxs:
        neigh_latent = torch.tensor(full_latents[ng_idx], dtype=torch.float32, device=device)

        for mech_idx in range(NUM_MECH_TYPES):

            mech_name = index_to_label[str(mech_idx)]

            pred_tokens = predict_autoregressive_latent(
                model, neigh_latent, mech_idx,
                tgt_seq_len, device,
                temperature=FIXED_TEMPERATURE, top_k=TOP_K
            )

            coord_tokens = [t for t in pred_tokens if t >= BIN_OFFSET]
            if len(coord_tokens) < 4:
                continue

            coords = binner.bin_to_value_torch(
                torch.tensor(coord_tokens, device=device) - BIN_OFFSET
            ).cpu().numpy()

            if coords.size % 2:
                coords = coords[:-1]

            pred_points = coords.reshape(-1, 2)

            Pp = simulate_curve(pred_points, mech_name)
            if Pp is None or Pp.shape[0] < minsteps:
                continue

            coup_idx_pred = coupler_index_for(mech_name)
            if coup_idx_pred < 0:
                continue

            px, py = Pp[:, coup_idx_pred, 0], Pp[:, coup_idx_pred, 1]
            px_n, py_n = normalize_curve(px, py)
            pred_curve = np.column_stack([px_n, py_n])

            best_dtw = best_variant_dtw(gt_curve, pred_curve)
            GLOBAL_DTW_VALUES.append(best_dtw)


# =========================================================
# MAIN
# =========================================================
def main():

    NUM_SAMPLES = 2  # adjust as needed (e.g., 100)

    for _ in range(NUM_SAMPLES):
        process_one_curve()

    if GLOBAL_DTW_VALUES:
        avg_dtw = np.mean(GLOBAL_DTW_VALUES)
        total = len(GLOBAL_DTW_VALUES)
        below2 = GLOBAL_DTW_BELOW_2
        percentage = below2 / total * 100

        print("\n==============================================")
        print(f"Total Predictions Evaluated: {total}")
        print(f"Average Best-DTW: {avg_dtw:.6f}")
        print(f"DTW < 2.0 Count: {below2}")
        print(f"DTW < 2.0 Percentage: {percentage:.2f}%")
        print("==============================================")

        with open("final_dtw_stats.txt", "w") as f:
            f.write(f"Total Predictions: {total}\n")
            f.write(f"Average DTW: {avg_dtw}\n")
            f.write(f"DTW < 2 count: {below2}\n")
            f.write(f"Percent < 2: {percentage}\n")

    else:
        print("No DTW values computed.")


if __name__ == "__main__":
    main()
