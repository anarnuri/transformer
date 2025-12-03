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


def safe_name(name: str, max_len: int = 30) -> str:
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
    decoder_input = torch.tensor([[sos_token]], dtype=torch.long, device=device)

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
                next_token = torch.multinomial(probs, 1)

            token = int(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            if token == eos_token:
                break

    return decoder_input.squeeze(0).cpu().numpy()


# =========================================================
# Normalization Helpers
# =========================================================
def get_pca_inclination(qx, qy, ax=None, label=''):
    """Performs PCA and returns inclination of major principal axis."""
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)

    covar = np.array([[covar_xx, covar_xy], [covar_yx, covar_yy]])
    eig_val, eig_vec = np.linalg.eig(covar)

    if eig_val[0] > eig_val[1]:
        phi = np.arctan2(eig_vec[1, 0], eig_vec[0, 0])
    else:
        phi = np.arctan2(eig_vec[1, 1], eig_vec[0, 1])

    return phi


def rotate_curve(x, y, theta):
    cpx = x * np.cos(theta) - y * np.sin(theta)
    cpy = x * np.sin(theta) + y * np.cos(theta)
    return cpx, cpy


def compute_norm_params(x, y):
    """Compute mean, scale, and rotation angle for normalization."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    x0, y0 = x - mean_x, y - mean_y
    denom = np.sqrt(np.var(x0) + np.var(y0)) + 1e-8
    phi = -get_pca_inclination(x0, y0)
    return mean_x, mean_y, denom, phi


def apply_norm(x, y, mean_x, mean_y, denom, phi):
    """Apply normalization with given parameters."""
    x0 = (x - mean_x) / denom
    y0 = (y - mean_y) / denom
    return rotate_curve(x0, y0, phi)


def normalize_curve(x, y):
    """Kept for compatibility; uses compute_norm_params/apply_norm."""
    mean_x, mean_y, denom, phi = compute_norm_params(x, y)
    return apply_norm(x, y, mean_x, mean_y, denom, phi)


# =========================================================
# Joint Plot Helper
# =========================================================
def plot_gt_with_joints(gt_curve, gt_joints, save_path, title="GT Coupler + Joints"):
    """
    gt_curve: (N,2)
    gt_joints: (M,2)
    """
    plt.figure(figsize=(6, 6))
    plt.plot(gt_curve[:, 0], gt_curve[:, 1], "r-", label="GT Coupler")

    if gt_joints is not None and gt_joints.size > 0:
        xs, ys = gt_joints[:, 0], gt_joints[:, 1]
        plt.scatter(xs, ys, c="blue", s=35, label="GT Joints")
        for i, (jx, jy) in enumerate(gt_joints):
            plt.text(jx, jy, f"{i+1}", fontsize=9, color="black")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def apply_variant_to_points(points, variant_name):
    """
    Apply the same DTW variant transform to joint points
    that was applied to the coupler curve.
    points: (M,2)
    variant_name: "orig", "orig_rev", "mirror_x", ...
    """
    if points is None or points.size == 0:
        return points

    pts = points.copy()

    if variant_name == "orig":
        return pts
    elif variant_name == "orig_rev":
        return pts[::-1]
    elif variant_name == "mirror_x":
        pts[:, 1] *= -1
        return pts
    elif variant_name == "mirror_x_rev":
        pts = pts[::-1]
        pts[:, 1] *= -1
        return pts
    elif variant_name == "mirror_y":
        pts[:, 0] *= -1
        return pts
    elif variant_name == "mirror_y_rev":
        pts = pts[::-1]
        pts[:, 0] *= -1
        return pts
    else:
        # Fallback: no transform
        return pts


# =========================================================
# DTW Variant Helpers (mirrors + reverse)
# =========================================================
def generate_curve_variants(curve):
    """
    Given curve of shape (N,2), produce mirror/reverse variants.
    Returns list of (name, variant_curve).
    """
    x = curve[:, 0]
    y = curve[:, 1]

    original = curve
    mirror_x = np.column_stack([x, -y])
    mirror_y = np.column_stack([-x, y])

    variants = [
        ("orig",         original),
        ("orig_rev",     original[::-1]),
        ("mirror_x",     mirror_x),
        ("mirror_x_rev", mirror_x[::-1]),
        ("mirror_y",     mirror_y),
        ("mirror_y_rev", mirror_y[::-1]),
    ]
    return variants


def compute_normalized_dtw(gt_curve, pred_curve):
    """
    Compute normalized DTW:
    dist / (len(path) * scale) * 100
    """
    path, dist = dtw_path(gt_curve, pred_curve)
    scale = np.sqrt(np.var(gt_curve) + np.var(pred_curve))
    scale = max(scale, 1e-8)
    dtw_distance = dist / (len(path) * scale) * 100.0
    return dtw_distance, path


def best_variant_dtw(gt_curve, pred_curve):
    """
    For a given pred_curve, evaluate DTW over:
    - original
    - reversed
    - mirrored across x (and reversed)
    - mirrored across y (and reversed)
    Return:
      best_dtw, best_variant_name, best_curve, best_path
    """
    variants = generate_curve_variants(pred_curve)

    best_name = None
    best_curve = None
    best_path = None
    best_dtw = float("inf")

    for name, variant in variants:
        dtw_val, path = compute_normalized_dtw(gt_curve, variant)
        if dtw_val < best_dtw:
            best_dtw = dtw_val
            best_name = name
            best_curve = variant
            best_path = path

    return best_dtw, best_name, best_curve, best_path


# =========================================================
# Joint Label Cleaning
# =========================================================
def clean_and_reshape_label(row):
    mask = (row != 2.0) & (row != -1.0)
    filtered = row[mask]
    if filtered.size % 2:
        filtered = filtered[:-1]
    return filtered.reshape(-1, 2) if filtered.size else np.zeros((0, 2))


# =========================================================
# Simulation Wrapper
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
        resp = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps([ex])).json()
        if isinstance(resp, list) and "poses" in resp[0]:
            return np.array(resp[0]["poses"])
    except:
        pass
    return None


# =========================================================
# PROCESS ONE CURVE
# =========================================================
def process_one_curve():

    latents = np.load(latent_path)
    labels_cont = np.load(labels_cont_path)
    encoded_labels = np.load(encoded_labels_path)
    N = latents.shape[0]

    query_idx = np.random.randint(0, N)
    query_latent = latents[query_idx: query_idx + 1]
    print(f"\nSelected query latent index = {query_idx}")

    MASTER_OUT = "ALL_RESULTS"
    os.makedirs(MASTER_OUT, exist_ok=True)

    curve_folder = os.path.join(MASTER_OUT, f"curve_{query_idx:05d}")
    if os.path.exists(curve_folder):
        shutil.rmtree(curve_folder)
    os.makedirs(curve_folder, exist_ok=True)

    # --- Ground Truth Simulation ---
    gt_points = clean_and_reshape_label(labels_cont[query_idx])
    gt_mech_idx = int(encoded_labels[query_idx])
    gt_mech_name = index_to_label[str(gt_mech_idx)]

    P_gt = simulate_curve(gt_points, gt_mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        print("GT simulation failed.")
        return

    coup_idx_gt = coupler_index_for(gt_mech_name)
    if coup_idx_gt < 0:
        print("No coupler index for GT.")
        return

    original_x = P_gt[:, coup_idx_gt, 0]
    original_y = P_gt[:, coup_idx_gt, 1]

    # Normalize GT coupler + joints using same transform
    mean_x_gt, mean_y_gt, denom_gt, phi_gt = compute_norm_params(original_x, original_y)
    gt_x, gt_y = apply_norm(original_x, original_y, mean_x_gt, mean_y_gt, denom_gt, phi_gt)
    gt_curve = np.column_stack([gt_x, gt_y])

    if gt_points.size > 0:
        jx_gt = gt_points[:, 0]
        jy_gt = gt_points[:, 1]
        jx_gt_n, jy_gt_n = apply_norm(jx_gt, jy_gt, mean_x_gt, mean_y_gt, denom_gt, phi_gt)
        gt_joints_n = np.column_stack([jx_gt_n, jy_gt_n])
    else:
        gt_joints_n = np.zeros((0, 2))

    # Save GT coupler + joints plot (with numbering)
    plot_gt_with_joints(
        gt_curve,
        gt_joints_n,
        save_path=os.path.join(curve_folder, "gt_coupler_and_joints.png"),
        title=f"GT {gt_mech_name} — Coupler & Joints"
    )

    orig_latent_tensor = torch.tensor(query_latent[0], dtype=torch.float32, device=device)

    # ===============================================
    # Evaluate ORIGINAL LATENT across all mechanisms
    # ===============================================
    for mech_idx in range(NUM_MECH_TYPES):
        mech_name = index_to_label[str(mech_idx)]
        mech_safe = safe_name(mech_name)

        pred_tokens = predict_autoregressive_latent(
            model, orig_latent_tensor, mech_idx,
            tgt_seq_len, device,
            temperature=FIXED_TEMPERATURE, top_k=TOP_K
        )

        coord_tokens = [t for t in pred_tokens if t >= BIN_OFFSET]
        if len(coord_tokens) < 4:
            continue

        coords_tensor = torch.tensor(coord_tokens, device=device) - BIN_OFFSET
        coords_float = binner.bin_to_value_torch(coords_tensor).cpu().numpy()
        if coords_float.size % 2:
            coords_float = coords_float[:-1]
        pred_points = coords_float.reshape(-1, 2)  # joint locations (unnormalized)

        Pp = simulate_curve(pred_points, mech_name)
        if Pp is None or Pp.shape[0] < minsteps:
            continue

        coup_idx_pred = coupler_index_for(mech_name)
        if coup_idx_pred < 0:
            continue

        px = Pp[:, coup_idx_pred, 0]
        py = Pp[:, coup_idx_pred, 1]

        # Normalize predicted coupler + joints using same transform
        mean_x_p, mean_y_p, denom_p, phi_p = compute_norm_params(px, py)
        px_n, py_n = apply_norm(px, py, mean_x_p, mean_y_p, denom_p, phi_p)
        pred_curve = np.column_stack([px_n, py_n])

        if pred_points.size > 0:
            jx_p = pred_points[:, 0]
            jy_p = pred_points[:, 1]
            jx_p_n, jy_p_n = apply_norm(jx_p, jy_p, mean_x_p, mean_y_p, denom_p, phi_p)
            pred_joints_n = np.column_stack([jx_p_n, jy_p_n])
        else:
            pred_joints_n = np.zeros((0, 2))

        # ==========================================
        #   T S L E A R N  D T W  with variants
        # ==========================================
        dtw_distance, best_name, pred_curve_best, path_best = best_variant_dtw(
            gt_curve, pred_curve
        )

        # Transform predicted joints with the same best variant
        pred_joints_best = apply_variant_to_points(pred_joints_n, best_name)

        print(f"DTW(Orig latent, {mech_name}) best={best_name} -> {dtw_distance:.3f}")

        # Plot: GT coupler + predicted coupler + predicted joints (numbered)
        plt.figure(figsize=(6, 6))
        plt.plot(gt_curve[:, 0], gt_curve[:, 1], "r-", label="GT Coupler")
        plt.plot(
            pred_curve_best[:, 0], pred_curve_best[:, 1],
            "b--", label=f"{mech_name} ({best_name})"
        )

        if pred_joints_best is not None and pred_joints_best.size > 0:
            xs = pred_joints_best[:, 0]
            ys = pred_joints_best[:, 1]
            plt.scatter(xs, ys, c="blue", s=35, label="Pred Joints")
            for i, (jx, jy) in enumerate(pred_joints_best):
                plt.text(jx, jy, f"{i+1}", fontsize=9, color="black")

        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                curve_folder,
                f"orig_latent_{mech_safe}_{best_name}_DTW_{dtw_distance:.3f}.png"
            )
        )
        plt.close()

    # ===============================================
    # Evaluate KNN NEIGHBORS
    # ===============================================
    knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1)
    knn.fit(latents)
    _, idxs = knn.kneighbors(query_latent)

    neighbor_idxs = idxs[0][1:]
    neighbor_names = [
        "first","second","third","fourth","fifth",
        "sixth","seventh","eighth","ninth","tenth"
    ]

    for k, ng_idx in enumerate(neighbor_idxs):
        neigh_name = neighbor_names[k]
        neigh_latent_tensor = torch.tensor(latents[ng_idx], dtype=torch.float32, device=device)

        for mech_idx in range(NUM_MECH_TYPES):
            mech_name = index_to_label[str(mech_idx)]
            mech_safe = safe_name(mech_name)

            pred_tokens = predict_autoregressive_latent(
                model, neigh_latent_tensor, mech_idx,
                tgt_seq_len, device,
                temperature=FIXED_TEMPERATURE, top_k=TOP_K
            )

            coord_tokens = [t for t in pred_tokens if t >= BIN_OFFSET]
            if len(coord_tokens) < 4:
                continue

            coords_tensor = torch.tensor(coord_tokens, device=device) - BIN_OFFSET
            coords_float = binner.bin_to_value_torch(coords_tensor).cpu().numpy()
            if coords_float.size % 2:
                coords_float = coords_float[:-1]
            pred_points = coords_float.reshape(-1, 2)  # joint locations (unnormalized)

            Pp = simulate_curve(pred_points, mech_name)
            if Pp is None or Pp.shape[0] < minsteps:
                continue

            coup_idx_pred = coupler_index_for(mech_name)
            if coup_idx_pred < 0:
                continue

            px = Pp[:, coup_idx_pred, 0]
            py = Pp[:, coup_idx_pred, 1]

            # Normalize predicted coupler + joints for neighbor
            mean_x_p, mean_y_p, denom_p, phi_p = compute_norm_params(px, py)
            px_n, py_n = apply_norm(px, py, mean_x_p, mean_y_p, denom_p, phi_p)
            pred_curve = np.column_stack([px_n, py_n])

            if pred_points.size > 0:
                jx_p = pred_points[:, 0]
                jy_p = pred_points[:, 1]
                jx_p_n, jy_p_n = apply_norm(jx_p, jy_p, mean_x_p, mean_y_p, denom_p, phi_p)
                pred_joints_n = np.column_stack([jx_p_n, jy_p_n])
            else:
                pred_joints_n = np.zeros((0, 2))

            # ==========================================
            #   T S L E A R N  D T W  with variants
            # ==========================================
            dtw_distance, best_name, pred_curve_best, path_best = best_variant_dtw(
                gt_curve, pred_curve
            )

            # Transform neighbor joints with the same best variant
            pred_joints_best = apply_variant_to_points(pred_joints_n, best_name)

            print(f"DTW({neigh_name}, {mech_name}) best={best_name} -> {dtw_distance:.3f}")

            # Plot GT coupler + neighbor predicted coupler + joints
            plt.figure(figsize=(6, 6))
            plt.plot(gt_curve[:, 0], gt_curve[:, 1], "r-", label="GT Coupler")
            plt.plot(
                pred_curve_best[:, 0], pred_curve_best[:, 1],
                "g--", label=f"{neigh_name} {mech_name} ({best_name})"
            )

            if pred_joints_best is not None and pred_joints_best.size > 0:
                xs = pred_joints_best[:, 0]
                ys = pred_joints_best[:, 1]
                plt.scatter(xs, ys, c="green", s=35, label="Pred Joints")
                for i, (jx, jy) in enumerate(pred_joints_best):
                    plt.text(jx, jy, f"{i+1}", fontsize=9, color="black")

            plt.axis("equal")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    curve_folder,
                    f"{neigh_name}_{mech_safe}_{best_name}_DTW_{dtw_distance:.3f}.png"
                )
            )
            plt.close()

    print(f"\nFinished curve {query_idx}")


# =========================================================
# MAIN LOOP
# =========================================================
def main():
    for _ in range(5):  # adjust number of samples as needed
        process_one_curve()


if __name__ == "__main__":
    main()
