# =========================================================
# IMPORTS
# =========================================================
import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from dataset import BarLinkageDataset  # your dataset class


# =========================================================
# CONFIG
# =========================================================
data_dir = "/home/anurizada/Documents/processed_dataset_17"

labels_cont_path   = os.path.join(data_dir, "labels_continuous.npy")
encoded_labels_path = os.path.join(data_dir, "encoded_labels.npy")
label_mapping_path  = os.path.join(data_dir, "label_mapping.json")
coupler_mapping_path = "/home/anurizada/Documents/transformer/BSIdict.json"

OUTPUT_DIR = "GT_VIS"   # where image_i.png / curve_i.png will be saved

API_ENDPOINT = "http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)


# =========================================================
# LOAD LABEL / MAPPING FILES
# =========================================================
labels_cont   = np.load(labels_cont_path)
encoded_labels = np.load(encoded_labels_path)

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


# =========================================================
# BASIC HELPERS
# =========================================================
def clean_and_reshape_label(row):
    """Filter out sentinel values (2.0, -1.0) and reshape to (N,2)."""
    mask = (row != 2.0) & (row != -1.0)
    filtered = row[mask]
    if filtered.size % 2:
        filtered = filtered[:-1]
    return filtered.reshape(-1, 2) if filtered.size else np.zeros((0, 2))


def get_pca_inclination(qx, qy):
    """Performs PCA and returns inclination of major principal axis."""
    cx, cy = np.mean(qx), np.mean(qy)
    covar_xx = np.mean((qx - cx)**2)
    covar_xy = np.mean((qx - cx)*(qy - cy))
    covar_yy = np.mean((qy - cy)**2)

    cov = np.array([[covar_xx, covar_xy],
                    [covar_xy, covar_yy]])

    eigvals, eigvecs = np.linalg.eig(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(major[1], major[0])


def rotate_curve(x, y, theta):
    return x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) + y*np.cos(theta)


def compute_norm_params(x, y):
    """Compute mean, scale, rotation angle."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    x0, y0 = x - mean_x, y - mean_y
    denom = np.sqrt(np.var(x0) + np.var(y0)) + 1e-8
    phi = -get_pca_inclination(x0, y0)
    return mean_x, mean_y, denom, phi


def apply_norm(x, y, mean_x, mean_y, denom, phi):
    x0 = (x - mean_x) / denom
    y0 = (y - mean_y) / denom
    return rotate_curve(x0, y0, phi)


# =========================================================
# SIMULATION WRAPPER
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
    except Exception as e:
        print(f"[simulate_curve] Error: {e}")

    return None


# =========================================================
# IMAGE SAVERS
# =========================================================
def save_raw_image(img, save_path):
    """Correct grayscale saving (force cmap='gray')."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    # Convert CHW → HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # If grayscale (H×W), keep it single-channel
    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Float → uint8
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")  # FIX: correct grayscale saving
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_curve_image(curve, save_path, title="Coupler Curve"):
    """Save only the simulated curve (no joints)."""
    plt.figure(figsize=(5, 5))
    plt.scatter(curve[:, 0], curve[:, 1], c="blue", s=10)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# MAIN LOOP
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = BarLinkageDataset(data_dir=data_dir)
    print(f"Loaded dataset with {len(dataset)} samples.")

    assert len(dataset) == labels_cont.shape[0] == encoded_labels.shape[0], \
        "Dataset length and label arrays must match."

    for idx in range(len(dataset)):
        sample = dataset[idx]
        img = sample["images"]   # your dataset key

        gt_points = clean_and_reshape_label(labels_cont[idx])
        mech_idx = int(encoded_labels[idx])
        mech_name = index_to_label[str(mech_idx)]

        if gt_points.size == 0:
            print(f"[{idx}] No valid joints, skipped.")
            continue

        P_gt = simulate_curve(gt_points, mech_name)
        if P_gt is None or P_gt.shape[0] < minsteps:
            print(f"[{idx}] Simulation failed, skipped.")
            continue

        coup_idx = coupler_index_for(mech_name)
        if coup_idx < 0:
            print(f"[{idx}] No coupler index, skipped.")
            continue

        x = P_gt[:, coup_idx, 0]
        y = P_gt[:, coup_idx, 1]

        mean_x, mean_y, denom, phi = compute_norm_params(x, y)
        xn, yn = apply_norm(x, y, mean_x, mean_y, denom, phi)
        curve = np.column_stack([xn, yn])

        img_path = os.path.join(OUTPUT_DIR, f"image_{idx+1}.png")
        curve_path = os.path.join(OUTPUT_DIR, f"curve_{idx+1}.png")

        save_raw_image(img, img_path)
        save_curve_image(curve, curve_path, title=f"Curve {idx+1} — {mech_name}")

        print(f"[OK] Saved: image_{idx+1}.png and curve_{idx+1}.png")

    print("\n✔ Done. All outputs saved to", OUTPUT_DIR)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
