# =========================================================
# IMPORTS
# =========================================================
import torch
import numpy as np
import json
import requests
import torch.nn.functional as F
from tslearn.metrics import dtw_path
from llama_latent_model import LatentLLaMA_SingleToken

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# =========================================================
# USER-DEFINED TEMPERATURE (CHANGE THIS AND RERUN)
# =========================================================
TEMPERATURE = 2.0       # ← Set this manually for each experiment


# =========================================================
# FIXED SET OF SAMPLE INDEXES
# =========================================================
FIXED_SAMPLE_IDXS = [
    37454, 95071, 73199, 59866, 15601, 15599, 86617, 60111, 70807,  2058,
    96990, 83244, 21233, 18340, 30424, 52475, 43195, 29123, 61185, 13949,
    29214, 36636, 45607, 78518, 19967, 51423, 59241,  4699, 45537, 36636,
    78518, 10000,  2058, 65088,  4505, 94888, 96564, 80840,   363, 28326,
    88053, 28319, 42515, 15602, 28918, 49851, 48947, 27126, 12811,  9905,
    66155, 16090, 54841, 69189, 22489, 40311, 34850, 70641,  9073, 79883,
    30424, 52192, 14223, 43135, 80219, 30505, 16949,  8258, 13261, 66060,
    34208, 82191, 93950, 19663, 61748,  2622, 97242, 84922, 56565,  6208,
    38063, 17701, 75094, 96895, 59789, 32533, 84123, 94356, 52185, 55517,
    12788, 58083, 54886, 97562,  8312,  2229, 80744, 31943
]


# =========================================================
# CONFIG
# =========================================================
API_ENDPOINT = "http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)

checkpoint_path = "./weights/CE_GAUS/LATENT_LLAMA_d768_h8_n6_bs512_lr0.0005_best.pth"

latent_path = "/home/anurizada/Documents/processed_dataset_17/vae_mu.npy"
labels_cont_path = "/home/anurizada/Documents/processed_dataset_17/labels_continuous.npy"
encoded_labels_path = "/home/anurizada/Documents/processed_dataset_17/encoded_labels.npy"

label_mapping_path = "/home/anurizada/Documents/processed_dataset_17/label_mapping.json"
coupler_mapping_path = "/home/anurizada/Documents/transformer/BSIdict.json"

SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2
NUM_SPECIAL_TOKENS = 3
NUM_BINS = 201
BIN_OFFSET = NUM_SPECIAL_TOKENS
tgt_seq_len = 17

GLOBAL_DTW_VALUES = []
GLOBAL_DTW_BELOW_2 = 0


# =========================================================
# Load label mappings
# =========================================================
with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)
index_to_label = label_mapping["index_to_label"]

with open(coupler_mapping_path, "r") as f:
    coupler_mapping = json.load(f)

def coupler_index_for(mech):
    if mech in coupler_mapping and "c" in coupler_mapping[mech]:
        cvec = coupler_mapping[mech]["c"]
        if isinstance(cvec, list) and 1 in cvec:
            return cvec.index(1)
    return -1


# =========================================================
# Coordinate Binner
# =========================================================
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.centers = (edges[:-1] + edges[1:]) / 2

    def bin_to_value_torch(self, idx):
        idx = torch.clamp(idx, 0, len(self.centers)-1)
        centers_t = torch.tensor(self.centers, device=idx.device)
        return centers_t[idx]

binner = CoordinateBinner(num_bins=NUM_BINS - 1)


# =========================================================
# Causal Mask
# =========================================================
def build_causal_mask(n, device):
    m = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)


# =========================================================
# Autoregressive prediction with temperature
# =========================================================
def predict_autoregressive_latent(model, latent, mech_idx, max_len, temperature):
    latent = latent.unsqueeze(0)
    mech_labels = torch.tensor([mech_idx], device=device)
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    with torch.no_grad():
        for _ in range(max_len):
            mask = build_causal_mask(decoder_input.size(1), device)
            logits = model(decoder_input, mask, latent, mech_labels)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            if temperature == 0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, 1)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            if next_token.item() == EOS_TOKEN:
                break

    return decoder_input.squeeze(0).cpu().numpy()


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
# Normalize curve
# =========================================================
def get_pca_inclination(x, y):
    cx, cy = np.mean(x), np.mean(y)
    cov = np.cov(x-cx, y-cy)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(v[1], v[0])

def rotate_curve(x, y, t):
    c, s = np.cos(t), np.sin(t)
    return x*c - y*s, x*s + y*c

def normalize_curve(x, y):
    x -= np.mean(x)
    y -= np.mean(y)
    d = np.sqrt(np.var(x) + np.var(y))
    x /= d; y /= d
    phi = -get_pca_inclination(x, y)
    return rotate_curve(x, y, phi)


# =========================================================
# Simulation
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
# DTW with symmetry variants
# =========================================================
def best_variant_dtw(gt, pred):
    global GLOBAL_DTW_BELOW_2
    variants = [
        pred,
        pred[::-1],
        np.column_stack([pred[:,0], -pred[:,1]]),
        np.column_stack([pred[::-1,0], -pred[::-1,1]]),
        np.column_stack([-pred[:,0], pred[:,1]]),
        np.column_stack([-pred[::-1,0], pred[::-1,1]]),
    ]

    best = 1e18
    for V in variants:
        _, dist = dtw_path(gt, V)
        scale = np.sqrt(np.var(gt) + np.var(V)) + 1e-12
        best = min(best, dist / (len(gt) * scale))

    best *= 100.0
    if best < 2.0:
        GLOBAL_DTW_BELOW_2 += 1

    return best


# =========================================================
# PROCESS ONE FIXED INDEX
# =========================================================
def process_curve_with_index(idx):

    print(f"Processing index {idx}")

    latents = np.load(latent_path)
    labels_cont = np.load(labels_cont_path)
    encoded_labels = np.load(encoded_labels_path)

    latent = torch.tensor(latents[idx], dtype=torch.float32, device=device)
    mech_idx = int(encoded_labels[idx])
    mech_name = index_to_label[str(mech_idx)]

    # Ground truth simulation
    gt_points_raw = labels_cont[idx]
    gt_points = clean_and_reshape_label(gt_points_raw)
    P_gt = simulate_curve(gt_points, mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        print("GT simulation failed.")
        return

    cidx = coupler_index_for(mech_name)
    if cidx < 0:
        print("No coupler index.")
        return

    gx = P_gt[:, cidx, 0]
    gy = P_gt[:, cidx, 1]
    gx, gy = normalize_curve(gx, gy)
    gt_curve = np.column_stack([gx, gy])

    # Predict using USER temperature
    tokens = predict_autoregressive_latent(model, latent, mech_idx, tgt_seq_len, TEMPERATURE)
    coord_tokens = [t for t in tokens if t >= BIN_OFFSET]

    if len(coord_tokens) < 4:
        print("Too few predicted tokens.")
        return

    coords = binner.bin_to_value_torch(
        torch.tensor(coord_tokens, device=device) - BIN_OFFSET
    ).cpu().numpy()

    pred_points = coords.reshape(-1, 2)

    Pp = simulate_curve(pred_points, mech_name)
    if Pp is None or Pp.shape[0] < minsteps:
        print("Predicted simulation failed.")
        return

    px = Pp[:, cidx, 0]
    py = Pp[:, cidx, 1]
    px, py = normalize_curve(px, py)
    pred_curve = np.column_stack([px, py])

    dtw_val = best_variant_dtw(gt_curve, pred_curve)
    GLOBAL_DTW_VALUES.append((idx, dtw_val))


# =========================================================
# MAIN
# =========================================================
def main():

    for idx in FIXED_SAMPLE_IDXS:
        process_curve_with_index(idx)

    if GLOBAL_DTW_VALUES:
        print("\n============== RESULTS =================")
        print(f"Temperature = {TEMPERATURE}")
        for (idx, dtw_val) in GLOBAL_DTW_VALUES:
            print(f"Index {idx}: DTW = {dtw_val:.4f}")

        avg_dtw = np.mean([d for (_, d) in GLOBAL_DTW_VALUES])
        print("----------------------------------------")
        print(f"Average DTW: {avg_dtw:.6f}")
        print(f"DTW < 2 count: {GLOBAL_DTW_BELOW_2}")
        print("========================================")
    else:
        print("No successful evaluations.")


# =========================================================
# LOAD MODEL AND RUN
# =========================================================
checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint["model_config"]

model = LatentLLaMA_SingleToken(
    tgt_seq_len=model_config["tgt_seq_len"],
    d_model=model_config["d_model"],
    h=model_config["h"],
    N=model_config["N"],
    num_labels=model_config["num_labels"],
    vocab_size=model_config["vocab_size"],
    latent_dim=model_config["latent_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

if __name__ == "__main__":
    main()
