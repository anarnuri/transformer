import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class BarLinkageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # ----------------------------
        # Load all base arrays
        # ----------------------------
        self.images = np.load(f"{data_dir}/images.npy")
        self.decoder_input_discrete = np.load(f"{data_dir}/decoder_input_discrete.npy")
        self.labels_discrete = np.load(f"{data_dir}/labels_discrete.npy")
        self.attention_masks = np.load(f"{data_dir}/attention_masks.npy")
        self.causal_masks = np.load(f"{data_dir}/causal_masks.npy")
        self.encoded_labels = np.load(f"{data_dir}/encoded_labels.npy")

        # ----------------------------
        # Optionally load VAE latents
        # ----------------------------
        vae_mu_path = os.path.join(data_dir, "vae_mu.npy")
        if os.path.exists(vae_mu_path):
            self.vae_mu = np.load(vae_mu_path)
            print(f"✅ Loaded VAE latents: {self.vae_mu.shape}")
        else:
            self.vae_mu = None
            print("⚠️ No vae_mu.npy found — continuing without latent vectors")

        # ----------------------------
        # Load label mapping
        # ----------------------------
        with open(f"{data_dir}/label_mapping.json", 'r') as f:
            self.label_mapping = json.load(f)

        # ----------------------------
        # Sanity checks
        # ----------------------------
        n = len(self.images)
        for name, arr in [
            ("decoder_input_discrete", self.decoder_input_discrete),
            ("labels_discrete", self.labels_discrete),
            ("attention_masks", self.attention_masks),
            ("causal_masks", self.causal_masks),
            ("encoded_labels", self.encoded_labels),
        ]:
            if len(arr) != n:
                raise ValueError(f"❌ Length mismatch: {name} has {len(arr)} samples, but images have {n}")

        if self.vae_mu is not None and len(self.vae_mu) != n:
            raise ValueError(
                f"❌ vae_mu.npy length mismatch: {len(self.vae_mu)} latents, but {n} images."
            )

        print(f"✅ Loaded dataset from {data_dir} with {n} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = {
            "images": torch.tensor(self.images[idx], dtype=torch.float32),
            "decoder_input_discrete": torch.tensor(self.decoder_input_discrete[idx], dtype=torch.long),
            "labels_discrete": torch.tensor(self.labels_discrete[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.bool),
            "causal_mask": torch.tensor(self.causal_masks[idx], dtype=torch.bool),
            "encoded_labels": torch.tensor(self.encoded_labels[idx], dtype=torch.long),
            "indices": torch.tensor(idx, dtype=torch.long),  # <-- added index for alignment
        }

        # Only include vae_mu if available
        if self.vae_mu is not None:
            sample["vae_mu"] = torch.tensor(self.vae_mu[idx], dtype=torch.float32)

        return sample
