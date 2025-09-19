import torch
from torch.utils.data import Dataset
import numpy as np
import json

class BarLinkageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = np.load(f"{data_dir}/images.npy")
        self.decoder_input_discrete = np.load(f"{data_dir}/decoder_input_discrete.npy")
        self.labels_discrete = np.load(f"{data_dir}/labels_discrete.npy")
        self.attention_masks = np.load(f"{data_dir}/attention_masks.npy")
        self.causal_masks = np.load(f"{data_dir}/causal_masks.npy")
        self.encoded_labels = np.load(f"{data_dir}/encoded_labels.npy")
        
        # Load label mapping
        with open(f"{data_dir}/label_mapping.json", 'r') as f:
            self.label_mapping = json.load(f)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            "images": torch.tensor(self.images[idx], dtype=torch.float32),
            "decoder_input_discrete": torch.tensor(self.decoder_input_discrete[idx], dtype=torch.long),
            "labels_discrete": torch.tensor(self.labels_discrete[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.bool),
            "causal_mask": torch.tensor(self.causal_masks[idx], dtype=torch.bool),
            "encoded_labels": torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }