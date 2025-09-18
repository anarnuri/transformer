import numpy as np
import torch
from torch.utils.data import Dataset
import os
from binner import CoordinateBinner  # Import your CoordinateBinner class

class SingleTransformerDataset(Dataset):
    def __init__(self, data_dir, kappa=1.0, num_bins=200, use_binning=True):
        """
        Dataset that loads individual .npy files with optional coordinate binning.
        """
        self.curves = np.load(os.path.join(data_dir, "images.npy"), mmap_mode='r')
        self.dec_in = np.load(os.path.join(data_dir, "decoder_input.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')
        self.masks = np.load(os.path.join(data_dir, "masks.npy"), mmap_mode='r')
        
        # Load text labels (can't be memory-mapped due to Python objects)
        self.text_labels = np.load(os.path.join(data_dir, "text_labels.npy"), allow_pickle=True)
        
        self.use_binning = use_binning
        if use_binning:
            self.binner = CoordinateBinner(kappa=kappa, num_bins=num_bins)
        else:
            self.binner = None
    
    def _discretize_with_special_tokens(self, tensor):
        """
        Discretize coordinates while preserving special tokens.
        Special tokens: SOS (-2), PAD (-1), EOS (2)
        """
        # Create mask for special tokens
        sos_mask = (tensor == -2.0).all(dim=-1)
        pad_mask = (tensor == -1.0).all(dim=-1)
        eos_mask = (tensor == 2.0).all(dim=-1)
        special_mask = sos_mask | pad_mask | eos_mask
        
        # Discretize non-special tokens
        discretized = torch.zeros_like(tensor, dtype=torch.long)
        
        # Handle x coordinates
        x_discrete = self.binner.value_to_bin_torch(tensor[:, 0])
        discretized[:, 0] = torch.where(special_mask, tensor[:, 0].long(), x_discrete)
        
        # Handle y coordinates
        y_discrete = self.binner.value_to_bin_torch(tensor[:, 1])
        discretized[:, 1] = torch.where(special_mask, tensor[:, 1].long(), y_discrete)
        
        return discretized

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        # Get the raw data
        curve_numerical = torch.tensor(self.curves[idx], dtype=torch.float32)
        decoder_input_continuous = torch.tensor(self.dec_in[idx], dtype=torch.float32)
        label_continuous = torch.tensor(self.labels[idx], dtype=torch.float32)
        decoder_mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        text_label = self.text_labels[idx]
        
        # Apply binning if enabled - CORRECTED: call binner methods
        if self.use_binning:
            decoder_input = self._discretize_with_special_tokens(decoder_input_continuous)
            label = self._discretize_with_special_tokens(label_continuous)
        else:
            decoder_input = decoder_input_continuous
            label = label_continuous
        
        return {
            "curve_numerical": curve_numerical,
            "decoder_input": decoder_input,
            "label": label,
            "decoder_mask": decoder_mask,
            "text_label": text_label,
            "decoder_input_continuous": decoder_input_continuous,
            "label_continuous": label_continuous,
        }
