import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
from PIL import Image
import os
from tqdm import tqdm
import json

# ------------------------------------------------------------------
# Prevent numpy and torch from truncating printed arrays/tensors
# ------------------------------------------------------------------
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
torch.set_printoptions(profile="full", linewidth=200)

# Define special tokens
SOS_TOKEN = 0    # Start of sequence
EOS_TOKEN = 1    # End of sequence  
PAD_TOKEN = 2    # Padding
NUM_SPECIAL_TOKENS = 3

# Offsets per your spec
MECH_OFFSET = 3    # 3..19 (17 mech types)
BIN_OFFSET  = 20   # bins start at 20


# ------------------------------------------------------------------
# Coordinate Binner
# ------------------------------------------------------------------
class CoordinateBinner:
    """Coordinate binner for values in [-10, 10] range, normalized to [-1, 1]"""
    def __init__(self, num_bins=200):
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-1, 1, num_bins + 1)
        
    def value_to_bin(self, value):
        """Convert continuous value to bin index"""
        normalized = value / 10.0  # [-10, 10] -> [-1, 1]
        return np.digitize(normalized, self.bin_edges) - 1


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ImageLinkageDataset(Dataset):
    def __init__(self, root_dir="bar_linkages", image_extensions=None, shuffle=True, 
                 max_joints=12, num_bins=200):
        self.root_dir = Path(root_dir)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        self.max_joints = max_joints
        self.num_bins = num_bins
        self.binner = CoordinateBinner(num_bins=num_bins)
        self.image_paths = []
        self.metadata = []
        self.label_to_index = {}
        self.index_to_label = {}
        
        self._collect_image_paths()
        if shuffle:
            self._shuffle_data()

    # --------------------------------------------------------------
    # Data Collection
    # --------------------------------------------------------------
    def _collect_image_paths(self):
        """Collect all image paths and parse their metadata"""
        if not self.root_dir.exists():
            raise ValueError(f"Directory '{self.root_dir}' does not exist!")
        
        all_labels = set()
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                for file_path in folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                        all_labels.add(folder.name)
        
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                label = folder.name
                for file_path in folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                        coords = self._find_coordinates_for_image(file_path)
                        if coords is not None:
                            self.image_paths.append(file_path)
                            self.metadata.append((coords, label))
                        else:
                            print(f"Warning: No coordinates found for {file_path.name}")

    def _find_coordinates_for_image(self, image_path):
        base_name = image_path.stem
        parent_dir = image_path.parent
        possible_extensions = ['.txt', '.npy', '.json', '.csv']

        for ext in possible_extensions:
            coord_file = parent_dir / f"{base_name}{ext}"
            if coord_file.exists():
                try:
                    if ext == '.npy':
                        return np.load(coord_file)
                    elif ext == '.txt':
                        return self._parse_txt_coordinates(coord_file)
                    elif ext == '.json':
                        return self._parse_json_coordinates(coord_file)
                    elif ext == '.csv':
                        return np.loadtxt(coord_file, delimiter=',')
                except Exception as e:
                    print(f"Error reading {coord_file}: {e}")
        return self._parse_coordinates_from_filename(image_path.name)

    def _parse_txt_coordinates(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if ',' in content:
                    coords = [float(x) for x in content.split(',')]
                else:
                    coords = [float(x) for x in content.split()]
                if len(coords) % 2 == 0:
                    return np.array(coords).reshape(-1, 2)
        except Exception as e:
            print(f"Error parsing text file {file_path.name}: {e}")
        return None

    def _parse_json_coordinates(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for key in ['coordinates', 'joints', 'points']:
                if key in data:
                    coords = np.array(data[key])
                    if coords.size % 2 == 0:
                        return coords.reshape(-1, 2)
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                    coords = np.array(value)
                    if coords.size % 2 == 0:
                        return coords.reshape(-1, 2)
        except Exception as e:
            print(f"Error parsing JSON file {file_path.name}: {e}")
        return None

    def _parse_coordinates_from_filename(self, filename):
        name = Path(filename).stem
        patterns = [
            r'\(([-?\d\.]+,[-?\d\.]+(?:,[-?\d\.]+,[-?\d\.]+)*)\)',
            r'([-?\d\.]+(?:_[-?\d\.]+)+)',
            r'([-?\d\.]+[-?\d\.]+(?:[-?\d\.]+[-?\d\.]+)*)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, name)
            if matches:
                coord_string = matches[0]
                numbers = re.findall(r'-?\d+\.?\d*', coord_string)
                coords = np.array([float(x) for x in numbers])
                if len(coords) % 2 == 0:
                    return coords.reshape(-1, 2)
        print(f"Warning: Could not parse coordinates from filename: {filename}")
        return None

    # --------------------------------------------------------------
    # Sequence creation and masking
    # --------------------------------------------------------------
    def _create_sequences(self, coordinates, text_label):
        # CHANGED: +2 to fit [SOS, MECH, ...bins...] and [MECH, ...bins..., EOS]
        max_seq_len = self.max_joints * 2 + 2

        flat = coordinates.flatten() if coordinates is not None else np.array([])
        n_coords = min(len(flat), self.max_joints * 2)
        coords = flat[:n_coords]

        # --- Continuous (unchanged logic/placement to keep minimal diffs) ---
        decoder_input_continuous = np.full(max_seq_len, -1.0)
        label_continuous = np.full(max_seq_len, -1.0)
        decoder_input_continuous[0] = -2.0
        decoder_input_continuous[1:1+n_coords] = coords
        label_continuous[:n_coords] = coords
        label_continuous[n_coords] = 2.0

        # --- Discrete (UPDATED to your spec) ---
        decoder_input_discrete = np.full(max_seq_len, PAD_TOKEN, dtype=int)
        label_discrete = np.full(max_seq_len, PAD_TOKEN, dtype=int)

        mech_idx = self.label_to_index[text_label]  # 0..16
        mech_token = MECH_OFFSET + mech_idx         # 3..19
        binned = self.binner.value_to_bin(coords) + BIN_OFFSET if n_coords > 0 else np.array([], dtype=int)

        # Decoder input: [SOS, MECH, BIN..., PAD...]
        decoder_input_discrete[0] = SOS_TOKEN
        decoder_input_discrete[1] = mech_token
        if n_coords > 0:
            decoder_input_discrete[2:2+n_coords] = binned

        # Label: [MECH, BIN..., EOS, PAD...]
        label_discrete[0] = mech_token
        if n_coords > 0:
            label_discrete[1:1+n_coords] = binned
        label_discrete[1 + n_coords] = EOS_TOKEN

        return {
            "continuous": {"decoder_input": decoder_input_continuous, "label": label_continuous},
            "discrete": {"decoder_input": decoder_input_discrete, "label": label_discrete},
            "text_label": text_label,
            "num_joints": n_coords // 2,
            "original_coords": coordinates if coordinates is not None else np.array([])
        }

    def _create_attention_mask(self, sequence_length, num_real_tokens):
        # CHANGED: cover SOS + MECH + n_coords bins
        mask = np.zeros(sequence_length, dtype=bool)
        mask[:num_real_tokens + 2] = True
        return mask

    def _create_causal_mask(self, sequence_length, num_real_tokens):
        # CHANGED: cover SOS + MECH + n_coords bins
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
        causal_mask = ~causal_mask
        padding_mask = torch.zeros(sequence_length, dtype=torch.bool)
        padding_mask[:num_real_tokens + 2] = True
        return causal_mask & padding_mask.unsqueeze(0) & padding_mask.unsqueeze(1)

    def _shuffle_data(self):
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        coords, label = self.metadata[idx]
        try:
            image = Image.open(image_path)
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(-1)
            image_tensor = image_tensor.permute(2, 0, 1)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros((3, 64, 64))

        seq = self._create_sequences(coords, label)

        # CHANGED: +2 to match the new max_seq_len above
        max_len = self.max_joints * 2 + 2
        num_real = seq["num_joints"] * 2
        attn_mask = self._create_attention_mask(max_len, num_real)
        causal_mask = self._create_causal_mask(max_len, num_real)
        return {
            "image": image_tensor,
            "text_label": label,
            "decoder_input_continuous": torch.tensor(seq["continuous"]["decoder_input"]),
            "label_continuous": torch.tensor(seq["continuous"]["label"]),
            "decoder_input_discrete": torch.tensor(seq["discrete"]["decoder_input"]),
            "label_discrete": torch.tensor(seq["discrete"]["label"]),
            "attention_mask": torch.tensor(attn_mask),
            "causal_mask": causal_mask,
            "encoded_label": torch.tensor(self.label_to_index.get(label, 0)),
            "num_joints": seq["num_joints"],
            "original_coords": seq["original_coords"]
        }


# ------------------------------------------------------------------
# Dataset Inspection (prints all tensors completely)
# ------------------------------------------------------------------
def inspect_dataset(dataset, num_samples=5):
    print("=" * 100)
    print("FULL DATASET INSPECTION")
    print("=" * 100)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.label_to_index}")
    print(f"Max joints: {dataset.max_joints}, bins: {dataset.num_bins}")
    print("-" * 100)

    for i in range(min(num_samples, len(dataset))):
        s = dataset[i]
        name = dataset.image_paths[i].name
        print(f"\n=== Sample {i}: {name} ===")
        print(f"Text label: {s['text_label']} | Encoded: {s['encoded_label']}")
        print(f"Num joints: {s['num_joints']}")
        print(f"Original coords:\n{s['original_coords']}")
        print(f"Image shape: {tuple(s['image'].shape)} | Range: [{s['image'].min():.3f}, {s['image'].max():.3f}]")

        print("\n--- Continuous sequences ---")
        print(f"Decoder input:\n{s['decoder_input_continuous']}")
        print(f"Label:\n{s['label_continuous']}")

        print("\n--- Discrete sequences ---")
        print(f"Decoder input:\n{s['decoder_input_discrete']}")
        print(f"Label:\n{s['label_discrete']}")

        print("\n--- Masks ---")
        print(f"Attention mask:\n{s['attention_mask']}")
        print(f"Causal mask:\n{s['causal_mask']}")

        print("-" * 100)


# ------------------------------------------------------------------
# NPY File Creation
# ------------------------------------------------------------------
def create_npy_files(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_images, all_decoder_inputs_discrete, all_labels_discrete = [], [], []
    all_decoder_inputs_continuous, all_labels_continuous = [], []
    all_attention_masks, all_causal_masks = [], []
    all_text_labels, all_encoded_labels = [], []

    print(f"Processing {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        all_images.append(sample["image"].numpy())
        all_decoder_inputs_discrete.append(sample["decoder_input_discrete"].numpy())
        all_labels_discrete.append(sample["label_discrete"].numpy())
        all_decoder_inputs_continuous.append(sample["decoder_input_continuous"].numpy())
        all_labels_continuous.append(sample["label_continuous"].numpy())
        all_attention_masks.append(sample["attention_mask"].numpy())
        all_causal_masks.append(sample["causal_mask"].numpy())
        all_text_labels.append(sample["text_label"])
        all_encoded_labels.append(sample["encoded_label"].numpy())

    all_images = np.array(all_images)
    all_decoder_inputs_discrete = np.array(all_decoder_inputs_discrete)
    all_labels_discrete = np.array(all_labels_discrete)
    all_decoder_inputs_continuous = np.array(all_decoder_inputs_continuous)
    all_labels_continuous = np.array(all_labels_continuous)
    all_attention_masks = np.array(all_attention_masks)
    all_causal_masks = np.array(all_causal_masks)
    all_encoded_labels = np.array(all_encoded_labels)

    print("Saving .npy files...")
    np.save(os.path.join(output_dir, "images.npy"), all_images)
    np.save(os.path.join(output_dir, "decoder_input_discrete.npy"), all_decoder_inputs_discrete)
    np.save(os.path.join(output_dir, "labels_discrete.npy"), all_labels_discrete)
    np.save(os.path.join(output_dir, "decoder_input_continuous.npy"), all_decoder_inputs_continuous)
    np.save(os.path.join(output_dir, "labels_continuous.npy"), all_labels_continuous)
    np.save(os.path.join(output_dir, "attention_masks.npy"), all_attention_masks)
    np.save(os.path.join(output_dir, "causal_masks.npy"), all_causal_masks)
    np.save(os.path.join(output_dir, "text_labels.npy"), np.array(all_text_labels, dtype=object))
    np.save(os.path.join(output_dir, "encoded_labels.npy"), all_encoded_labels)

    # CHANGED: vocab size accounts for mech block and bin block (bins start at BIN_OFFSET)
    vocab_size = BIN_OFFSET + dataset.num_bins
    with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
        json.dump({
            'label_to_index': dataset.label_to_index,
            'index_to_label': dataset.index_to_label,
            'special_tokens': {'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN, 'PAD': PAD_TOKEN},
            'num_bins': dataset.num_bins,
            'coordinate_range': [-10, 10],
            'vocab_size': vocab_size,
            'label_vocab_size': len(dataset.label_to_index),
            'mech_token_offset': MECH_OFFSET,
            'bin_token_offset': BIN_OFFSET
        }, f, indent=2)

    print(f"\nFiles saved to {output_dir} successfully.")


# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    dataset = ImageLinkageDataset(
        root_dir="/home/anurizada/Documents/bar_linkages",
        max_joints=8,
        num_bins=200
    )
    inspect_dataset(dataset, num_samples=5)

    response = input("\nDo you want to proceed with creating .npy files? (y/n): ")
    if response.lower() == 'y':
        output_dir = "/home/anurizada/Documents/processed_dataset"
        create_npy_files(dataset, output_dir)
        print(f"\nFiles created successfully in {output_dir}!")
    else:
        print("Operation cancelled.")
