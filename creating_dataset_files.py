import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
from PIL import Image
import os
from tqdm import tqdm
import json

# Define special tokens
SOS_TOKEN = 0    # Start of sequence
EOS_TOKEN = 1    # End of sequence  
PAD_TOKEN = 2    # Padding
NUM_SPECIAL_TOKENS = 3

class CoordinateBinner:
    """Coordinate binner for values in [-10, 10] range, normalized to [-1, 1]"""
    def __init__(self, num_bins=200):
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-1, 1, num_bins + 1)
        
    def value_to_bin(self, value):
        """Convert continuous value to bin index"""
        normalized = value / 10.0  # [-10, 10] -> [-1, 1]
        return np.digitize(normalized, self.bin_edges) - 1

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
    
    def _collect_image_paths(self):
        """Collect all image paths and parse their metadata"""
        if not self.root_dir.exists():
            raise ValueError(f"Directory '{self.root_dir}' does not exist!")
        
        # First pass: collect all unique labels
        all_labels = set()
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                for file_path in folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                        # Extract label from folder name
                        label = folder.name
                        all_labels.add(label)
        
        # Create label mapping
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        # Second pass: collect data with coordinates
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                label = folder.name
                # Look for coordinate files (e.g., .txt, .npy, .json) or parse from image filenames
                for file_path in folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                        # Try to find corresponding coordinate file
                        coords = self._find_coordinates_for_image(file_path)
                        if coords is not None:
                            self.image_paths.append(file_path)
                            self.metadata.append((coords, label))
                        else:
                            print(f"Warning: No coordinates found for {file_path.name}")
    
    def _find_coordinates_for_image(self, image_path):
        """
        Try to find coordinates for the given image.
        Looks for files with same name but different extensions.
        """
        base_name = image_path.stem
        parent_dir = image_path.parent
        
        # Check for common coordinate file formats
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
        
        # If no coordinate file found, try to parse from filename
        return self._parse_coordinates_from_filename(image_path.name)
    
    def _parse_txt_coordinates(self, file_path):
        """Parse coordinates from text file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                # Try to parse as space or comma separated values
                if ',' in content:
                    coords = [float(x) for x in content.split(',')]
                else:
                    coords = [float(x) for x in content.split()]
                
                if len(coords) % 2 == 0:
                    return np.array(coords).reshape(-1, 2)
                else:
                    print(f"Warning: Odd number of coordinates in {file_path.name}")
                    return None
        except Exception as e:
            print(f"Error parsing text file {file_path.name}: {e}")
            return None
    
    def _parse_json_coordinates(self, file_path):
        """Parse coordinates from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Look for common keys that might contain coordinates
                if 'coordinates' in data:
                    coords = np.array(data['coordinates'])
                elif 'joints' in data:
                    coords = np.array(data['joints'])
                elif 'points' in data:
                    coords = np.array(data['points'])
                else:
                    # Try to find any array in the JSON
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                            coords = np.array(value)
                            break
                    else:
                        print(f"No coordinates found in JSON file {file_path.name}")
                        return None
                
                if coords.size % 2 == 0:
                    return coords.reshape(-1, 2)
                else:
                    print(f"Warning: Odd number of coordinates in {file_path.name}")
                    return None
        except Exception as e:
            print(f"Error parsing JSON file {file_path.name}: {e}")
            return None
    
    def _parse_coordinates_from_filename(self, filename):
        """
        Parse coordinates from filename pattern like:
        - linkage_type_x1_y1_x2_y2.png
        - linkage_type_(x1,y1,x2,y2).png
        - linkage_type_x1y1x2y2.png
        """
        name_without_ext = Path(filename).stem
        
        # Try different patterns
        patterns = [
            # Pattern 1: coordinates in parentheses: (x1,y1,x2,y2)
            r'\(([-?\d\.]+,[-?\d\.]+(?:,[-?\d\.]+,[-?\d\.]+)*)\)',
            # Pattern 2: coordinates separated by underscores: x1_y1_x2_y2
            r'([-?\d\.]+(?:_[-?\d\.]+)+)',
            # Pattern 3: coordinates concatenated: x1y1x2y2
            r'([-?\d\.]+[-?\d\.]+(?:[-?\d\.]+[-?\d\.]+)*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, name_without_ext)
            if matches:
                try:
                    if pattern == patterns[0]:  # Parentheses pattern
                        coord_string = matches[0]
                        coord_values = [float(x) for x in coord_string.split(',')]
                    elif pattern == patterns[1]:  # Underscore pattern
                        coord_string = matches[0]
                        coord_values = [float(x) for x in coord_string.split('_')]
                    else:  # Concatenated pattern
                        # This is tricky - need to split numbers properly
                        coord_string = matches[0]
                        # Split on transitions between digits and non-digits
                        coord_values = [float(x) for x in re.findall(r'-?\d+\.?\d*', coord_string)]
                    
                    if len(coord_values) % 2 == 0:
                        return np.array(coord_values).reshape(-1, 2)
                    else:
                        print(f"Warning: Odd number of coordinates in {filename}")
                        return None
                except ValueError as e:
                    continue
        
        print(f"Warning: Could not parse coordinates from filename: {filename}")
        return None

    def _create_sequences(self, coordinates, text_label):
        """
        Create decoder input and label sequences.
        """
        max_seq_len = self.max_joints * 2 + 1
        
        # Flatten coordinates
        if coordinates is not None and len(coordinates) > 0:
            flat_coords = coordinates.flatten()
        else:
            flat_coords = np.array([])
        
        n_coords = min(len(flat_coords), self.max_joints * 2)
        actual_coords = flat_coords[:n_coords]
        
        # CONTINUOUS sequences
        decoder_input_continuous = np.full(max_seq_len, -1.0)  # PAD
        label_continuous = np.full(max_seq_len, -1.0)          # PAD
        
        decoder_input_continuous[0] = -2.0  # SOS
        if n_coords > 0:
            decoder_input_continuous[1:1+n_coords] = actual_coords
        
        if n_coords > 0:
            label_continuous[:n_coords] = actual_coords
        label_continuous[n_coords] = 2.0  # EOS
        
        # DISCRETE sequences
        decoder_input_discrete = np.full(max_seq_len, PAD_TOKEN, dtype=int)
        label_discrete = np.full(max_seq_len, PAD_TOKEN, dtype=int)
        
        if n_coords > 0:
            binned_coords = self.binner.value_to_bin(actual_coords) + NUM_SPECIAL_TOKENS
        
        decoder_input_discrete[0] = SOS_TOKEN
        if n_coords > 0:
            decoder_input_discrete[1:1+n_coords] = binned_coords
        
        if n_coords > 0:
            label_discrete[:n_coords] = binned_coords
        label_discrete[n_coords] = EOS_TOKEN
        
        return {
            'continuous': {
                'decoder_input': decoder_input_continuous,
                'label': label_continuous
            },
            'discrete': {
                'decoder_input': decoder_input_discrete,
                'label': label_discrete
            },
            'text_label': text_label,
            'num_joints': n_coords // 2,
            'original_coords': coordinates if coordinates is not None else np.array([])
        }
    
    def _create_attention_mask(self, sequence_length, num_real_tokens):
        """Create attention mask (1 for real tokens, 0 for padding)"""
        mask = np.zeros(sequence_length, dtype=bool)
        mask[:num_real_tokens + 1] = True  # +1 for SOS token
        return mask
    
    def _create_causal_mask(self, sequence_length, num_real_tokens):
        """Create causal mask for decoder that accounts for padding tokens"""
        # Create the base triangular causal mask
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
        causal_mask = ~causal_mask  # True for allowed positions, False for masked
        
        # Create padding mask: False for padding tokens, True for real tokens
        padding_mask = torch.zeros(sequence_length, dtype=torch.bool)
        padding_mask[:num_real_tokens + 1] = True  # +1 for SOS token
        
        # Combine causal mask with padding mask
        # For each position, we need to mask out padding tokens in both rows and columns
        final_mask = causal_mask & padding_mask.unsqueeze(0) & padding_mask.unsqueeze(1)
        
        return final_mask
    
    def _shuffle_data(self):
        """Shuffle the dataset"""
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        joint_coords, text_label = self.metadata[idx]
        
        # Load image
        try:
            image = Image.open(image_path)
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(-1)
            image_tensor = image_tensor.permute(2, 0, 1)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros((3, 64, 64), dtype=torch.float32)
        
        # Create sequences
        sequences = self._create_sequences(joint_coords, text_label)
        max_seq_len = self.max_joints * 2 + 1
        num_real_tokens = sequences['num_joints'] * 2
        
        # Create masks - UPDATED to include num_real_tokens
        attention_mask = self._create_attention_mask(max_seq_len, num_real_tokens)
        causal_mask = self._create_causal_mask(max_seq_len, num_real_tokens)  # Updated call
        
        return {
            "image": image_tensor,
            "text_label": text_label,
            "decoder_input_continuous": torch.tensor(sequences['continuous']['decoder_input'], dtype=torch.float32),
            "label_continuous": torch.tensor(sequences['continuous']['label'], dtype=torch.float32),
            "decoder_input_discrete": torch.tensor(sequences['discrete']['decoder_input'], dtype=torch.long),
            "label_discrete": torch.tensor(sequences['discrete']['label'], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "causal_mask": causal_mask,  # Now properly handles padding
            "encoded_label": torch.tensor(self.label_to_index.get(text_label, 0), dtype=torch.long),
            "num_joints": sequences['num_joints'],
            "original_coords": sequences['original_coords']
        }

def inspect_dataset(dataset, num_samples=5):
    """Comprehensive function to inspect dataset samples and verify correctness"""
    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.label_to_index)}")
    print(f"Classes: {list(dataset.label_to_index.keys())}")
    print(f"Max joints: {dataset.max_joints}")
    print(f"Number of bins: {dataset.num_bins}")
    print(f"Vocabulary size: {NUM_SPECIAL_TOKENS + dataset.num_bins}")
    
    print(f"\nInspecting {num_samples} samples:")
    print("-" * 80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        filename = dataset.image_paths[i].name
        
        print(f"\nSample {i}: {filename}")
        print(f"  Text label: '{sample['text_label']}'")
        print(f"  Encoded label: {sample['encoded_label'].item()}")
        print(f"  Num joints: {sample['num_joints']}")
        print(f"  Original coords shape: {sample['original_coords'].shape}")
        print(f"  Original coordinates:\n{sample['original_coords']}")
        
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        
        print(f"  Decoder input continuous (first 10): {sample['decoder_input_continuous'][:10].numpy()}")
        print(f"  Label continuous (first 10): {sample['label_continuous'][:10].numpy()}")
        
        print(f"  Decoder input discrete (first 10): {sample['decoder_input_discrete'][:10].numpy()}")
        print(f"  Label discrete (first 10): {sample['label_discrete'][:10].numpy()}")
        
        print(f"  Attention mask (first 10): {sample['attention_mask'][:10].numpy()}")
        print(f"  Causal mask shape: {sample['causal_mask'].shape}")
        
        # Verify special tokens
        discrete_input = sample['decoder_input_discrete'].numpy()
        discrete_label = sample['label_discrete'].numpy()
        
        print(f"  SOS token position: {np.where(discrete_input == SOS_TOKEN)[0]}")
        print(f"  EOS token position: {np.where(discrete_label == EOS_TOKEN)[0]}")
        print(f"  PAD token count: {np.sum(discrete_input == PAD_TOKEN)}")
        
        # Verify binning
        if sample['num_joints'] > 0:
            original_coords_flat = sample['original_coords'].flatten()[:10]  # First 10 coords
            binned_coords = discrete_input[1:1+len(original_coords_flat)] - NUM_SPECIAL_TOKENS
            reconstructed = (binned_coords / dataset.num_bins * 2 - 1) * 10  # Reverse binning
            
            print(f"  Original first coords: {original_coords_flat}")
            print(f"  Binned coords: {binned_coords}")
            print(f"  Reconstructed: {reconstructed}")
            print(f"  Reconstruction error: {np.abs(original_coords_flat - reconstructed).mean():.6f}")
        
        print("-" * 80)


def create_npy_files(dataset, output_dir):
    """
    Process the dataset and save ALL components as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store ALL data
    all_images = []
    all_decoder_inputs_discrete = []
    all_labels_discrete = []
    all_decoder_inputs_continuous = []
    all_labels_continuous = []
    all_attention_masks = []
    all_causal_masks = []
    all_text_labels = []
    all_encoded_labels = []
    
    print(f"Processing {len(dataset)} samples...")
    
    # Process each sample
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
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_decoder_inputs_discrete = np.array(all_decoder_inputs_discrete)
    all_labels_discrete = np.array(all_labels_discrete)
    all_decoder_inputs_continuous = np.array(all_decoder_inputs_continuous)
    all_labels_continuous = np.array(all_labels_continuous)
    all_attention_masks = np.array(all_attention_masks)
    all_causal_masks = np.array(all_causal_masks)
    all_encoded_labels = np.array(all_encoded_labels)
    
    # Save ALL files
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
    
    # CORRECT vocabulary size calculation
    vocab_size = NUM_SPECIAL_TOKENS + dataset.num_bins  # Only special tokens + coordinate bins
    
    # Save label mapping with CORRECT vocabulary size
    with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
        json.dump({
            'label_to_index': dataset.label_to_index,
            'index_to_label': dataset.index_to_label,
            'special_tokens': {
                'SOS': SOS_TOKEN,
                'EOS': EOS_TOKEN,
                'PAD': PAD_TOKEN,
                'NUM_SPECIAL_TOKENS': NUM_SPECIAL_TOKENS
            },
            'num_bins': dataset.num_bins,
            'coordinate_range': [-10, 10],
            'vocab_size': vocab_size,  # CORRECT: 3 + 200 = 203
            'label_vocab_size': len(dataset.label_to_index)  # Separate for classification
        }, f, indent=2)
    
    print(f"Files saved to {output_dir}:")
    print(f"  - images.npy: {all_images.shape}")
    print(f"  - decoder_input_discrete.npy: {all_decoder_inputs_discrete.shape}")
    print(f"  - labels_discrete.npy: {all_labels_discrete.shape}")
    print(f"  - decoder_input_continuous.npy: {all_decoder_inputs_continuous.shape}")
    print(f"  - labels_continuous.npy: {all_labels_continuous.shape}")
    print(f"  - attention_masks.npy: {all_attention_masks.shape}")
    print(f"  - causal_masks.npy: {all_causal_masks.shape}")
    print(f"  - text_labels.npy: {len(all_text_labels)}")
    print(f"  - encoded_labels.npy: {all_encoded_labels.shape}")
    print(f"  - Vocabulary size: {vocab_size} (special tokens + coordinate bins)")
    print(f"  - Label vocabulary size: {len(dataset.label_to_index)} (for classification)")


if __name__ == "__main__":
    # Create the dataset
    dataset = ImageLinkageDataset(
        root_dir="/home/anurizada/Documents/bar_linkages", 
        max_joints=12,
        num_bins=200
    )
    
    # Inspect the dataset thoroughly
    inspect_dataset(dataset, num_samples=5)
    
    # Ask for confirmation before creating files
    response = input("\nDo you want to proceed with creating .npy files? (y/n): ")
    if response.lower() == 'y':
        output_dir = "/home/anurizada/Documents/processed_dataset"
        create_npy_files(dataset, output_dir)
        print(f"\nFiles created successfully in {output_dir}!")
    else:
        print("Operation cancelled.")