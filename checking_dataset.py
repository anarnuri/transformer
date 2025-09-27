import numpy as np
import json
import os

def inspect_npy_files(data_dir):
    """Load and inspect all .npy files in the directory - entry by entry"""
    print("=" * 80)
    print("INSPECTING .NPY FILES - ENTRY BY ENTRY")
    print("=" * 80)
    
    # Load all files
    files_to_load = [
        'images.npy', 'decoder_input_discrete.npy', 'labels_discrete.npy',
        'decoder_input_continuous.npy', 'labels_continuous.npy',
        'attention_masks.npy', 'causal_masks.npy',
        'text_labels.npy', 'encoded_labels.npy'
    ]
    
    data = {}
    
    # First load all data
    for file_name in files_to_load:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            data[file_name] = np.load(file_path, allow_pickle=True)
            print(f"Loaded {file_name}: {data[file_name].shape}")
        else:
            print(f"Warning: {file_name} not found in {data_dir}")
    
    # Get the number of entries (use the first available file as reference)
    num_entries = 5
    for file_data in data.values():
        if num_entries is None:
            num_entries = len(file_data)
        break
    
    if num_entries is None:
        print("No data files found!")
        return
    
    # Show entries 1-10
    for entry_idx in range(min(10, num_entries)):
        print(f"\n{'='*60}")
        print(f"ENTRY {entry_idx}")
        print(f"{'='*60}")
        
        for file_name in files_to_load:
            if file_name in data:
                entry_data = data[file_name][entry_idx]
                print(f"\n{file_name}:")
                print(f"  Shape: {entry_data.shape if hasattr(entry_data, 'shape') else 'scalar'}")
                print(f"  Data type: {type(entry_data).__name__}")
                
                if isinstance(entry_data, (np.str_, str)):
                    print(f"  Value: '{entry_data}'")
                    
                elif hasattr(entry_data, 'ndim'):
                    if entry_data.ndim == 0:
                        print(f"  Value: {entry_data}")
                        
                    elif entry_data.ndim == 1:
                        print(f"  Values: {entry_data}")
                        if 'discrete' in file_name:
                            sos_pos = np.where(entry_data == 0)[0]
                            eos_pos = np.where(entry_data == 1)[0]
                            pad_pos = np.where(entry_data == 2)[0]
                            coord_pos = np.where(entry_data >= 3)[0]
                            print(f"  SOS at: {sos_pos}, EOS at: {eos_pos}, PAD at: {pad_pos}, COORD at: {coord_pos}")
                            
                    elif entry_data.ndim == 2:
                        if entry_data.size <= 50:
                            print(f"  Values:\n{entry_data}")
                        else:
                            print(f"  Min: {entry_data.min():.3f}, Max: {entry_data.max():.3f}")
                            if 'causal' in file_name:
                                print(f"  Mask pattern (first 10x10):\n{entry_data[:, :]}")
                                
                    elif entry_data.ndim == 3:
                        print(f"  Shape: {entry_data.shape}")
                        print(f"  Range: [{entry_data.min():.3f}, {entry_data.max():.3f}]")
                        if 'image' in file_name:
                            print(f"  Channel means: {[f'{m:.3f}' for m in entry_data.mean(axis=(1, 2))]}")
                
                print(f"  -" * 20)
    
    # Load and display label mapping
    label_mapping_path = os.path.join(data_dir, "label_mapping.json")
    if os.path.exists(label_mapping_path):
        print(f"\n{'='*60}")
        print("LABEL MAPPING")
        print(f"{'='*60}")
        
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        print(f"Vocabulary size: {label_mapping['vocab_size']}")
        print(f"Number of bins: {label_mapping['num_bins']}")
        print(f"Special tokens: {label_mapping['special_tokens']}")

def main():
    data_dir = "/home/anurizada/Documents/processed_dataset"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist!")
        return
    
    inspect_npy_files(data_dir)

if __name__ == "__main__":
    main()