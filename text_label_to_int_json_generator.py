import numpy as np
import json
import os

def create_simple_label_mapping(data_dir, output_path=None):
    """
    Create a simple JSON file with label to index mapping.
    
    Args:
        data_dir: Directory containing the text_labels.npy file
        output_path: Path to save the JSON file (optional)
    
    Returns:
        dict: Label to index mapping
    """
    # Load text labels
    text_labels = np.load(os.path.join(data_dir, "text_labels.npy"), allow_pickle=True)
    
    # Get unique labels and sort them alphabetically for consistent ordering
    unique_labels = sorted(np.unique(text_labels))
    
    # Create mapping from text label to numerical index
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join(data_dir, "label_mapping.json")
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(label_to_index, f, indent=2)
    
    print(f"Label mapping saved to {output_path}")
    return label_to_index

# Usage
if __name__ == "__main__":
    data_dir = "/home/anurizada/Documents/processed_dataset"
    mapping = create_simple_label_mapping(data_dir)
    
    # Print the first 10 mappings as example
    print("First 10 label mappings:")
    for i, (label, index) in enumerate(list(mapping.items())[:10]):
        print(f"  '{label}' → {index}")