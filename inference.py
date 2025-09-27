import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SingleImageTransformer
from dataset import BarLinkageDataset  # Your dataset class

class InferenceEngine:
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model_config = self.checkpoint['model_config']
        
        # Initialize model
        self.model = SingleImageTransformer(
            tgt_seq_len=self.model_config['tgt_seq_len'],
            d_model=self.model_config['d_model'],
            h=self.model_config['h'],
            N=self.model_config['N'],
            num_labels=self.model_config['num_labels'],
            vocab_size=self.model_config['vocab_size']+1,
        ).to(device)
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model configuration: {self.model_config}")
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. Returns Boolean mask with batch dimension (1, sz, sz).
        True positions are allowed, False positions are masked.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # Add batch dimension: (1, sz, sz)
        # mask = mask.unsqueeze(0)
        return mask
    
    def predict_iterative(self, dataloader, max_samples=100, return_details=False):
        """
        Generate predictions iteratively (token by token) using autoregressive decoding
        """
        all_predictions = []
        all_targets = []
        all_images = []
        all_labels = []
        samples_processed = 0
        
        # Special tokens (adjust based on your tokenization)
        sos_token = 0  # Start of sequence
        eos_token = 1  # End of sequence  
        pad_token = 2  # Padding token
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running iterative inference"):
                if samples_processed >= max_samples:
                    break
                    
                images = batch["images"].to(self.device)
                labels = batch["encoded_labels"].to(self.device)
                target_tokens = batch["labels_discrete"].to(self.device)
                
                batch_size = images.shape[0]
                max_seq_len = self.model_config['tgt_seq_len']
                
                # Initialize decoder input with SOS token
                decoder_input = torch.full((batch_size, 1), sos_token, 
                                         dtype=torch.long, device=self.device)
                
                # Store predictions for each sample
                batch_predictions = [[] for _ in range(batch_size)]
                completed = [False] * batch_size
                
                # Autoregressive decoding
                for step in range(max_seq_len):
                    # Create causal mask for current sequence length
                    seq_len = decoder_input.shape[1]
                    causal_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)
                    print("Iterative")
                    print(causal_mask)
                    # Forward pass
                    predictions, image_emb, label_emb = self.model(
                        decoder_input, causal_mask, images, labels
                    )
                    
                    # Get next token prediction (only use the last predicted token)
                    next_token_logits = predictions[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1)
                    
                    # Update decoder input with new token
                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
                    
                    # Store predictions and check for completion
                    for i in range(batch_size):
                        if not completed[i]:
                            batch_predictions[i].append(next_token[i].item())
                            # Check for EOS token or max length
                            if next_token[i].item() == eos_token or len(batch_predictions[i]) >= max_seq_len:
                                completed[i] = True
                    
                    # Stop if all sequences are completed
                    if all(completed):
                        break
                
                # Process each sample in the batch
                for i in range(batch_size):
                    if samples_processed >= max_samples:
                        break
                        
                    pred_seq = np.array(batch_predictions[i])
                    target_seq = target_tokens[i].cpu().numpy()
                    
                    # Remove padding from target
                    target_seq = target_seq[target_seq != pad_token]
                    
                    all_predictions.append(pred_seq)
                    all_targets.append(target_seq)
                    
                    if return_details:
                        all_images.append(images[i].cpu())
                        all_labels.append(labels[i].cpu())
                    
                    samples_processed += 1
        
        print(f"Processed {samples_processed} samples iteratively")
        
        if return_details:
            return all_predictions, all_targets, all_images, all_labels
        return all_predictions, all_targets
    
    def predict_single_sequence(self, image, label, max_length=50):
        """
        Predict a single sequence given image and label tensors
        """
        # Ensure inputs are properly shaped
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(label.shape) == 1:
            label = label.unsqueeze(0)
            
        image = image.to(self.device)
        label = label.to(self.device)
        
        sos_token = 0
        eos_token = 1
        
        with torch.no_grad():
            # Initialize with SOS token
            decoder_input = torch.tensor([[sos_token]], dtype=torch.long, device=self.device)
            predicted_tokens = []
            
            for step in range(max_length):
                seq_len = decoder_input.shape[1]
                causal_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)
                
                print("Single")
                print(causal_mask)

                predictions, _, _ = self.model(
                    decoder_input, causal_mask, image, label
                )
                
                next_token_logits = predictions[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1).item()
                
                predicted_tokens.append(next_token)
                
                # Stop if EOS token is generated
                if next_token == eos_token:
                    break
                
                # Update decoder input
                decoder_input = torch.cat([
                    decoder_input, 
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)
        
        return np.array(predicted_tokens)
    
    # Keep the original predict_batch for comparison
    def predict_batch(self, dataloader, max_samples=100, return_details=False):
        """
        Original method using teacher forcing (for comparison)
        """
        all_predictions = []
        all_targets = []
        all_images = []
        all_labels = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running inference"):
                if samples_processed >= max_samples:
                    break
                    
                decoder_input = batch["decoder_input_discrete"].to(self.device)
                decoder_mask = batch["causal_mask"].to(self.device)
                images = batch["images"].to(self.device)
                labels = batch["encoded_labels"].to(self.device)
                target_tokens = batch["labels_discrete"].to(self.device)
                
                print("Batch")
                print(decoder_mask)

                # Forward pass
                predictions, image_emb, label_emb = self.model(
                    decoder_input, decoder_mask, images, labels
                )
                
                # Get predicted tokens (greedy decoding)
                pred_tokens = predictions.argmax(dim=-1)
                
                # Remove padding for comparison (assuming pad_token=2)
                pad_token = 2
                for i in range(pred_tokens.shape[0]):
                    if samples_processed >= max_samples:
                        break
                        
                    pred_seq = pred_tokens[i].cpu().numpy()
                    target_seq = target_tokens[i].cpu().numpy()
                    
                    # Remove padding
                    pred_seq = pred_seq[target_seq != pad_token]
                    target_seq = target_seq[target_seq != pad_token]
                    
                    all_predictions.append(pred_seq)
                    all_targets.append(target_seq)
                    
                    if return_details:
                        all_images.append(images[i].cpu())
                        all_labels.append(labels[i].cpu())
                    
                    samples_processed += 1
        
        print(f"Processed {samples_processed} samples")
        
        if return_details:
            return all_predictions, all_targets, all_images, all_labels
        return all_predictions, all_targets
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate various evaluation metrics
        """
        exact_match = 0
        token_accuracy = 0
        total_tokens = 0
        sequence_lengths = []
        
        for pred, target in zip(predictions, targets):
            # Exact sequence match
            if len(pred) == len(target) and np.array_equal(pred, target):
                exact_match += 1
            
            # Token-level accuracy
            min_len = min(len(pred), len(target))
            if min_len > 0:
                token_correct = np.sum(pred[:min_len] == target[:min_len])
                token_accuracy += token_correct
                total_tokens += min_len
            
            sequence_lengths.append(len(target))
        
        exact_match_rate = exact_match / len(predictions) if predictions else 0
        token_accuracy_rate = token_accuracy / total_tokens if total_tokens > 0 else 0
        
        return {
            'exact_match_rate': exact_match_rate,
            'token_accuracy': token_accuracy_rate,
            'avg_sequence_length': np.mean(sequence_lengths),
            'total_samples': len(predictions)
        }
    
    def visualize_comparison(self, predictions_iter, targets_iter, predictions_batch, targets_batch, num_samples=10):
        """
        Visualize both iterative and batch predictions vs ground truth for comparison
        """
        print("\n" + "="*100)
        print("PREDICTION COMPARISON: ITERATIVE vs BATCH (TEACHER FORCING)")
        print("="*100)
        
        for i in range(min(num_samples, len(predictions_iter))):
            print(f"\nSample {i+1}:")
            print(f"Ground Truth:    {targets_iter[i]}")
            print(f"Iterative:       {predictions_iter[i]}")
            print(f"Batch (TF):      {predictions_batch[i]}")
            
            iter_match = np.array_equal(predictions_iter[i], targets_iter[i])
            batch_match = np.array_equal(predictions_batch[i], targets_iter[i])
            
            print(f"Iterative Match: {iter_match}")
            print(f"Batch Match:     {batch_match}")
            
            # Check if predictions are the same between methods
            methods_match = np.array_equal(predictions_iter[i], predictions_batch[i])
            print(f"Methods Agree:   {methods_match}")
            print("-" * 60)


def main():
    # Configuration
    checkpoint_path = "weights/d512_h8_n1_bs512_lr0.0001_best.pth"
    data_dir = "/home/anurizada/Documents/processed_dataset"
    batch_size = 32
    
    # Initialize inference engine
    inferencer = InferenceEngine(checkpoint_path)
    
    # Load dataset
    dataset = BarLinkageDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Compare both methods
    print("Running iterative inference...")
    predictions_iter, targets_iter = inferencer.predict_iterative(dataloader, max_samples=10)
    
    print("\nRunning batch inference (teacher forcing)...")
    predictions_batch, targets_batch = inferencer.predict_batch(dataloader, max_samples=10)
    
    # Calculate metrics for both
    metrics_iter = inferencer.calculate_metrics(predictions_iter, targets_iter)
    metrics_batch = inferencer.calculate_metrics(predictions_batch, targets_batch)
    
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)
    print("\nIterative Prediction (Autoregressive):")
    for metric, value in metrics_iter.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nBatch Prediction (Teacher Forcing):")
    for metric, value in metrics_batch.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compare the two methods
    print("\n" + "="*100)
    print("METHOD COMPARISON SUMMARY")
    print("="*100)
    print(f"Exact Match Rate Difference: {metrics_batch['exact_match_rate'] - metrics_iter['exact_match_rate']:.4f}")
    print(f"Token Accuracy Difference:   {metrics_batch['token_accuracy'] - metrics_iter['token_accuracy']:.4f}")
    
    # Visualize comparison side by side
    inferencer.visualize_comparison(predictions_iter, targets_iter, predictions_batch, targets_batch, num_samples=15)

if __name__ == "__main__":
    main()