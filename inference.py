import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

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
            vocab_size=self.model_config['vocab_size'] + 1,
        ).to(device)
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model configuration: {self.model_config}")

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask

    def predict_iterative(self, dataloader, max_samples=100, top_k=10, temperature=1.0, use_top_k=True, return_details=False):
        """
        Generate predictions iteratively (token by token) using autoregressive decoding.
        Supports both greedy (argmax) and top-k sampling.
        """
        all_predictions, all_targets, all_images, all_labels = [], [], [], []
        samples_processed = 0

        sos_token, eos_token, pad_token = 0, 1, 2
        
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
                decoder_input = torch.full((batch_size, 1), sos_token, device=self.device)
                batch_predictions = [[] for _ in range(batch_size)]
                completed = [False] * batch_size

                # Autoregressive decoding
                for step in range(max_seq_len):
                    seq_len = decoder_input.shape[1]
                    causal_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)

                    # Forward pass
                    predictions, image_emb, label_emb = self.model(
                        decoder_input, causal_mask, images, labels
                    )

                    # Take logits for the last time step
                    next_logits = predictions[:, -1, :] / temperature
                    probs = F.softmax(next_logits, dim=-1)

                    if use_top_k:
                        # --- TOP-K SAMPLING ---
                        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                        next_token = torch.multinomial(topk_probs, num_samples=1)
                        next_token = topk_indices.gather(-1, next_token)
                        next_token = next_token.squeeze(1)
                    else:
                        # --- GREEDY DECODING ---
                        next_token = torch.argmax(probs, dim=-1)

                    # Append next token
                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

                    # Store predictions
                    for i in range(batch_size):
                        if not completed[i]:
                            batch_predictions[i].append(next_token[i].item())
                            if next_token[i].item() == eos_token or len(batch_predictions[i]) >= max_seq_len:
                                completed[i] = True

                    if all(completed):
                        break

                # Collect results
                for i in range(batch_size):
                    if samples_processed >= max_samples:
                        break

                    pred_seq = np.array(batch_predictions[i])
                    target_seq = target_tokens[i].cpu().numpy()
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

    def predict_single_sequence(self, image, label, max_length=25, top_k=10, temperature=1.0, use_top_k=True):
        """
        Predict a single sequence with top-k or greedy decoding.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(label.shape) == 1:
            label = label.unsqueeze(0)

        image = image.to(self.device)
        label = label.to(self.device)

        sos_token, eos_token = 0, 1
        decoder_input = torch.tensor([[sos_token]], dtype=torch.long, device=self.device)
        predicted_tokens = []

        with torch.no_grad():
            for step in range(max_length):
                seq_len = decoder_input.shape[1]
                causal_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)
                predictions, _, _ = self.model(decoder_input, causal_mask, image, label)

                next_logits = predictions[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)

                if use_top_k:
                    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                    next_token = torch.multinomial(topk_probs, num_samples=1)
                    next_token = topk_indices.gather(-1, next_token)
                    next_token = next_token.item()
                else:
                    next_token = torch.argmax(probs, dim=-1).item()

                predicted_tokens.append(next_token)

                if next_token == eos_token:
                    break

                decoder_input = torch.cat([
                    decoder_input,
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)

        return np.array(predicted_tokens)

    def predict_batch(self, dataloader, max_samples=100, return_details=False):
        """
        Original method using teacher forcing (for comparison)
        """
        all_predictions, all_targets, all_images, all_labels = [], [], [], []
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

                predictions, _, _ = self.model(decoder_input, decoder_mask, images, labels)
                pred_tokens = predictions.argmax(dim=-1)

                pad_token = 2
                for i in range(pred_tokens.shape[0]):
                    if samples_processed >= max_samples:
                        break
                        
                    pred_seq = pred_tokens[i].cpu().numpy()
                    target_seq = target_tokens[i].cpu().numpy()
                    
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
        exact_match, token_accuracy, total_tokens = 0, 0, 0
        sequence_lengths = []

        for pred, target in zip(predictions, targets):
            if len(pred) == len(target) and np.array_equal(pred, target):
                exact_match += 1

            min_len = min(len(pred), len(target))
            if min_len > 0:
                token_correct = np.sum(pred[:min_len] == target[:min_len])
                token_accuracy += token_correct
                total_tokens += min_len

            sequence_lengths.append(len(target))
        
        return {
            'exact_match_rate': exact_match / len(predictions) if predictions else 0,
            'token_accuracy': token_accuracy / total_tokens if total_tokens > 0 else 0,
            'avg_sequence_length': np.mean(sequence_lengths),
            'total_samples': len(predictions)
        }

    def visualize_comparison(self, predictions_iter, targets_iter, predictions_batch, targets_batch, num_samples=10):
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
            
            methods_match = np.array_equal(predictions_iter[i], predictions_batch[i])
            print(f"Methods Agree:   {methods_match}")
            print("-" * 60)


def main():
    checkpoint_path = "weights/d2048_h32_n6_bs1024_lr0.0001.pth"
    data_dir = "/home/anurizada/Documents/processed_dataset_105"
    batch_size = 10
    
    inferencer = InferenceEngine(checkpoint_path)
    dataset = BarLinkageDataset(data_dir=data_dir)
    dataset = torch.utils.data.Subset(dataset, range(1000000))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # --- Run iterative inference with Top-K ---
    print("Running iterative inference with top-k sampling...")
    predictions_iter, targets_iter = inferencer.predict_iterative(
        dataloader, max_samples=10, top_k=10, temperature=0.9, use_top_k=False
    )
    
    # --- Run batch inference (teacher forcing) ---
    print("\nRunning batch inference (teacher forcing)...")
    predictions_batch, targets_batch = inferencer.predict_batch(dataloader, max_samples=10)
    
    # --- Metrics ---
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
    
    print("\n" + "="*100)
    print("METHOD COMPARISON SUMMARY")
    print("="*100)
    print(f"Exact Match Rate Difference: {metrics_batch['exact_match_rate'] - metrics_iter['exact_match_rate']:.4f}")
    print(f"Token Accuracy Difference:   {metrics_batch['token_accuracy'] - metrics_iter['token_accuracy']:.4f}")
    
    inferencer.visualize_comparison(predictions_iter, targets_iter, predictions_batch, targets_batch, num_samples=15)


if __name__ == "__main__":
    main()
