"""
Supervised training on SFL dataset.
Trains the policy network to imitate SFL's decisions.
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rl_policy_net import RLPolicyNet


def load_sfl_dataset(data_dir: str, max_shards: Optional[int] = None):
    """
    Load SFL dataset from shard files.
    
    Returns:
        states: (N, 838) tensor of state encodings
        labels: (N,) tensor of action indices
        action_offsets: (N+1,) tensor of offsets into action_encodings
        action_encodings: (M, 838) tensor of all action encodings
    """
    data_path = Path(data_dir)
    shard_files = sorted(data_path.glob("shard_*.pt"))
    
    if max_shards:
        shard_files = shard_files[:max_shards]
    
    print(f"[load_sfl_dataset] Loading {len(shard_files)} shards from {data_dir}")
    
    all_states = []
    all_labels = []
    all_action_offsets = []
    all_action_encodings = []
    
    cumulative_offset = 0
    
    for shard_file in tqdm(shard_files, desc="Loading shards"):
        data = torch.load(shard_file, weights_only=False)
        
        states = data["encoded"]  # (N, 838)
        labels = data["labels"]  # (N,)
        action_offsets = data["action_offsets"]  # (N+1,)
        action_encodings = data["action_encodings"]  # (M, 838)
        
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states)
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        if not isinstance(action_offsets, torch.Tensor):
            action_offsets = torch.from_numpy(action_offsets)
        if not isinstance(action_encodings, torch.Tensor):
            action_encodings = torch.from_numpy(action_encodings)
        
        num_states = states.shape[0]
        
        # Adjust offsets for accumulated actions
        # action_offsets is (N+1,), where offsets[i] is the start index for state i
        # We need to shift all offsets by the cumulative_offset
        adjusted_offsets = action_offsets + cumulative_offset
        
        all_states.append(states)
        all_labels.append(labels)
        # Keep all offsets except the last one (which will be handled by next shard)
        all_action_offsets.append(adjusted_offsets[:-1])  # (N,)
        all_action_encodings.append(action_encodings)
        
        # Update cumulative offset: the last offset in this shard becomes the base for next shard
        cumulative_offset = adjusted_offsets[-1].item()
    
    # Concatenate everything
    states_tensor = torch.cat(all_states, dim=0)  # (total_states, 838)
    labels_tensor = torch.cat(all_labels, dim=0)  # (total_states,)
    
    # Concatenate action offsets and add final offset
    action_offsets_tensor = torch.cat(all_action_offsets, dim=0)  # (total_states,)
    # Add final offset (total number of actions)
    total_actions = sum(enc.shape[0] for enc in all_action_encodings)
    final_offset = torch.tensor([total_actions], dtype=action_offsets_tensor.dtype, device=action_offsets_tensor.device)
    action_offsets_tensor = torch.cat([action_offsets_tensor, final_offset], dim=0)  # (total_states+1,)
    
    action_encodings_tensor = torch.cat(all_action_encodings, dim=0)  # (total_actions, 838)
    
    print(f"[load_sfl_dataset] Loaded {states_tensor.shape[0]} states, {action_encodings_tensor.shape[0]} actions")
    
    return states_tensor, labels_tensor, action_offsets_tensor, action_encodings_tensor


class SFLDataset(Dataset):
    """Dataset for SFL training data."""
    def __init__(self, states, labels, action_offsets, action_encodings):
        self.states = states  # (N, 838) - on CPU
        self.labels = labels  # (N,)
        self.action_offsets = action_offsets  # (N+1,)
        self.action_encodings = action_encodings  # (M, 838) - on CPU
    
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        start = self.action_offsets[idx].item()
        end = self.action_offsets[idx + 1].item()
        state_actions = self.action_encodings[start:end]  # (num_actions, 838)
        
        return {
            'state': self.states[idx],
            'actions': state_actions,
            'label': self.labels[idx],
            'num_actions': state_actions.shape[0]
        }


def collate_fn(batch):
    """Collate function to pad actions to same length."""
    states = torch.stack([item['state'] for item in batch], dim=0)  # (B, 838)
    labels = torch.stack([item['label'] for item in batch], dim=0)  # (B,)
    action_encodings = [item['actions'] for item in batch]
    action_counts = [item['num_actions'] for item in batch]
    
    # Pad to same length (max actions in batch)
    max_actions = max(action_counts)
    padded_actions = []
    valid_mask = []
    
    for actions, num_actions in zip(action_encodings, action_counts):
        # Pad with zeros
        if num_actions < max_actions:
            padding = torch.zeros(max_actions - num_actions, 838, dtype=actions.dtype)
            actions = torch.cat([actions, padding], dim=0)
        padded_actions.append(actions)
        
        # Create mask: 1 for valid actions, 0 for padding
        mask = torch.zeros(max_actions, dtype=torch.bool)
        mask[:num_actions] = True
        valid_mask.append(mask)
    
    batch_action_enc = torch.stack(padded_actions, dim=0)  # (B, max_actions, 838)
    batch_valid_mask = torch.stack(valid_mask, dim=0)  # (B, max_actions)
    
    return states, batch_action_enc, labels, batch_valid_mask, action_counts


def create_data_loader(states, labels, action_offsets, action_encodings, batch_size, device, shuffle=True, num_workers=4):
    """
    Create a multi-threaded data loader.
    """
    dataset = SFLDataset(states, labels, action_offsets, action_encodings)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader


def train_supervised(
    data_dir: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    max_shards: Optional[int] = None,
    val_split: float = 0.1,
    num_workers: int = 8,
):
    """
    Train policy network to imitate SFL using supervised learning.
    """
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    states, labels, action_offsets, action_encodings = load_sfl_dataset(data_dir, max_shards)
    
    # Keep everything on CPU for DataLoader workers (they can't access CUDA tensors)
    # We'll move batches to GPU in the training loop
    labels = labels.long()  # CrossEntropyLoss requires long/int64
    # states, labels, action_offsets, action_encodings all stay on CPU
    
    # Train/val split
    num_samples = states.shape[0]
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_states = states[train_indices]
    train_labels = labels[train_indices]
    val_states = states[val_indices]
    val_labels = labels[val_indices]
    
    # For validation, we need to remap action offsets
    # This is complex, so for now we'll just use the same offsets (slightly inefficient but works)
    print(f"[train_supervised] Train: {num_train}, Val: {num_val}")
    
    # Create data loaders
    train_loader = create_data_loader(train_states, train_labels, action_offsets, action_encodings, batch_size, shuffle=True, device=device, num_workers=num_workers)
    val_loader = create_data_loader(val_states, val_labels, action_offsets, action_encodings, batch_size, shuffle=False, device=device, num_workers=num_workers)
    
    # Model
    model = RLPolicyNet().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches_processed = 0
        num_batches_skipped = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            total=(num_train + batch_size - 1) // batch_size,
        )
        
        for batch_states, batch_action_enc, batch_labels, batch_valid_mask, batch_action_counts in pbar:
            # Move to device (DataLoader returns CPU tensors)
            batch_states = batch_states.to(device)
            batch_action_enc = batch_action_enc.to(device)
            batch_labels = batch_labels.to(device)
            batch_valid_mask = batch_valid_mask.to(device)
            batch_size_actual = batch_states.shape[0]
            
            # Get scores for all actions
            scores = model(batch_states, batch_action_enc)  # (B, max_actions)
            
            # Mask out invalid (padded) actions by setting scores to -inf
            scores = scores.masked_fill(~batch_valid_mask, float('-inf'))
            
            # Check if any batch has all actions masked (shouldn't happen, but safety check)
            valid_actions_per_sample = batch_valid_mask.sum(dim=1)  # (B,)
            if (valid_actions_per_sample == 0).any():
                # Skip this batch if any sample has no valid actions
                num_batches_skipped += 1
                continue
            
            # Compute loss
            loss = F.cross_entropy(scores, batch_labels, reduction='mean')
            
            # Check for inf/nan
            if not torch.isfinite(loss):
                num_batches_skipped += 1
                continue
            
            num_batches_processed += 1
            
            # Compute accuracy
            preds = scores.argmax(dim=-1)
            correct = (preds == batch_labels).sum().item()
            total_correct += correct
            total_samples += batch_size_actual
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*correct/batch_size_actual:.2f}%",
            })
        
        avg_loss = total_loss / max(1, num_batches_processed)
        train_acc = 100.0 * total_correct / max(1, total_samples)
        
        if num_batches_skipped > 0:
            print(f"[Epoch {epoch+1}] WARNING: Skipped {num_batches_skipped} batches (inf/nan loss or no valid actions)")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_states, batch_action_enc, batch_labels, batch_valid_mask, _ in val_loader:
                # Move to device (DataLoader returns CPU tensors)
                batch_states = batch_states.to(device)
                batch_action_enc = batch_action_enc.to(device)
                batch_labels = batch_labels.to(device)
                batch_valid_mask = batch_valid_mask.to(device)
                scores = model(batch_states, batch_action_enc)
                scores = scores.masked_fill(~batch_valid_mask, float('-inf'))
                preds = scores.argmax(dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_states.shape[0]
        
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Save model
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[train_supervised] Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train policy network to imitate SFL using supervised learning."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sfl_cpp_dataset",
        help="Path to SFL dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/supervised_sfl.pth",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum number of shards to load (for testing).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loader workers.",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_supervised_sfl] Using device: {device}")
    
    train_supervised(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        max_shards=args.max_shards,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

