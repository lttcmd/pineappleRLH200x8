"""
Supervised training loop for PolicyNet using C++-generated imitation data.

Dataset shards are produced by collect_sfl_dataset_cpp.py and contain tensors:
  - "encoded": float32 [N, 838]
  - "labels": int64 [N]
  - "action_offsets": int64 [N + 1]
  - "action_encodings": float32 [sum_actions, 838]
"""

import argparse
import glob
import os
import warnings
from typing import Dict, Iterator, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from policy_net import PolicyNet

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`",
)


class HeuristicPolicyDataset(torch.utils.data.Dataset):
    """Loads shards into contiguous tensors (fits in RAM for ~100k samples)."""

    def __init__(self, shard_paths: Sequence[str]):
        if not shard_paths:
            raise ValueError("Dataset is empty. Check shard paths.")

        states: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        starts: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        action_chunks: List[torch.Tensor] = []

        action_offset = 0

        for path in shard_paths:
            shard = torch.load(path, map_location="cpu")
            encoded = torch.as_tensor(shard["encoded"], dtype=torch.float32)
            label = torch.as_tensor(shard["labels"], dtype=torch.long)
            offsets = torch.as_tensor(shard["action_offsets"], dtype=torch.long).reshape(-1)
            action_states = torch.as_tensor(shard["action_encodings"], dtype=torch.float32)

            diff = (offsets[1:] - offsets[:-1]).reshape(-1)
            start = (offsets[:-1] + action_offset).reshape(-1)

            valid = diff > 0
            if valid.any():
                states.append(encoded[valid])
                labels.append(label[valid])
                starts.append(start[valid])
                counts.append(diff[valid])
            action_chunks.append(action_states)

            action_offset += action_states.shape[0]

        if not states:
            raise ValueError("All shard entries had zero legal actions; dataset empty.")

        self.states = torch.cat(states, dim=0)
        self.labels = torch.cat(labels, dim=0)
        self.action_starts = torch.cat(starts, dim=0)
        self.action_counts = torch.cat(counts, dim=0)
        self.action_encodings = torch.cat(action_chunks, dim=0)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = int(self.action_starts[idx].item())
        count = int(self.action_counts[idx].item())
        actions = self.action_encodings[start : start + count]
        return {
            "state": self.states[idx],
            "label": self.labels[idx],
            "actions": actions,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    states = torch.stack([item["state"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    max_actions = max(item["actions"].shape[0] for item in batch)
    action_dim = batch[0]["actions"].shape[1] if max_actions > 0 else 0
    actions = torch.zeros(len(batch), max_actions, action_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_actions, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["actions"].shape[0]
        if n > 0:
            actions[i, :n] = item["actions"]
            mask[i, :n] = True
    return {"state": states, "actions": actions, "mask": mask, "labels": labels}


def train_policy_net(
    data_dir: str,
    output_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    grad_accum_steps: int,
    use_amp: bool,
    shuffle_shards: bool,
    prefetch_factor: int,
    seed: int,
):
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.pt")))
    if not shard_paths:
        raise ValueError(f"No shards found in {data_dir}")

    dataset = HeuristicPolicyDataset(shard_paths)
    total_examples = len(dataset)
    if total_examples == 0:
        raise ValueError("Shard set contains zero usable examples.")

    if seed is not None:
        torch.manual_seed(seed)

    model = PolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=use_amp and device.type == "cuda")
    accum = max(1, grad_accum_steps)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle_shards,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    print(f"Found {total_examples:,} samples across {len(shard_paths)} shards.")

    for epoch in range(epochs):
        dataloader = DataLoader(dataset, **loader_kwargs)

        running_loss = 0.0
        correct = 0
        seen = 0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(total=total_examples, desc=f"Epoch {epoch+1}/{epochs}", unit="samples")

        for batch in dataloader:
            state = batch["state"].to(device, non_blocking=pin_memory)
            actions = batch["actions"].to(device, non_blocking=pin_memory)
            mask = batch["mask"].to(device, non_blocking=pin_memory)
            labels = batch["labels"].to(device, non_blocking=pin_memory)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(state, actions)
                logits = logits.masked_fill(~mask, -1e9)
                loss = F.cross_entropy(logits, labels)

            batch_loss = loss.detach().float().item()
            loss = loss / accum
            scaler.scale(loss).backward()

            preds = logits.detach().argmax(dim=1)
            correct += (preds == labels).sum().item()
            batch_size_actual = labels.size(0)
            seen += batch_size_actual
            running_loss = 0.9 * running_loss + 0.1 * batch_loss
            steps += 1

            if steps % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.update(batch_size_actual)
            acc = correct / max(1, seen)
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "acc": f"{acc*100:.2f}%", "lr": f"{lr:.2e}"})

        # Flush leftover grads
        if steps % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        pbar.close()
        epoch_acc = correct / max(1, seen)
        print(f"Epoch {epoch+1}/{epochs}: accuracy={epoch_acc*100:.2f}%, avg_loss={running_loss:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"Saved PolicyNet to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PolicyNet via heuristic imitation.")
    parser.add_argument("--data_dir", type=str, default="data/sfl_cpp_dataset")
    parser.add_argument("--output", type=str, default="policy_net.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", help="Pin host batches for faster H2D copies.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp mixed precision.")
    parser.add_argument("--no_shuffle_shards", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Workers prefetch batches (if num_workers>0).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_policy_net(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.amp,
        shuffle_shards=not args.no_shuffle_shards,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


