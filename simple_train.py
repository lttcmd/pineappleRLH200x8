"""
Simple foul-vs-non-foul pretraining for the OFC value network.

Goal:
  - Train the foul head to reliably distinguish fouled vs non-fouled episodes.
  - Ignore score magnitude/royalties for this phase.
  - Save weights in a checkpoint format that the main RL trainer (train.py)
    can resume from (episode 0).

Usage (on your Linux server, from repo root):
  source .venv/bin/activate
  python simple_train.py

This will create:
  - value_net_foul_pretrain.pth
  - value_net_checkpoint_ep0.pth  (so train.py can resume from this)
"""

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from state_encoding import get_input_dim
from value_net import ValueNet


try:
    import ofc_cpp as _CPP
except Exception as e:  # pragma: no cover - environment-dependent
    _CPP = None
    raise RuntimeError(
        "C++ module ofc_cpp is required for simple foul pretraining "
        "(simple_train.py). Make sure the extension is built."
    ) from e


def generate_foul_batch(
    num_episodes: int,
    base_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of encoded states and foul labels using C++ random episodes.

    For each episode:
      - If final score < 0 => foul_label = 1.0
      - Else               => foul_label = 0.0
    All states in an episode share the same label.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    encoded, offsets, scores = _CPP.generate_random_episodes(
        np.uint64(base_seed), int(num_episodes)
    )

    encoded_np = np.asarray(encoded, dtype=np.float32)
    offsets_np = np.asarray(offsets, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)

    if scores_np.shape[0] == 0:
        # No episodes generated; return empty tensors
        return (
            torch.empty((0, encoded_np.shape[1]), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )

    samples = []
    labels = []
    num_eps = scores_np.shape[0]
    if offsets_np.shape[0] != num_eps + 1:
        # Defensive: inconsistent offsets, bail out with empty batch
        return (
            torch.empty((0, encoded_np.shape[1]), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )

    for e in range(num_eps):
        s0 = int(offsets_np[e])
        s1 = int(offsets_np[e + 1])
        foul = 1.0 if scores_np[e] < 0.0 else 0.0
        for s in range(s0, s1):
            if s < encoded_np.shape[0]:
                samples.append(encoded_np[s])
                labels.append(foul)

    if not samples:
        return (
            torch.empty((0, encoded_np.shape[1]), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )

    x = torch.from_numpy(np.stack(samples).astype(np.float32))
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y


def main():
    # Config: you can adjust these if needed
    input_dim = get_input_dim()
    hidden_dim = 512
    lr = 1e-3
    total_iters = 3000          # number of pretraining iterations
    episodes_per_iter = 256     # C++ episodes per iteration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Foul pretraining device: {device}")

    model = ValueNet(input_dim, hidden_dim=hidden_dim).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # We expect fouls to be common; modest pos_weight to balance a bit
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], device=device))

    running_loss = 0.0
    running_foul_acc = 0.0
    running_nonfoul_acc = 0.0
    running_seen = 0

    start_time = time.time()
    pbar = tqdm(range(total_iters), desc="Foul pretraining", unit="iter")
    for it in pbar:
        seed = int(time.time() * 1_000_000) + it
        x, y = generate_foul_batch(episodes_per_iter, seed)
        if x.shape[0] == 0:
            continue

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        _, foul_logit, _, _ = model(x)
        loss = criterion(foul_logit, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(foul_logit)
            pred_labels = (preds >= 0.5).float()
            foul_mask = (y == 1.0)
            nonfoul_mask = (y == 0.0)
            if foul_mask.any():
                foul_acc = (pred_labels[foul_mask] == y[foul_mask]).float().mean().item()
            else:
                foul_acc = 0.0
            if nonfoul_mask.any():
                nonfoul_acc = (pred_labels[nonfoul_mask] == y[nonfoul_mask]).float().mean().item()
            else:
                nonfoul_acc = 0.0

        batch_size = x.shape[0]
        running_seen += batch_size
        alpha = 0.9
        if running_seen <= batch_size:
            running_loss = loss.item()
            running_foul_acc = foul_acc
            running_nonfoul_acc = nonfoul_acc
        else:
            running_loss = alpha * running_loss + (1 - alpha) * loss.item()
            running_foul_acc = alpha * running_foul_acc + (1 - alpha) * foul_acc
            running_nonfoul_acc = alpha * running_nonfoul_acc + (1 - alpha) * nonfoul_acc

        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "foul_acc": f"{running_foul_acc*100:.1f}%",
            "nonfoul_acc": f"{running_nonfoul_acc*100:.1f}%",
        })

    elapsed = time.time() - start_time
    print(f"\nFoul pretraining complete in {elapsed/60:.1f} minutes.")

    # Save foul-pretrained weights
    torch.save(model.state_dict(), "value_net_foul_pretrain.pth")
    print("Saved foul-pretrained weights to value_net_foul_pretrain.pth")

    # Also save as an RL-style checkpoint so train.py can resume from episode 0
    torch.save(model.state_dict(), "value_net_checkpoint_ep0.pth")
    print("Saved RL checkpoint at episode 0 to value_net_checkpoint_ep0.pth")


if __name__ == "__main__":
    main()


