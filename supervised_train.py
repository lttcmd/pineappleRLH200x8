"""
Supervised pretraining for the OFC value network.

We generate structurally valid boards using heuristics, score them via ofc_scoring/ofc_cpp,
and train only the value head to predict the final score.

Produces:
  - supervised_value_net.pth            (weights after supervised training)
  - value_net_checkpoint_ep0.pth        (so train.py can start RL from episode 0)

Usage:
  source .venv/bin/activate
  python supervised_train.py
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from ofc_env import OfcEnv, State
from ofc_scoring import score_board
from ofc_types import Card, Rank, Suit
from state_encoding import encode_state, get_input_dim
from value_net import ValueNet

try:
    import ofc_cpp as _CPP
except Exception:
    _CPP = None


def int_to_card(i: int) -> Card:
    rank = Rank((i // 4) + 2)
    suit = Suit(i % 4)
    return Card(rank, suit)


def build_valid_board(rng: random.Random) -> Tuple[State, float]:
    """
    Construct a heuristic board with a bias toward valid, royalty-capable layouts.
    Strategy:
      - Randomly sample disjoint five-card hands for bottom/middle and three cards for top.
      - Enforce bottom >= middle >= top ordering via quick validation.
      - Score using ofc_cpp if available, else Python scoring.
    """
    deck = list(range(52))
    rng.shuffle(deck)
    bottom_idx = deck[:5]
    middle_idx = deck[5:10]
    top_idx = deck[10:13]

    bottom = [int_to_card(i) for i in bottom_idx]
    middle = [int_to_card(i) for i in middle_idx]
    top = [int_to_card(i) for i in top_idx]

    # quick validity check
    sc, is_foul = score_board(bottom, middle, top)
    attempts = 0
    while is_foul and attempts < 10:
        rng.shuffle(deck)
        bottom_idx = deck[:5]
        middle_idx = deck[5:10]
        top_idx = deck[10:13]
        bottom = [int_to_card(i) for i in bottom_idx]
        middle = [int_to_card(i) for i in middle_idx]
        top = [int_to_card(i) for i in top_idx]
        sc, is_foul = score_board(bottom, middle, top)
        attempts += 1

    board = bottom + middle + top
    state = State(
        board=board,
        round=5,
        current_draw=[],
        deck=[],
        cards_placed_this_round=0,
    )

    if _CPP is not None:
        import numpy as np

        b = np.array(bottom_idx, dtype=np.int16)
        m = np.array(middle_idx, dtype=np.int16)
        t = np.array(top_idx, dtype=np.int16)
        sc_cpp, _ = _CPP.score_board_from_ints(b, m, t)
        sc = float(sc_cpp)

    return state, float(sc)


def build_dataset(num_samples: int, seed: int = 1234) -> List[Tuple[torch.Tensor, float]]:
    rng = random.Random(seed)
    samples = []
    with tqdm(range(num_samples), desc="Generating boards", unit="board") as pbar:
        for _ in pbar:
            state, score = build_valid_board(rng)
            encoded = encode_state(state)
            samples.append((encoded, score))
    return samples


def supervised_train(samples: List[Tuple[torch.Tensor, float]], epochs: int = 5, batch_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(get_input_dim(), hidden_dim=512).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    data = samples
    n = len(data)
    indices = list(range(n))

    for epoch in range(epochs):
        random.shuffle(indices)
        running = 0.0
        count = 0
        with tqdm(range(0, n, batch_size), desc=f"Supervised epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for start in pbar:
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                batch_states = torch.stack([data[i][0] for i in batch_idx])
                batch_scores = torch.tensor([data[i][1] for i in batch_idx], dtype=torch.float32).unsqueeze(1)

                batch_states = batch_states.to(device)
                batch_scores = batch_scores.to(device)

                optimizer.zero_grad()
                values, foul_logit, _, _ = model(batch_states)
                loss = mse(values, batch_scores)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running = 0.9 * running + 0.1 * loss.item()
                count += 1
                pbar.set_postfix({"loss": f"{running:.4f}"})

    torch.save(model.state_dict(), "supervised_value_net.pth")
    torch.save(model.state_dict(), "value_net_checkpoint_ep0.pth")
    print("Saved supervised_value_net.pth and value_net_checkpoint_ep0.pth")


def main():
    num_samples = 20000
    samples = build_dataset(num_samples)
    supervised_train(samples, epochs=5, batch_size=256)


if __name__ == "__main__":
    main()


