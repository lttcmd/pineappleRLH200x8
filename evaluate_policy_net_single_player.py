import argparse
import random
from typing import List

import torch
from tqdm import tqdm

from ofc_env import OfcEnv, State, Action
from ofc_scoring import score_board
from policy_net_policy import PolicyNetPolicy


def play_single_episode(env: OfcEnv, policy: PolicyNetPolicy) -> float:
    """Play one single-player hand and return the final score."""
    state: State = env.reset()
    done = False

    while not done:
        legal_actions: List[Action] = env.legal_actions(state)
        if not legal_actions:
            # No legal moves; treat as zero-score hand
            break
        action = policy.choose_action(env, state, legal_actions)
        state, _reward, done = env.step(state, action)

    # Compute final score using Python scoring (foul-aware)
    if not all(slot is not None for slot in state.board):
        return 0.0
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    score, _fouled = score_board(bottom, middle, top)
    return float(score)


def evaluate_policy(
    checkpoint_path: str,
    episodes: int,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    env = OfcEnv(soft_mask=True, use_cpp=True)
    policy = PolicyNetPolicy(checkpoint_path=checkpoint_path, device=device)

    total_score = 0.0
    scores: List[float] = []

    for _ in tqdm(range(episodes), desc="Episodes"):
        score = play_single_episode(env, policy)
        total_score += score
        scores.append(score)

    avg_score = total_score / max(1, episodes)
    foul_count = sum(1 for s in scores if s <= -6.0)
    foul_rate = foul_count / max(1, episodes)

    print(f"Episodes: {episodes}")
    print(f"Average single-player score: {avg_score:.3f}")
    print(f"Foul rate (score <= -6): {foul_rate*100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PolicyNet in a single-player OFC setting."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PolicyNet checkpoint (.pth).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of single-player episodes to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (e.g. 'cuda' or 'cpu'). Defaults to auto.",
    )
    args = parser.parse_args()

    evaluate_policy(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()


