"""
Two-player simulator for Open Face Chinese Poker self-play.

Allows two policies to build boards independently from the same shuffled deck and
compares their final scores. Useful for evaluating whether a new policy outperforms
baseline placement strategies.
"""

import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state_batch, get_input_dim
from value_net import ValueNet


# -----------------------------------------------------------------------------
# Policy interfaces
# -----------------------------------------------------------------------------

class Policy:
    """Abstract policy interface."""

    def choose_action(self, env: OfcEnv, state: State, legal_actions: List[Action]) -> Action:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Baseline policy that selects uniformly from legal actions."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def choose_action(self, env: OfcEnv, state: State, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions available for random policy")
        return self.rng.choice(legal_actions)


class ValueNetPolicy(Policy):
    """
    Policy that loads a ValueNet checkpoint and chooses the action that maximizes
    value - penalty * foul_probability.
    """

    def __init__(self, checkpoint_path: str, penalty: float = 10.0, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ValueNet(get_input_dim(), hidden_dim=512).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.penalty = penalty

    def choose_action(self, env: OfcEnv, state: State, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions for ValueNetPolicy")

        # Simulate each action to generate next states
        next_states = []
        for action in legal_actions:
            next_state, _, _ = env.step(state, action)
            next_states.append(next_state)

        encoded = encode_state_batch(next_states).to(self.device)
        with torch.no_grad():
            values, foul_logit, _, _ = self.model(encoded)
            values = values.squeeze()
            foul_prob = torch.sigmoid(foul_logit).squeeze()
            combined = values - self.penalty * foul_prob

        idx = int(combined.argmax().item())
        return legal_actions[idx]


# -----------------------------------------------------------------------------
# Match simulation
# -----------------------------------------------------------------------------

@dataclass
class MatchResult:
    score_a: float
    score_b: float
    foul_a: bool
    foul_b: bool
    winner: int  # 1 if A wins, -1 if B wins, 0 tie


def simulate_match(
    policy_a: Policy,
    policy_b: Policy,
    seed: Optional[int] = None,
    max_steps: int = 50,
) -> MatchResult:
    rng = random.Random(seed)
    env = OfcEnv()
    state_a, state_b = env.reset_two_boards()
    done = False
    step = 0

    while not done and step < max_steps:
        legal_a = env.legal_actions(state_a)
        legal_b = env.legal_actions(state_b)
        if not legal_a or not legal_b:
            break
        action_a = policy_a.choose_action(env, state_a, legal_a)
        action_b = policy_b.choose_action(env, state_b, legal_b)
        state_a, state_b, done = env.step_two_boards(state_a, action_a, state_b, action_b)
        step += 1

    score_a = env.score(state_a)
    score_b = env.score(state_b)
    foul_a = score_a < 0
    foul_b = score_b < 0
    winner = 0
    if score_a > score_b:
        winner = 1
    elif score_b > score_a:
        winner = -1

    return MatchResult(
        score_a=score_a,
        score_b=score_b,
        foul_a=foul_a,
        foul_b=foul_b,
        winner=winner,
    )


def run_benchmark(
    checkpoint_path: Optional[str],
    matches: int,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    rng = random.Random(seed)
    policy_a = ValueNetPolicy(checkpoint_path, penalty=10.0) if checkpoint_path else RandomPolicy(rng)
    policy_b = RandomPolicy(rng)

    wins = 0
    losses = 0
    ties = 0
    foul_rate_a = 0
    foul_rate_b = 0
    avg_diff = 0.0

    for i in range(matches):
        result = simulate_match(policy_a, policy_b, seed=rng.randint(0, 1 << 30))
        if result.winner == 1:
            wins += 1
        elif result.winner == -1:
            losses += 1
        else:
            ties += 1
        foul_rate_a += 1 if result.foul_a else 0
        foul_rate_b += 1 if result.foul_b else 0
        avg_diff += (result.score_a - result.score_b)

    stats = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "foul_rate_a": foul_rate_a / matches,
        "foul_rate_b": foul_rate_b / matches,
        "avg_score_diff": avg_diff / matches,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Two-player OFC simulator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to ValueNet checkpoint for Player A (default: random policy)",
    )
    parser.add_argument("--matches", type=int, default=50, help="Number of matches to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    stats = run_benchmark(args.checkpoint, args.matches, seed=args.seed)
    print("Benchmark results:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


