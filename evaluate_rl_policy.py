"""
Evaluate trained RL policy network.
"""

import argparse
import json
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm

import ofc_cpp
from ofc_env import OfcEnv, State, Action
from rl_policy_net import RLPolicyNet
from state_encoding import encode_state


def play_episode(env: OfcEnv, model: RLPolicyNet, device: torch.device) -> tuple[float, bool]:
    """
    Play one episode with the trained policy.
    
    Returns:
        score: Final score
        fouled: Whether the board was fouled
    """
    state: State = env.reset()
    done = False
    
    while not done:
        legal_actions: List[Action] = env.legal_actions(state)
        if not legal_actions:
            return -6.0, True
        
        # Encode state
        state_enc = encode_state(state).to(device)
        
        # Encode actions
        action_encodings = []
        for action in legal_actions:
            next_state, _, _ = env.step(state, action)
            next_enc = encode_state(next_state)
            action_encodings.append(next_enc)
        
        action_enc = torch.stack(action_encodings, dim=0).to(device)
        
        # Get scores and pick best action
        with torch.no_grad():
            scores = model(state_enc, action_enc)
            action_idx = int(torch.argmax(scores).item())
        
        action = legal_actions[action_idx]
        state, _reward, done = env.step(state, action)
        
        if done:
            # Check if complete
            if not all(slot is not None for slot in state.board):
                return -6.0, True
            
            # Score using canonical C++ scorer
            bottom = [state.board[i] for i in range(5)]
            middle = [state.board[i] for i in range(5, 10)]
            top = [state.board[i] for i in range(10, 13)]
            
            bottom_ints = np.array([c.to_int() if c is not None else -1 for c in bottom], dtype=np.int16)
            middle_ints = np.array([c.to_int() if c is not None else -1 for c in middle], dtype=np.int16)
            top_ints = np.array([c.to_int() if c is not None else -1 for c in top], dtype=np.int16)
            
            score, fouled = ofc_cpp.score_board_from_ints(bottom_ints, middle_ints, top_ints)
            
            if fouled:
                return -6.0, True
            return float(score), False


def evaluate_policy(
    checkpoint_path: str,
    episodes: int,
    seed: int,
    device: str,
) -> None:
    """Evaluate trained RL policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = OfcEnv(soft_mask=False, use_cpp=False)
    model = RLPolicyNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    
    scores: List[float] = []
    foul_count = 0
    pass_count = 0
    royalty_count = 0
    royalty_scores: List[float] = []
    
    for _ in tqdm(range(episodes), desc="Evaluating"):
        score, fouled = play_episode(env, model, device)
        scores.append(score)
        
        if fouled or score <= -6.0:
            foul_count += 1
        elif score == 0.0:
            pass_count += 1
        else:
            royalty_count += 1
            royalty_scores.append(score)
    
    n = max(1, episodes)
    avg_score = np.mean(scores)
    foul_rate = foul_count / n
    pass_rate = pass_count / n
    royalty_rate = royalty_count / n
    avg_royalty = np.mean(royalty_scores) if royalty_scores else 0.0
    
    stats = {
        "episodes": episodes,
        "fouls": foul_count,
        "passes": pass_count,
        "royalty_boards": royalty_count,
        "foul_rate": foul_rate,
        "pass_rate": pass_rate,
        "royalty_rate": royalty_rate,
        "avg_reward": avg_score,
        "avg_royalty": avg_royalty,
    }
    
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL policy network."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Defaults to auto.",
    )
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_policy(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()


