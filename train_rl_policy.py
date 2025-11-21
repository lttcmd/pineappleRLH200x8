"""
RL Training for OFC Policy Network.

Uses REINFORCE with baseline. Can optionally initialize from SFL dataset
or use SFL as a baseline for comparison.
"""

import argparse
import random
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import ofc_cpp
from ofc_env import OfcEnv, State, Action
from ofc_scoring import score_board
from rl_policy_net import RLPolicyNet
from state_encoding import encode_state


def compute_reward(state: State) -> float:
    """
    Compute final reward for a completed game state.
    
    Returns:
        -6.0 for fouls/incomplete boards
        score (>= 0) for valid boards with royalties
    """
    # Check if board is complete
    if not all(slot is not None for slot in state.board):
        return -6.0
    
    # Extract rows
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    
    # Score using canonical C++ scorer
    bottom_ints = np.array([c.to_int() if c is not None else -1 for c in bottom], dtype=np.int16)
    middle_ints = np.array([c.to_int() if c is not None else -1 for c in middle], dtype=np.int16)
    top_ints = np.array([c.to_int() if c is not None else -1 for c in top], dtype=np.int16)
    
    score, fouled = ofc_cpp.score_board_from_ints(bottom_ints, middle_ints, top_ints)
    
    if fouled:
        return -6.0
    return float(score)


def run_episode(
    env: OfcEnv,
    model: RLPolicyNet,
    device: torch.device,
) -> tuple[List[torch.Tensor], List[torch.Tensor], float]:
    """
    Run one episode with the current policy.
    
    Returns:
        log_probs: List of log probabilities for chosen actions
        entropies: List of policy entropies
        reward: Final scalar reward
    """
    state: State = env.reset()
    done = False
    
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    
    while not done:
        legal_actions: List[Action] = env.legal_actions(state)
        if not legal_actions:
            # No legal moves; treat as foul
            return log_probs, entropies, -6.0
        
        # Encode state
        state_enc = encode_state(state).to(device)  # (838,)
        
        # Encode each action as next-state encoding
        action_encodings = []
        for action in legal_actions:
            next_state, _, _ = env.step(state, action)
            next_enc = encode_state(next_state)
            action_encodings.append(next_enc)
        
        action_enc = torch.stack(action_encodings, dim=0).to(device)  # (num_actions, 838)
        
        # Get scores from policy
        scores = model(state_enc, action_enc)  # (num_actions,)
        
        # Sample action
        probs = F.softmax(scores, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Take action
        action = legal_actions[int(action_idx.item())]
        state, _reward, done = env.step(state, action)
        
        if done:
            reward = compute_reward(state)
            break
    
    return log_probs, entropies, reward


def train_rl_policy(
    output_path: str,
    episodes: int,
    lr: float,
    entropy_coef: float,
    baseline_momentum: float,
    device: torch.device,
    seed: int,
    init_from_sfl: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> None:
    """
    Train RL policy using REINFORCE with baseline.
    
    Args:
        output_path: Where to save the trained model
        episodes: Number of training episodes
        lr: Learning rate
        entropy_coef: Entropy regularization coefficient
        baseline_momentum: Momentum for moving-average baseline
        device: Torch device
        seed: Random seed
        init_from_sfl: Optional path to SFL dataset for supervised pre-training
    """
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Environment
    env = OfcEnv(soft_mask=False, use_cpp=False)
    
    # Model
    model = RLPolicyNet().to(device)
    
    # Optional: Load pre-trained checkpoint (from supervised training)
    if checkpoint:
        print(f"[train_rl_policy] Loading checkpoint: {checkpoint}")
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"[train_rl_policy] Loaded pre-trained weights from {checkpoint}")
        else:
            print(f"[train_rl_policy] WARNING: Checkpoint not found: {checkpoint}, starting from scratch")
    
    # Optional: Initialize from SFL dataset (supervised pre-training) - deprecated
    if init_from_sfl:
        print(f"[train_rl_policy] WARNING: --init-from-sfl is deprecated, use --checkpoint instead")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Moving baseline for variance reduction
    baseline: Optional[float] = None
    
    pbar = tqdm(range(episodes), desc="RL Training")
    for ep in pbar:
        log_probs, entropies, reward = run_episode(env, model, device)
        
        # Update baseline
        if baseline is None:
            baseline = reward
        else:
            baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
        
        # Compute advantage
        advantage = reward - baseline
        
        # Compute loss
        if log_probs:
            log_prob_tensor = torch.stack(log_probs)  # (T,)
            entropy_tensor = torch.stack(entropies)  # (T,)
            
            # REINFORCE loss: -advantage * log_prob
            policy_loss = -(advantage * log_prob_tensor).mean()
            
            # Entropy regularization: encourage exploration
            entropy_loss = -entropy_coef * entropy_tensor.mean()
            
            loss = policy_loss + entropy_loss
        else:
            loss = torch.tensor(0.0, device=device)
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({
            "reward": f"{reward:.2f}",
            "baseline": f"{baseline:.2f}",
            "adv": f"{advantage:.2f}",
        })
    
    # Save model
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[train_rl_policy] Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RL policy network for OFC using REINFORCE."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_policy.pth",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient.",
    )
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.9,
        help="Momentum for moving-average baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--init-from-sfl",
        type=str,
        default=None,
        help="Optional: Path to SFL dataset for supervised pre-training (deprecated, use --checkpoint instead).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: Path to pre-trained model checkpoint (from supervised training).",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_rl_policy] Using device: {device}")
    
    train_rl_policy(
        output_path=args.output,
        episodes=args.episodes,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        baseline_momentum=args.baseline_momentum,
        device=device,
        seed=args.seed,
        init_from_sfl=args.init_from_sfl,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()


