"""
Analyze supervised training results and model performance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import ofc_cpp
from ofc_env import OfcEnv, State, Action
from rl_policy_net import RLPolicyNet
from state_encoding import encode_state


def evaluate_model(
    checkpoint_path: str,
    episodes: int = 2000,
    seed: int = 42,
    device: str = "cuda",
) -> Dict:
    """Evaluate a trained model and return statistics."""
    print(f"\n[Evaluating] {checkpoint_path}")
    
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    env = OfcEnv(soft_mask=False, use_cpp=False)
    model = RLPolicyNet().to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    model.eval()
    
    scores: List[float] = []
    foul_count = 0
    pass_count = 0
    royalty_count = 0
    royalty_scores: List[float] = []
    
    for _ in tqdm(range(episodes), desc="Playing episodes", leave=False):
        state: State = env.reset()
        done = False
        
        while not done:
            legal_actions: List[Action] = env.legal_actions(state)
            if not legal_actions:
                scores.append(-6.0)
                foul_count += 1
                break
            
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
                scores_tensor = model(state_enc, action_enc)
                action_idx = int(torch.argmax(scores_tensor).item())
            
            action = legal_actions[action_idx]
            state, _reward, done = env.step(state, action)
            
            if done:
                # Check if complete
                if not all(slot is not None for slot in state.board):
                    scores.append(-6.0)
                    foul_count += 1
                    break
                
                # Score using canonical C++ scorer
                bottom = [state.board[i] for i in range(5)]
                middle = [state.board[i] for i in range(5, 10)]
                top = [state.board[i] for i in range(10, 13)]
                
                bottom_ints = np.array([c.to_int() if c is not None else -1 for c in bottom], dtype=np.int16)
                middle_ints = np.array([c.to_int() if c is not None else -1 for c in middle], dtype=np.int16)
                top_ints = np.array([c.to_int() if c is not None else -1 for c in top], dtype=np.int16)
                
                score, fouled = ofc_cpp.score_board_from_ints(bottom_ints, middle_ints, top_ints)
                
                scores.append(float(score))
                if fouled or score <= -6.0:
                    foul_count += 1
                elif score == 0.0:
                    pass_count += 1
                else:
                    royalty_count += 1
                    royalty_scores.append(score)
                break
    
    n = len(scores)
    if n == 0:
        return None
    
    stats = {
        "checkpoint": str(checkpoint_path),
        "episodes": n,
        "fouls": foul_count,
        "passes": pass_count,
        "royalty_boards": royalty_count,
        "foul_rate": foul_count / n,
        "pass_rate": pass_count / n,
        "royalty_rate": royalty_count / n,
        "avg_reward": np.mean(scores),
        "avg_royalty": np.mean(royalty_scores) if royalty_scores else 0.0,
        "std_reward": np.std(scores),
        "min_reward": np.min(scores),
        "max_reward": np.max(scores),
    }
    
    return stats


def analyze_checkpoint(
    checkpoint_path: str,
    episodes: int = 2000,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Analyze a single checkpoint."""
    stats = evaluate_model(checkpoint_path, episodes, seed, device)
    
    if stats is None:
        print("Failed to evaluate model.")
        return
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Episodes: {stats['episodes']}")
    print()
    print("Outcomes:")
    print(f"  Fouls:     {stats['fouls']:5d} ({stats['foul_rate']*100:5.2f}%)")
    print(f"  Passes:    {stats['passes']:5d} ({stats['pass_rate']*100:5.2f}%)")
    print(f"  Royalties: {stats['royalty_boards']:5d} ({stats['royalty_rate']*100:5.2f}%)")
    print()
    print("Rewards:")
    print(f"  Average: {stats['avg_reward']:7.3f}")
    print(f"  Std Dev: {stats['std_reward']:7.3f}")
    print(f"  Min:     {stats['min_reward']:7.3f}")
    print(f"  Max:     {stats['max_reward']:7.3f}")
    if stats['royalty_boards'] > 0:
        print(f"  Avg Royalty (when >0): {stats['avg_royalty']:7.3f}")
    print("=" * 60)
    
    # Compare to SFL baseline
    print("\nFor comparison, SFL heuristic typically achieves:")
    print("  Foul rate: ~25-30%")
    print("  Pass rate: ~40-50%")
    print("  Royalty rate: ~20-30%")
    print("  Avg reward: ~-0.1 to +0.5")


def compare_checkpoints(
    checkpoint_paths: List[str],
    episodes: int = 2000,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Compare multiple checkpoints."""
    print("\n" + "=" * 80)
    print("COMPARING CHECKPOINTS")
    print("=" * 80)
    
    all_stats = []
    for ckpt_path in checkpoint_paths:
        stats = evaluate_model(ckpt_path, episodes, seed, device)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("No valid checkpoints to compare.")
        return
    
    # Print comparison table
    print(f"\n{'Checkpoint':<30} {'Foul%':<8} {'Pass%':<8} {'Royalty%':<10} {'Avg Reward':<12}")
    print("-" * 80)
    for stats in all_stats:
        name = Path(stats['checkpoint']).name
        print(f"{name:<30} {stats['foul_rate']*100:6.2f}%  {stats['pass_rate']*100:6.2f}%  "
              f"{stats['royalty_rate']*100:8.2f}%  {stats['avg_reward']:10.3f}")
    
    # Find best
    best = max(all_stats, key=lambda s: s['avg_reward'])
    print(f"\nBest model: {Path(best['checkpoint']).name}")
    print(f"  Avg Reward: {best['avg_reward']:.3f}")
    print(f"  Royalty Rate: {best['royalty_rate']*100:.2f}%")


def analyze_dataset(data_dir: str, max_shards: Optional[int] = None) -> None:
    """Analyze the dataset statistics."""
    from train_supervised_sfl import load_sfl_dataset
    
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    states, labels, action_offsets, action_encodings = load_sfl_dataset(data_dir, max_shards)
    
    print(f"Total states: {len(states):,}")
    print(f"Total actions: {len(action_encodings):,}")
    print(f"Avg actions per state: {len(action_encodings) / len(states):.2f}")
    
    # Check for corrupted labels
    corrupted = 0
    action_counts = []
    for i in range(len(states)):
        num_actions = action_offsets[i+1].item() - action_offsets[i].item()
        action_counts.append(num_actions)
        label = labels[i].item()
        if label < 0 or label >= num_actions:
            corrupted += 1
    
    print(f"\nLabel Statistics:")
    print(f"  Corrupted labels: {corrupted:,} ({corrupted/len(states)*100:.2f}%)")
    print(f"  Valid labels: {len(states)-corrupted:,} ({(len(states)-corrupted)/len(states)*100:.2f}%)")
    print(f"  Min actions per state: {min(action_counts)}")
    print(f"  Max actions per state: {max(action_counts)}")
    print(f"  Avg actions per state: {np.mean(action_counts):.2f}")
    
    # Label distribution
    unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
    print(f"\nLabel Distribution (first 10):")
    for label, count in zip(unique_labels[:10], counts[:10]):
        print(f"  Label {label}: {count:,} ({count/len(labels)*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training results and model performance."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to analyze (single model).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Paths to checkpoints to compare (multiple models).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset directory to analyze.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of episodes for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    print(f"Using device: {device}")
    
    if args.checkpoint:
        analyze_checkpoint(args.checkpoint, args.episodes, args.seed, device)
    elif args.compare:
        compare_checkpoints(args.compare, args.episodes, args.seed, device)
    elif args.dataset:
        analyze_dataset(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

