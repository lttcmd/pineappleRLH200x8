"""
Action selection using trained value network.
"""
import torch
from typing import List

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state, encode_state_batch, get_input_dim
from value_net import ValueNet


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_best_action_with_value_net(
    state: State,
    legal_actions: List[Action],
    model: ValueNet,
    env: OfcEnv,
    device: torch.device = None
) -> Action:
    """
    Choose the best action using the trained value network.
    Optimized with batch encoding for faster performance.
    
    Args:
        state: Current game state
        legal_actions: List of legal actions
        model: Trained ValueNet
        env: OFC environment (for simulating actions)
        device: Device to run on (auto-detected if None)
    
    Returns:
        Best action according to value network
    """
    if not legal_actions:
        return None
    
    if device is None:
        device = get_device()
    
    model.eval()
    
    with torch.no_grad():
        # Simulate all actions and batch encode (much faster)
        next_states = [env.step(state, action)[0] for action in legal_actions]
        encoded_batch = encode_state_batch(next_states).to(device)
        # Model now returns (value, foul_logit, round0_logits, feas_logit)
        values, foul_logit, _, _ = model(encoded_batch)
        values = values.squeeze()
        foul_prob = torch.sigmoid(foul_logit).squeeze()
        # Foul-aware selection
        penalty = 8.0  # Default penalty weight
        combined = values - penalty * foul_prob
        
        # Find best action
        if combined.dim() == 0:  # Single action
            best_idx = 0
        else:
            best_idx = combined.argmax().item()
        
        return legal_actions[best_idx]


def choose_best_action_beam_search(
    state: State,
    legal_actions: List[Action],
    model: ValueNet,
    env: OfcEnv,
    beam_width: int = 10,
    gamma: float = 0.5,
    penalty: float = 8.0,
    device: torch.device = None
) -> Action:
    """
    Choose best action using beam search with value network.
    Optimized with batch encoding.
    
    Args:
        state: Current game state
        legal_actions: List of legal actions
        model: Trained ValueNet
        env: OFC environment
        beam_width: Number of top candidates to keep
        device: Device to run on (auto-detected if None)
    
    Returns:
        Best action from beam search
    """
    if not legal_actions:
        return None
    
    if device is None:
        device = get_device()
    
    model.eval()
    
    with torch.no_grad():
        # Batch process all actions (much faster)
        next_states = [env.step(state, action)[0] for action in legal_actions]
        encoded_batch = encode_state_batch(next_states).to(device)
        # Model returns (value, foul_logit, round0_logits, feas_logit)
        values, foul_logit, _, _ = model(encoded_batch)
        values = values.squeeze()
        foul_prob = torch.sigmoid(foul_logit).squeeze()
        combined = values - penalty * foul_prob
        
        # Convert to list and pair with actions
        if combined.dim() == 0:  # Single action
            return legal_actions[0]
        
        scores_list = combined.detach().cpu().tolist()
        candidates = list(zip(scores_list, legal_actions, next_states))
        
        # Sort by combined score (descending) and take top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:max(1, beam_width)]
        
        # Depth-2 lookahead: for each beam candidate, evaluate best next move
        best_total = None
        best_action = None
        for score1, act1, st1 in beam:
            la2 = env.legal_actions(st1)
            if not la2:
                total = score1
            else:
                st2_list = [env.step(st1, a2)[0] for a2 in la2]
                enc2 = encode_state_batch(st2_list).to(device)
                v2, f2, _, _ = model(enc2)
                v2 = v2.squeeze()
                f2 = torch.sigmoid(f2).squeeze()
                comb2 = v2 - penalty * f2
                best2 = comb2.max().item() if comb2.ndim > 0 else float(comb2.item())
                total = score1 + gamma * best2
            if (best_total is None) or (total > best_total):
                best_total = total
                best_action = act1
        
        return best_action if best_action is not None else legal_actions[0]


def load_trained_model(model_path: str = 'value_net.pth', device: torch.device = None) -> ValueNet:
    """Load a trained value network from file."""
    if device is None:
        device = get_device()
    
    input_dim = get_input_dim()
    model = ValueNet(input_dim, hidden_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded model on device: {device}")
    return model

