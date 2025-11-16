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
        values = model(encoded_batch).squeeze()
        
        # Find best action
        if values.dim() == 0:  # Single action
            best_idx = 0
        else:\n            best_idx = values.argmax().item()
        
        return legal_actions[best_idx]


def choose_best_action_beam_search(
    state: State,
    legal_actions: List[Action],
    model: ValueNet,
    env: OfcEnv,
    beam_width: int = 10,
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
        values = model(encoded_batch).squeeze()
        
        # Convert to list and pair with actions
        if values.dim() == 0:  # Single action
            return legal_actions[0]
        
        values_list = values.cpu().tolist()
        candidates = list(zip(values_list, legal_actions))
        
        # Sort by value (descending) and take top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return the best one
        return candidates[0][1]


def load_trained_model(model_path: str = 'value_net.pth', device: torch.device = None) -> ValueNet:
    """Load a trained value network from file."""
    if device is None:
        device = get_device()
    
    input_dim = get_input_dim()
    model = ValueNet(input_dim, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded model on device: {device}")
    return model

