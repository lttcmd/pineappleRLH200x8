"""
State encoding for neural network input.
Converts State objects into PyTorch tensors.
"""
import torch
import numpy as np
import os
_USE_CPP = os.getenv("OFC_USE_CPP", "1") == "1"
_CPP = None
if _USE_CPP:
    try:
        import ofc_cpp as _CPP
    except Exception:
        _CPP = None
from typing import Any, Optional
from ofc_env import State
from ofc_types import Card


def encode_state(state: State) -> torch.Tensor:
    """
    Encode state into a fixed-size tensor for neural network input.
    Optimized to pre-allocate tensor and use direct indexing.
    
    Returns:
        Tensor of shape (input_dim,) - flattened feature vector
    """
    # Pre-allocate entire tensor at once (much faster than cat)
    encoded = torch.zeros(838, dtype=torch.float32)
    
    # Encode board (13 slots) - direct indexing into pre-allocated tensor
    offset = 0
    for i in range(13):
        if state.board[i] is not None:
            encoded[offset + state.board[i].to_int()] = 1.0
        offset += 52
    
    # Encode current round (0-4) as one-hot
    if 0 <= state.round < 5:
        encoded[676 + state.round] = 1.0
    
    # Encode cards remaining in deck (normalized)
    encoded[681] = len(state.deck) / 52.0
    
    # Encode current draw (3 cards) - direct indexing
    offset = 682
    for i in range(min(3, len(state.current_draw))):
        encoded[offset + state.current_draw[i].to_int()] = 1.0
        offset += 52
    
    return encoded


def encode_state_batch(states,
                       boards: Optional[np.ndarray] = None,
                       rounds: Optional[np.ndarray] = None,
                       draws: Optional[np.ndarray] = None,
                       deck_sizes: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Encode multiple states at once for better performance.
    Ultra-optimized with numpy for speed.
    
    Args:
        states: List of State objects
    
    Returns:
        Tensor of shape (batch_size, input_dim)
    """
    batch_size = len(states)
    if _CPP is not None:
        if boards is None:
            boards = np.full((batch_size, 13), -1, dtype=np.int16)
        if rounds is None:
            rounds = np.zeros((batch_size,), dtype=np.int8)
        if draws is None:
            draws = np.full((batch_size, 3), -1, dtype=np.int16)
        if deck_sizes is None:
            deck_sizes = np.zeros((batch_size,), dtype=np.int16)
        for idx, state in enumerate(states):
            for i in range(13):
                if state.board[i] is not None:
                    boards[idx, i] = state.board[i].to_int()
            rounds[idx] = int(state.round)
            deck_sizes[idx] = len(state.deck)
            for i in range(min(3, len(state.current_draw))):
                draws[idx, i] = state.current_draw[i].to_int()
        encoded = _CPP.encode_state_batch_ints(boards, rounds, draws, deck_sizes)
        return torch.from_numpy(encoded)
    # Fallback Python/Numpy path
    batch_size = len(states)
    encoded_batch = np.zeros((batch_size, 838), dtype=np.float32)
    for idx, state in enumerate(states):
        offset = 0
        for i in range(13):
            if state.board[i] is not None:
                encoded_batch[idx, offset + state.board[i].to_int()] = 1.0
            offset += 52
        if 0 <= state.round < 5:
            encoded_batch[idx, 676 + state.round] = 1.0
        encoded_batch[idx, 681] = len(state.deck) / 52.0
        offset = 682
        for i in range(min(3, len(state.current_draw))):
            encoded_batch[idx, offset + state.current_draw[i].to_int()] = 1.0
            offset += 52
    return torch.from_numpy(encoded_batch)


def get_input_dim() -> int:
    """Get the input dimension for the value network."""
    # 13 slots * 52 (card encoding) = 676
    # + 5 (round one-hot) = 681
    # + 1 (deck ratio) = 682
    # + 3 * 52 (current draw) = 838
    return 13 * 52 + 5 + 1 + 3 * 52

