"""
Clean RL Policy Network for OFC.

Simple architecture: takes state encoding (838-dim) and scores candidate actions.
Each action is encoded as the resulting next-state encoding.
"""

import torch
import torch.nn as nn
from typing import List

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state


class RLPolicyNet(nn.Module):
    """
    Simple policy network that scores candidate actions.
    
    Architecture:
    - State encoder: 838 -> 256 -> 128
    - Action scorer: (state_embedding, action_encoding) -> score
    """
    
    def __init__(self, state_dim: int = 838, hidden_dim: int = 512):
        super().__init__()
        # State embedding network - larger and deeper
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Action scoring: concatenate state embedding with action encoding
        # Larger network to handle variable action spaces
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim // 2 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state_enc: torch.Tensor, action_enc: torch.Tensor) -> torch.Tensor:
        """
        Score actions given a state.
        
        Args:
            state_enc: (batch_size, state_dim) or (state_dim,)
            action_enc: (batch_size, num_actions, state_dim) or (num_actions, state_dim)
        
        Returns:
            scores: (batch_size, num_actions) or (num_actions,)
        """
        # Handle single state case
        if state_enc.dim() == 1:
            state_enc = state_enc.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if action_enc.dim() == 2:
            action_enc = action_enc.unsqueeze(0)
            squeeze_action = True
        else:
            squeeze_action = False
        
        batch_size = state_enc.shape[0]
        num_actions = action_enc.shape[1]
        
        # Embed state: (batch_size, state_dim) -> (batch_size, hidden_dim//2)
        state_embed = self.state_net(state_enc)  # (batch_size, hidden_dim//2)
        
        # Expand state embedding for each action
        state_embed_expanded = state_embed.unsqueeze(1).expand(
            batch_size, num_actions, -1
        )  # (batch_size, num_actions, hidden_dim//2)
        
        # Concatenate state embedding with action encoding
        combined = torch.cat([state_embed_expanded, action_enc], dim=-1)
        # (batch_size, num_actions, hidden_dim//2 + state_dim)
        
        # Score each action
        scores = self.action_net(combined).squeeze(-1)  # (batch_size, num_actions)
        
        if squeeze_output or squeeze_action:
            scores = scores.squeeze(0)
        
        return scores
    
    def score_actions(
        self,
        env: OfcEnv,
        state: State,
        actions: List[Action],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convenience method: score a list of actions for a given state.
        
        Args:
            env: OFC environment
            state: Current game state
            actions: List of candidate actions
            device: Torch device
        
        Returns:
            scores: (num_actions,) tensor of action scores
        """
        if not actions:
            return torch.empty(0, device=device)
        
        # Encode current state
        state_enc = encode_state(state).to(device)  # (838,)
        
        # For each action, simulate next state and encode it
        action_encodings = []
        for action in actions:
            next_state, _, _ = env.step(state, action)
            next_enc = encode_state(next_state)
            action_encodings.append(next_enc)
        
        # Stack: (num_actions, 838)
        action_enc = torch.stack(action_encodings, dim=0).to(device)
        
        # Score actions
        with torch.no_grad():
            scores = self.forward(state_enc, action_enc)  # (num_actions,)
        
        return scores


