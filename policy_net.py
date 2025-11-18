"""
Policy network for OFC action selection.

This module defines a fresh policy architecture that:
  - Encodes the full board state (same 838-dim encoding used elsewhere).
  - For each candidate action, encodes the resulting next state (or the delta
    between next and current state) so the network sees the actual placements.
  - Combines the state and action embeddings to produce logits over actions.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state


def encode_state_vector(state: State) -> torch.Tensor:
    """
    Encode a State into the standard 838-dim tensor (float32).
    """
    return encode_state(state).float()


def build_action_state_encodings(
    env: OfcEnv,
    state: State,
    actions: List[Action],
    use_delta: bool = True,
) -> List[torch.Tensor]:
    """
    For each action, simulate the next state and return either:
      - The raw encoded next-state vector (if use_delta=False), or
      - The delta encoding (next_state_enc - current_state_enc) if use_delta=True.
    """
    if not actions:
        return []

    base_enc = encode_state_vector(state)
    encodings: List[torch.Tensor] = []
    for action in actions:
        next_state, _, _ = env.step(state, action)
        next_enc = encode_state_vector(next_state)
        if use_delta:
            encodings.append(next_enc - base_enc)
        else:
            encodings.append(next_enc)
    return encodings


class PolicyNet(nn.Module):
    """
    Policy network that scores candidate actions given a state.

    Args:
        input_dim: dimensionality of state/action encodings (default 838)
        state_hidden_dim: size of the state embedding
        action_hidden_dim: size of the action embedding
        joint_hidden_dim: size of the joint (state + action) embedding
    """

    def __init__(
        self,
        input_dim: int = 838,
        state_hidden_dim: int = 512,
        action_hidden_dim: int = 512,
        joint_hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(input_dim, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.ReLU(),
        )
        self.joint_head = nn.Sequential(
            nn.Linear(state_hidden_dim + action_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )

    def forward(
        self,
        state_enc: torch.Tensor,
        action_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state_enc: Tensor of shape (B, input_dim)
            action_enc: Tensor of shape (B, A, input_dim)

        Returns:
            logits: Tensor of shape (B, A) with unnormalized scores for each action.
        """
        if state_enc.ndim != 2:
            raise ValueError("state_enc must be (B, input_dim)")
        if action_enc.ndim != 3:
            raise ValueError("action_enc must be (B, A, input_dim)")
        if state_enc.shape[0] != action_enc.shape[0]:
            raise ValueError("Batch dimension mismatch between state_enc and action_enc")

        B, A, _ = action_enc.shape
        state_emb = self.state_encoder(state_enc)              # (B, state_hidden_dim)
        action_flat = action_enc.view(B * A, self.input_dim)
        action_emb = self.action_encoder(action_flat)          # (B*A, action_hidden_dim)
        action_emb = action_emb.view(B, A, -1)

        # Repeat state embedding for each action
        state_emb_expanded = state_emb.unsqueeze(1).expand(-1, A, -1)
        joint = torch.cat([state_emb_expanded, action_emb], dim=-1)  # (B, A, state+action_dim)

        logits = self.joint_head(joint).squeeze(-1)  # (B, A)
        return logits

    def score_actions(
        self,
        env: OfcEnv,
        state: State,
        actions: List[Action],
        device: Optional[torch.device] = None,
        use_delta: bool = True,
    ) -> torch.Tensor:
        """
        Convenience helper: given a Python State and list of actions, return logits tensor.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not actions:
            return torch.empty(0, device=device)

        base = encode_state_vector(state).unsqueeze(0).to(device)
        action_vecs = build_action_state_encodings(env, state, actions, use_delta=use_delta)
        action_tensor = torch.stack(action_vecs, dim=0).unsqueeze(0).to(device)
        logits = self.forward(base, action_tensor)
        return logits.squeeze(0)



