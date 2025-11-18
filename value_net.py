"""
Value Network for OFC.
Learns to estimate expected final score from a given state.
"""
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Neural network with multiple heads:
    - value_head: expected final score (regression)
    - foul_head: foul probability (classification via sigmoid on logits)
    - round0_head: auxiliary logits for round-0 placements
    - feas_head: feasibility proxy (classification logit)
    - action_head: optional action-quality head q(s,a) for batched (state, action) pairs
    """

    # Compact action encoding dimension (13 slots → 13 features)
    ACTION_ENC_DIM = 13
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        trunk_dim = hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, trunk_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(trunk_dim, 1)
        self.foul_head = nn.Linear(trunk_dim, 1)
        # Round-0 policy over 12 fixed candidate placements (logits)
        self.round0_head = nn.Linear(trunk_dim, 12)
        # Feasibility proxy head
        self.feas_head = nn.Linear(trunk_dim, 1)
        # Action-quality head q(s,a) over compact action encoding
        self.action_head = nn.Sequential(
            nn.Linear(trunk_dim + self.ACTION_ENC_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for board-judging heads.

        Returns:
            value: (B,1)
            foul_logit: (B,1)
            round0_logits: (B,12)
            feas_logit: (B,1)
        """
        h = self.trunk(x)
        value = self.value_head(h)
        foul_logit = self.foul_head(h)
        round0_logits = self.round0_head(h)
        feas_logit = self.feas_head(h)
        return value, foul_logit, round0_logits, feas_logit

    def forward_action(self, x: torch.Tensor, action_enc: torch.Tensor) -> torch.Tensor:
        """
        Compute q(s,a) for a batch of (state, action) pairs.

        Args:
            x: encoded states, shape (B, input_dim)
            action_enc: action encodings, shape (B, ACTION_ENC_DIM)

        Returns:
            q_sa: (B, 1) tensor of normalized action-quality predictions.
        """
        h = self.trunk(x)
        if action_enc.dim() != 2 or action_enc.shape[1] != self.ACTION_ENC_DIM:
            raise ValueError(
                f"action_enc must be (B,{self.ACTION_ENC_DIM}), "
                f"got {tuple(action_enc.shape)}"
            )
        ha = torch.cat([h, action_enc], dim=1)
        q_sa = self.action_head(ha)
        return q_sa

