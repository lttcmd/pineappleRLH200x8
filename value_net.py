"""
Value Network for OFC.
Learns to estimate expected final score from a given state.
"""
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Neural network with two heads:
    - value_head: expected final score (regression)
    - foul_head: foul probability (classification via sigmoid on logits)
    """
    
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
    
    def forward(self, x: torch.Tensor):
        """
        Returns:
            value: (B,1)
            foul_logit: (B,1)
        """
        h = self.trunk(x)
        value = self.value_head(h)
        foul_logit = self.foul_head(h)
        return value, foul_logit

