"""
Value Network for OFC.
Learns to estimate expected final score from a given state.
"""
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Neural network that estimates the value (expected final score) of a state."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        """
        Args:
            input_dim: Size of encoded state vector
            hidden_dim: Size of hidden layers (default 512 for better GPU utilization)
        """
        super().__init__()
        
        # Larger network with more layers for better GPU parallelization
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of encoded states, shape (batch_size, input_dim)
        
        Returns:
            Value estimates, shape (batch_size, 1)
        """
        return self.net(x)

