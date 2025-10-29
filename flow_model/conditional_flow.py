"""
Conditional Flow Model Components

This module contains neural network components for conditional flow matching,
including a conditional vector field and its wrapper class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ------------------------------
# Conditional Vector Field
# ------------------------------
class ConditionalVectorField(nn.Module):
    """
    A conditional vector field network that takes features, time, and conditions as input.
    
    Args:
        feature_dim (int): Dimension of the input features
        condition_dim (int): Dimension of the conditioning variables
    """
    def __init__(self, feature_dim=15, condition_dim=2):
        super().__init__()
        input_dim = feature_dim + 1 + condition_dim  # (x + t + c)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, feature_dim)
        )

    def forward(self, t, f, c, **kwargs):
        """
        Args:
            t: Scalar time or [batch_size] tensor
            f: Feature tensor [batch_size, feature_dim]
            c: Condition tensor [batch_size, condition_dim]
        
        Returns:
            Vector field output [batch_size, feature_dim]
        """
        batch_size = f.shape[0]

        # Convert t to tensor if it's not already one
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=f.device)

        # Reshape t to (batch_size, 1) format
        if t.dim() == 0:
            # t is a scalar, e.g., tensor(0.2)
            # Expand to (batch_size, 1)
            t = t.view(1, 1).repeat(batch_size, 1)
        elif t.dim() == 1:
            # t is 1D, e.g., [1] or [batch_size]
            if t.shape[0] == 1:
                # Single element, e.g., [0.2]
                # Repeat to batch_size
                t = t.repeat(batch_size).view(batch_size, 1)
            elif t.shape[0] == batch_size:
                # Already batch_size elements, e.g., [0.2, 0.2, ...]
                # Just reshape
                t = t.view(-1, 1)
            else:
                raise ValueError(f"Unexpected t shape {t.shape} vs batch_size {batch_size}")
        else:
            raise ValueError(f"Unexpected t dim {t.dim()}, t.shape={t.shape}")

        # Now concatenate: f [batch_size, feature_dim], 
        # t [batch_size, 1], c [batch_size, condition_dim]
        inputs = torch.cat([f, t, c], dim=-1)
        return self.net(inputs)




# ------------------------------
# Vector Field Wrapper Class
# ------------------------------

class WrappedConditionalVectorField(nn.Module):
    """
    Wrapper class for conditional vector field that fixes the condition.
    
    Args:
        vector_field: The conditional vector field model
        condition: Fixed condition tensor to use for all forward passes
    """
    def __init__(self, vector_field, condition):
        super().__init__()
        self.vector_field = vector_field
        self.condition = condition

    def forward(self, t, x, **kwargs):
        """Forward pass with fixed condition."""
        return self.vector_field(t, x, self.condition)