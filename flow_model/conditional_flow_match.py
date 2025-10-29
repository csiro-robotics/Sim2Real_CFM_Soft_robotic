import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
import torch.optim as optim
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher


class ConditionEncoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    def forward(self, c):
        return self.net(c)

# 矢量场模型（已定义）
class ConditionalMLP(nn.Module):
    def __init__(self, data_dim=1, context_dim=2, time_varying=True):
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim
        self.time_varying = time_varying
        input_size = data_dim + 1 + context_dim if time_varying else data_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim)
        )
    def forward(self, t, x, args=None):
        if not args:
            raise ValueError("Context must be passed as `args=(context,)`")
        context = args[0]
        t = t.unsqueeze(-1) if t.dim() < 2 else t
        net_input = torch.cat([x, t, context], dim=-1) if self.time_varying else torch.cat([x, context], dim=-1)
        return self.net(net_input)