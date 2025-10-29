import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F


#--------------------------
# NN for data matching
#--------------------------


class ForcePredictor(nn.Module):
    def __init__(self, input_dim=5, num_points=7):  # 默认 7，与 selected_idx_set 一致
        super().__init__()
        self.output_dim = 1 + 2 * num_points  # [force, x1, y1, x2, y2, ...]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.output_dim)
        )
    
    def forward(self, c):
        return self.layers(c)
    


class FinRayForcePredictor(nn.Module):
    def __init__(self, input_dim=5, num_points=9, hidden_size=256, dropout_rate=0.2):
        super().__init__()
        self.output_dim = 1 + 2 * num_points  # [force, x1, y1, x2, y2, ...]
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_size, hidden_size*2, dropout_rate)
        self.res_block2 = ResidualBlock(hidden_size*2, hidden_size*4, dropout_rate)
        self.res_block3 = ResidualBlock(hidden_size*4, hidden_size*2, dropout_rate)
        self.res_block4 = ResidualBlock(hidden_size*2, hidden_size, dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, self.output_dim)
        
        # Force-specific branch
        self.force_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size//2, 1)
        )
        
        # Position-specific branch
        self.position_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 2 * num_points)
        )
        
    def forward(self, c):
        # Main branch
        x = self.input_embedding(c)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        main_output = self.output_layer(x)
        
        # Specialized branches (optional use)
        force_output = self.force_branch(c)
        position_output = self.position_branch(c)
        
        # Combine outputs - can be weighted based on validation performance
        # For now just use the main branch output
        return main_output

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        
        self.main_path = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate)
        )
        
        # Skip connection with dimension adjustment if needed
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.main_path(x) + self.skip_connection(x)