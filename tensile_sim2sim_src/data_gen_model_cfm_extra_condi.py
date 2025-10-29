import os
import json
import glob
import time
import math

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdyn
from torchdyn.core import NeuralODE

import ot as pot

from flow_model.data_model import ForcePredictor
from flow_model.conditional_flow import ConditionalVectorField, WrappedConditionalVectorField
from utils.data_funcs import read_json_file, load_processed_data, validate_and_clean_data
from torchcfm.optimal_transport import OTPlanSampler

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def augment_features(data):
    force = data[:, 0:1]
    positions = data[:, 1:-2]
    disp = data[:, -2:-1]
    speed = data[:, -1:]
    disp_sq = disp ** 2
    speed_sq = speed ** 2
    disp_speed = disp * speed
    return torch.cat([force, positions, disp, speed, disp_sq, speed_sq, disp_speed], dim=1)

def normalize_data(data, mean, std):
    return (data - mean) / (std + 1e-6)


def train_cfm(model_sofa, model_warp, sofa_val_data_raw, sofa_mean, sofa_std, num_points=7):

    batch_size = 200
    sofa_data_aug = augment_features(sofa_val_data_raw).to(device)
    sofa_data_norm = normalize_data(sofa_data_aug, sofa_mean, sofa_std).to(device)
    c_sofa = sofa_data_norm[:, -5:]
    
    model_sofa = model_sofa.to(device)
    model_warp = model_warp.to(device)
    model_sofa.eval()
    model_warp.eval()
    ot_sampler = OTPlanSampler(method="exact")
    FM = ConditionalFlowMatcher(sigma=0.01)
    model = ConditionalVectorField(feature_dim=15, condition_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    print("Training CFM Model...")

    for k in tqdm(range(10000)):

        idx = torch.randperm(sofa_val_data_raw.shape[0])[:batch_size]
        c_sofa_batch = sofa_data_norm[idx, -5:]

        with torch.no_grad():
            x0_total = model_warp(sofa_data_norm[idx, -5:])
            x1_total = model_sofa(sofa_data_norm[idx, -5:])
            x0 = x0_total
            x1 = x1_total

        optimizer.zero_grad()

        t, f_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)

        t = t.to(device)
        f_t = f_t.to(device)
        u_t = u_t.to(device)

        v_t = model(t, f_t, c_sofa_batch)
        loss = F.mse_loss(v_t, u_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (k + 1) % 2000 == 0:
            print(f"CFM Epoch {k+1}: loss {loss.item():.4f}")

    model.eval()
    torch.save(model.state_dict(), "dataset_model/cfm_model.pth")
    print("CFM model saved as dataset_model/cfm_model.pth")

    try:
        wrapped_model = WrappedConditionalVectorField(model, c_sofa[:100])
        node = NeuralODE(wrapped_model, solver="dopri5", sensitivity="adjoint")
        with torch.no_grad():
            traj = node.trajectory(x0[:100], t_span=torch.linspace(0, 1, 100))
            plot_trajectories(traj.cpu().numpy())

            x1_pred = traj[-1]
            x1_true = x1[:100]
            print("\nCFM Validation (First 10 Samples):")
            for i in range(10):
                print(f"Sample {i+1}: True x1 (force): {x1_true[i, 0]:.4f}, Predicted x1 (force): {x1_pred[i, 0]:.4f}")
    except Exception as e:
        print(f"Error during trajectory computation: {str(e)}")


def main():
    sofa_val_data_raw = validate_and_clean_data(load_processed_data(is_sofa=True), "SOFA Validation")
    warp_val_data_raw = validate_and_clean_data(load_processed_data(is_sofa=False), "Warp Validation")

    checkpoint_sofa = torch.load("dataset_model/predictor_sofa.pth", map_location=device)
    num_points = checkpoint_sofa.get('max_points', 7)
    predictor_sofa = ForcePredictor(input_dim=5, num_points=num_points)
    predictor_sofa.load_state_dict(checkpoint_sofa['model_state_dict'])
    sofa_mean = checkpoint_sofa['mean']
    sofa_std = checkpoint_sofa['std']

    checkpoint_warp = torch.load("dataset_model/predictor_warp.pth", map_location=device)
    predictor_warp = ForcePredictor(input_dim=5, num_points=num_points)
    predictor_warp.load_state_dict(checkpoint_warp['model_state_dict'])
    warp_mean = checkpoint_warp['mean']
    warp_std = checkpoint_warp['std']

    print("\nTraining CFM...")
    train_cfm(predictor_sofa, predictor_warp, sofa_val_data_raw, sofa_mean, sofa_std, num_points=num_points)

if __name__ == "__main__":
    main()