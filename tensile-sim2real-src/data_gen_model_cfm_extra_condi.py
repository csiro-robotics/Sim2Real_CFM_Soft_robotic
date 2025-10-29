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
from utils.data_funcs_tensile_sim2real_gen import read_json_file, load_processed_data, validate_and_clean_data
from flow_model.conditional_flow import ConditionalVectorField, WrappedConditionalVectorField

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


def train_cfm(model_exp, model_warp, exp_val_data_raw, exp_mean, exp_std, num_points=7):

    batch_size = 200
    exp_data_aug = augment_features(exp_val_data_raw).to(device)
    exp_data_norm = normalize_data(exp_data_aug, exp_mean, exp_std).to(device)
    c_exp = exp_data_norm[:, -5:]
    
    model_exp = model_exp.to(device)
    model_warp = model_warp.to(device)
    model_exp.eval()
    model_warp.eval()
    ot_sampler = OTPlanSampler(method="exact")
    FM = ConditionalFlowMatcher(sigma=0.01)
    model = ConditionalVectorField(feature_dim=15, condition_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    print("Training CFM Model...")

    for k in tqdm(range(10000)):

        idx = torch.randperm(exp_val_data_raw.shape[0])[:batch_size]
        c_exp_batch = exp_data_norm[idx, -5:]

        with torch.no_grad():
            x0_total = model_warp(exp_data_norm[idx, -5:])
            x1_total = model_exp(exp_data_norm[idx, -5:])
            x0 = x0_total
            x1 = x1_total

        optimizer.zero_grad()

        t, f_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)

        t = t.to(device)
        f_t = f_t.to(device)
        u_t = u_t.to(device)

        v_t = model(t, f_t, c_exp_batch)
        loss = F.mse_loss(v_t, u_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (k + 1) % 2000 == 0:
            print(f"CFM Epoch {k+1}: loss {loss.item():.4f}")

    model.eval()
    torch.save(model.state_dict(), "dataset_model/cfm_model_tensile_sim2real_gen.pth")
    print("CFM model saved as dataset_model/cfm_model_tensile_sim2real_gen.pth")

    try:
        wrapped_model = WrappedConditionalVectorField(model, c_exp[:100])
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
    exp_val_data_raw = validate_and_clean_data(load_processed_data(is_exp=True, validation=True), "exp Validation")
    warp_val_data_raw = validate_and_clean_data(load_processed_data(is_exp=False, validation=True), "Warp Validation")

    checkpoint_exp = torch.load("dataset_model_sim2real/predictor_exp_gen.pth", map_location=device)
    num_points = checkpoint_exp.get('max_points', 7)
    predictor_exp = ForcePredictor(input_dim=5, num_points=num_points)
    predictor_exp.load_state_dict(checkpoint_exp['model_state_dict'])
    exp_mean = checkpoint_exp['mean']
    exp_std = checkpoint_exp['std']

    checkpoint_warp = torch.load("dataset_model_sim2real/predictor_sim_gen.pth", map_location=device)
    predictor_warp = ForcePredictor(input_dim=5, num_points=num_points)
    predictor_warp.load_state_dict(checkpoint_warp['model_state_dict'])
    warp_mean = checkpoint_warp['mean']
    warp_std = checkpoint_warp['std']

    print("\nTraining CFM...")
    train_cfm(predictor_exp, predictor_warp, exp_val_data_raw, exp_mean, exp_std, num_points=num_points)

if __name__ == "__main__":
    main()