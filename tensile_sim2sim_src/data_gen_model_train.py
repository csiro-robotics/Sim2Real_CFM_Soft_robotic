import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from flow_model.data_model import ForcePredictor
from utils.data_funcs import read_json_file, load_processed_data, validate_and_clean_data
from tqdm import tqdm


def train_predictor(model, data, optimizer, scheduler, epochs=3000, name=""):
    losses = []
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        c = data[:, -5:]
        f_true = data[:, :15]
        f_pred = model(c)
        loss = F.mse_loss(f_pred, f_true)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (epoch + 1) % 200 == 0:
            print(f"{name} Epoch {epoch+1}: loss {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, label=f"{name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{name} Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.close()
    return model


def augment_features(data):
    force = data[:, 0:1]
    positions = data[:, 1:-2]
    disp = data[:, -2:-1]
    speed = data[:, -1:]
    disp_sq = disp ** 2
    speed_sq = speed ** 2
    disp_speed = disp * speed
    return torch.cat([force, positions, disp, speed, disp_sq, speed_sq, disp_speed], dim=1)

def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = data.mean(dim=0)
        std = data.std(dim=0)
    return (data - mean) / (std + 1e-6), mean, std

def evaluate_predictor(model_sofa, model_warp, sofa_val_data, warp_val_data, sofa_mean, sofa_std, warp_mean, warp_std, max_points=7):
    model_sofa.eval()
    model_warp.eval()
    with torch.no_grad():
        c_sofa = sofa_val_data[:, -5:]
        f_true_sofa = sofa_val_data[:, :15]
        f_pred_sofa = model_sofa(c_sofa)
        mse_sofa = F.mse_loss(f_pred_sofa, f_true_sofa).item()
        mae_sofa = torch.mean(torch.abs(f_pred_sofa - f_true_sofa)).item()

        c_warp = warp_val_data[:, -5:]
        f_true_warp = warp_val_data[:, :15]
        f_pred_warp = model_warp(c_warp)
        mse_warp = F.mse_loss(f_pred_warp, f_true_warp).item()
        mae_warp = torch.mean(torch.abs(f_pred_warp - f_true_warp)).item()

        print(f"SOFA Validation - MSE: {mse_sofa:.4f}, MAE: {mae_sofa:.4f}")
        print(f"Warp Validation - MSE: {mse_warp:.4f}, MAE: {mae_warp:.4f}")

        f_true_sofa_raw = f_true_sofa * sofa_std[:15] + sofa_mean[:15]
        f_pred_sofa_raw = f_pred_sofa * sofa_std[:15] + sofa_mean[:15]
        c_sofa_raw = c_sofa[:, :2] * sofa_std[-5:-3] + sofa_mean[-5:-3]
        sofa_force_true = f_true_sofa_raw[:, 0].numpy()
        sofa_force_pred = f_pred_sofa_raw[:, 0].numpy()
        sofa_disp = c_sofa_raw[:, 0].numpy()
        sofa_positions = f_pred_sofa_raw[:, 1:].numpy().reshape(-1, max_points, 2)

        f_true_warp_raw = f_true_warp * warp_std[:15] + warp_mean[:15]
        f_pred_warp_raw = f_pred_warp * warp_std[:15] + warp_mean[:15]
        c_warp_raw = c_warp[:, :2] * warp_std[-5:-3] + warp_mean[-5:-3]
        warp_force_true = f_true_warp_raw[:, 0].numpy()
        warp_force_pred = f_pred_warp_raw[:, 0].numpy()
        warp_disp = c_warp_raw[:, 0].numpy()
        warp_positions = f_pred_warp_raw[:, 1:].numpy().reshape(-1, max_points, 2)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].scatter(sofa_disp, sofa_force_true, alpha=0.5, label="SOFA True", c='blue')
        axs[0].scatter(sofa_disp, sofa_force_pred, alpha=0.5, label="SOFA Pred", c='lightblue')
        axs[0].scatter(warp_disp, warp_force_true, alpha=0.5, label="Warp True", c='red')
        axs[0].scatter(warp_disp, warp_force_pred, alpha=0.5, label="Warp Pred", c='pink')
        axs[0].set_xlabel("Displacement")
        axs[0].set_ylabel("Force")
        axs[0].set_title("Displacement vs Force (SOFA vs Warp)")
        axs[0].legend()
        axs[0].grid(True)

        sofa_positions_true = f_true_sofa_raw[:, 1:].numpy().reshape(-1, max_points, 2)
        warp_positions_true = f_true_warp_raw[:, 1:].numpy().reshape(-1, max_points, 2)
        sofa_trajectory = sofa_positions_true.reshape(-1, 2)
        warp_trajectory = warp_positions_true.reshape(-1, 2)
        axs[1].plot(sofa_trajectory[:, 0], sofa_trajectory[:, 1], 'b-', alpha=0.5, label="SOFA True")
        axs[1].plot(warp_trajectory[:, 0], warp_trajectory[:, 1], 'r-', alpha=0.5, label="Warp True")
        axs[1].set_xlabel("X Position")
        axs[1].set_ylabel("Y Position")
        axs[1].set_title("Complete Position Trajectory (SOFA vs Warp - True Data)")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
        plt.close()

    model_sofa.train()
    model_warp.train()


def main():
    sofa_data_raw = validate_and_clean_data(load_processed_data(is_sofa=True), "SOFA Train")
    warp_data_raw = validate_and_clean_data(load_processed_data(is_sofa=False), "Warp Train")
    sofa_val_data_raw = validate_and_clean_data(load_processed_data(is_sofa=True, validation=True), "SOFA Validation")
    warp_val_data_raw = validate_and_clean_data(load_processed_data(is_sofa=False, validation=True), "Warp Validation")

    sofa_data_aug = augment_features(sofa_data_raw)
    warp_data_aug = augment_features(warp_data_raw)
    sofa_val_data_aug = augment_features(sofa_val_data_raw)
    warp_val_data_aug = augment_features(warp_val_data_raw)

    sofa_data, sofa_mean, sofa_std = normalize_data(sofa_data_aug)
    warp_data, warp_mean, warp_std = normalize_data(warp_data_aug)
    sofa_val_data, _, _ = normalize_data(sofa_val_data_aug, sofa_mean, sofa_std)
    warp_val_data, _, _ = normalize_data(warp_val_data_aug, warp_mean, warp_std)

    predictor_sofa = ForcePredictor(input_dim=5, num_points=7)
    optimizer_sofa = torch.optim.Adam(predictor_sofa.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_sofa = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sofa, T_max=3000)
    print("Training SOFA Predictor...")
    predictor_sofa = train_predictor(predictor_sofa, sofa_data, optimizer_sofa, scheduler_sofa, epochs=3000, name="SOFA")

    predictor_warp = ForcePredictor(input_dim=5, num_points=7)
    optimizer_warp = torch.optim.Adam(predictor_warp.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_warp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_warp, T_max=3000)
    print("Training Warp Predictor...")
    predictor_warp = train_predictor(predictor_warp, warp_data, optimizer_warp, scheduler_warp, epochs=3000, name="Warp")

    print("\nEvaluating Predictors...")
    evaluate_predictor(predictor_sofa, predictor_warp, sofa_val_data, warp_val_data, sofa_mean, sofa_std, warp_mean, warp_std, max_points=7)
    
    torch.save({
        'model_state_dict': predictor_sofa.state_dict(),
        'mean': sofa_mean,
        'std': sofa_std,
        'max_points': 7
    }, "dataset_model_sim2sim/predictor_sofa.pth")
    torch.save({
        'model_state_dict': predictor_warp.state_dict(),
        'mean': warp_mean,
        'std': warp_std,
        'max_points': 7
    }, "dataset_model_sim2sim/predictor_warp.pth")
    print("Models and normalization parameters saved in dataset_model_sim2sim/predictor_sofa.pth and predictor_warp.pth")

if __name__ == "__main__":
    main()