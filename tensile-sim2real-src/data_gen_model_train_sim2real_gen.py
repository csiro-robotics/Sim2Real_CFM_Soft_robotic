import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from flow_model.data_model import ForcePredictor
from utils.data_funcs_tensile_sim2real_gen import read_json_file, load_processed_data, validate_and_clean_data
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def train_predictor(model, data, optimizer, scheduler, epochs=3000, name=""):
    losses = []
    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training {name}"):
        optimizer.zero_grad()
        c = data[:, -5:]
        f_true = data[:, :15]
        f_pred = model(c)
        loss = F.mse_loss(f_pred, f_true)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (epoch + 1) % 500 == 0:
            print(f"{name} Epoch {epoch+1}: loss {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")

    print(f"{name} Training finished. Final loss: {losses[-1]:.4f}")
    return model


def augment_features(data):
    if data.shape[1] != 17:
        raise ValueError(f"Input data should have 17 columns, but got {data.shape[1]}")

    force = data[:, 0:1]
    positions = data[:, 1:15]
    disp = data[:, 15:16]
    speed = data[:, 16:17]

    disp_sq = disp ** 2
    speed_sq = speed ** 2
    disp_speed = disp * speed

    return torch.cat([force, positions, disp, speed, disp_sq, speed_sq, disp_speed], dim=1)


def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (data - mean) / std, mean, std


def evaluate_predictor(model_exp, model_sim, exp_val_data, sim_val_data, exp_mean, exp_std, sim_mean, sim_std, max_points=7):
    model_exp.eval()
    model_sim.eval()
    print("\n--- Evaluating Predictors ---")
    with torch.no_grad():
        c_exp = exp_val_data[:, -5:]
        f_true_exp = exp_val_data[:, :15]
        f_pred_exp = model_exp(c_exp)
        mse_exp = F.mse_loss(f_pred_exp, f_true_exp).item()
        mae_exp = torch.mean(torch.abs(f_pred_exp - f_true_exp)).item()

        c_sim = sim_val_data[:, -5:]
        f_true_sim = sim_val_data[:, :15]
        f_pred_sim = model_sim(c_sim)
        mse_sim = F.mse_loss(f_pred_sim, f_true_sim).item()
        mae_sim = torch.mean(torch.abs(f_pred_sim - f_true_sim)).item()

        print(f"Experiment Validation - MSE: {mse_exp:.4f}, MAE: {mae_exp:.4f}")
        print(f"Simulation Validation - MSE: {mse_sim:.4f}, MAE: {mae_sim:.4f}")

        exp_std_safe = torch.where(exp_std < 1e-6, torch.ones_like(exp_std), exp_std)
        sim_std_safe = torch.where(sim_std < 1e-6, torch.ones_like(sim_std), sim_std)

        f_true_exp_raw = f_true_exp * exp_std_safe[:15] + exp_mean[:15]
        f_pred_exp_raw = f_pred_exp * exp_std_safe[:15] + exp_mean[:15]
        f_true_sim_raw = f_true_sim * sim_std_safe[:15] + sim_mean[:15]
        f_pred_sim_raw = f_pred_sim * sim_std_safe[:15] + sim_mean[:15]

        c_exp_raw_disp_speed = c_exp[:, :2] * exp_std_safe[-5:-3] + exp_mean[-5:-3]
        c_sim_raw_disp_speed = c_sim[:, :2] * sim_std_safe[-5:-3] + sim_mean[-5:-3]

        exp_force_true = f_true_exp_raw[:, 0].cpu().numpy()
        exp_force_pred = f_pred_exp_raw[:, 0].cpu().numpy()
        exp_disp = c_exp_raw_disp_speed[:, 0].cpu().numpy()
        exp_positions_true = f_true_exp_raw[:, 1:].cpu().numpy().reshape(-1, max_points, 2)

        sim_force_true = f_true_sim_raw[:, 0].cpu().numpy()
        sim_force_pred = f_pred_sim_raw[:, 0].cpu().numpy()
        sim_disp = c_sim_raw_disp_speed[:, 0].cpu().numpy()
        sim_positions_true = f_true_sim_raw[:, 1:].cpu().numpy().reshape(-1, max_points, 2)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        axs[0].scatter(exp_disp, exp_force_true, alpha=0.6, label="Exp True", c='blue', s=10)
        axs[0].scatter(exp_disp, exp_force_pred, alpha=0.6, label="Exp Pred", c='lightblue', s=10, marker='x')
        axs[0].scatter(sim_disp, sim_force_true, alpha=0.6, label="Sim True", c='red', s=10)
        axs[0].scatter(sim_disp, sim_force_pred, alpha=0.6, label="Sim Pred", c='pink', s=10, marker='x')
        axs[0].set_xlabel("Displacement")
        axs[0].set_ylabel("Force")
        axs[0].set_title("Displacement vs Force (Exp vs Sim)")
        axs[0].legend()
        axs[0].grid(True)

        colors = plt.cm.viridis(np.linspace(0, 1, max_points))
        for i in range(max_points):
            axs[1].plot(exp_positions_true[:, i, 0], exp_positions_true[:, i, 1], color=colors[i], linestyle='-', alpha=0.7, label=f'Exp True Pt {i}' if i==0 else None)
            axs[1].plot(sim_positions_true[:, i, 0], sim_positions_true[:, i, 1], color=colors[i], linestyle='--', alpha=0.7, label=f'Sim True Pt {i}' if i==0 else None)
            axs[1].scatter(exp_positions_true[0, i, 0], exp_positions_true[0, i, 1], color=colors[i], marker='o', s=30, edgecolors='black')
            axs[1].scatter(sim_positions_true[0, i, 0], sim_positions_true[0, i, 1], color=colors[i], marker='s', s=30, edgecolors='black')

        axs[1].set_xlabel("X Position")
        axs[1].set_ylabel("Y Position")
        axs[1].set_title("True Position Trajectories (Exp vs Sim)")
        axs[1].legend(['Exp True', 'Sim True'])
        axs[1].grid(True)
        axs[1].axis('equal')

        plt.tight_layout()
        plt.savefig("predictor_evaluation_sim2real.png")
        print("Evaluation image saved to predictor_evaluation_sim2real.png")
        plt.show()
        plt.close()

    model_exp.train()
    model_sim.train()


def main():
    exp_data_raw = validate_and_clean_data(load_processed_data(is_exp=True), "exp Train")
    warp_data_raw = validate_and_clean_data(load_processed_data(is_exp=False), "Warp Train")
    exp_val_data_raw = validate_and_clean_data(load_processed_data(is_exp=True, validation=True), "exp Validation")
    warp_val_data_raw = validate_and_clean_data(load_processed_data(is_exp=False, validation=True), "Warp Validation")

    exp_data_aug = augment_features(exp_data_raw)
    warp_data_aug = augment_features(warp_data_raw)
    exp_val_data_aug = augment_features(exp_val_data_raw)
    warp_val_data_aug = augment_features(warp_val_data_raw)

    exp_data, exp_mean, exp_std = normalize_data(exp_data_aug)
    warp_data, warp_mean, warp_std = normalize_data(warp_data_aug)
    exp_val_data, _, _ = normalize_data(exp_val_data_aug, exp_mean, exp_std)
    warp_val_data, _, _ = normalize_data(warp_val_data_aug, warp_mean, warp_std)

    predictor_exp = ForcePredictor(input_dim=5, num_points=7)
    optimizer_exp = torch.optim.Adam(predictor_exp.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_exp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_exp, T_max=6000)
    print("\n--- Training Experiment Predictor ---")
    predictor_exp = train_predictor(predictor_exp, exp_data, optimizer_exp, scheduler_exp, epochs=6000, name="Exp")

    predictor_sim = ForcePredictor(input_dim=5, num_points=7)
    optimizer_sim = torch.optim.Adam(predictor_sim.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler_sim = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sim, T_max=3000)
    print("\n--- Training Simulation Predictor ---")
    predictor_sim = train_predictor(predictor_sim, warp_data, optimizer_sim, scheduler_sim, epochs=3000, name="Sim")

    evaluate_predictor(predictor_exp, predictor_sim, exp_val_data, warp_val_data, exp_mean, exp_std, warp_mean, warp_std, max_points=7)

    output_dir = "dataset_model_sim2real"
    os.makedirs(output_dir, exist_ok=True)

    exp_model_path = os.path.join(output_dir, "predictor_exp_gen.pth")
    torch.save({
        'model_state_dict': predictor_exp.state_dict(),
        'mean': exp_mean,
        'std': exp_std,
        'max_points': 7
    }, exp_model_path)
    print(f"Experiment predictor model saved to {exp_model_path}")

    sim_model_path = os.path.join(output_dir, "predictor_sim_gen.pth")
    torch.save({
        'model_state_dict': predictor_sim.state_dict(),
        'mean': warp_mean,
        'std': warp_std,
        'max_points': 7
    }, sim_model_path)
    print(f"Simulation predictor model saved to {sim_model_path}")

if __name__ == "__main__":
    main()