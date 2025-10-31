import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE

from flow_model.data_model import ForcePredictor
from utils.data_funcs import read_json_file, load_processed_data, validate_and_clean_data
from flow_model.conditional_flow import ConditionalVectorField, WrappedConditionalVectorField

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

def plot_trajectories1(traj):
    n = 2000

    fig, ax = plt.subplots(figsize=(single_col_width, height))

    ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    ax.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")

    ax.legend(["Transformed data: z(S)", "Flow", "Original data: z(0)"], loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Flow Trajectories")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig("flow_trajectories.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.show()


mpl.rcParams.update({
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "font.family": "Arial",
})


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

def validate_cfm_full(model_warp, model_sofa, warp_val_data_raw, sofa_val_data_raw,
                      warp_mean, warp_std,
                      sofa_mean, sofa_std,
                      num_points=7,
                      cfm_model_path="dataset_model/cfm_model.pth"):
    
    global single_col_width, height
    
    if 'single_col_width' not in globals() or 'height' not in globals():
        mm = 1/25.4
        single_col_width = 85 * mm
        height = 70 * mm

    os.makedirs("figures", exist_ok=True)

    warp_data_aug = augment_features(warp_val_data_raw)
    warp_data_norm = normalize_data(warp_data_aug, warp_mean, warp_std)
    sofa_data_aug = augment_features(sofa_val_data_raw)
    sofa_data_norm = normalize_data(sofa_data_aug, sofa_mean, sofa_std)

    c_warp = warp_data_norm[:, -5:]

    model_warp.eval()
    model_sofa.eval()
    with torch.no_grad():
        x0_total = model_warp(warp_data_norm[:, -5:])
        x1_total = model_sofa(warp_data_norm[:, -5:])
    
    x0 = x0_total
    x1 = x1_total

    cfm_model = ConditionalVectorField(feature_dim=15, condition_dim=5)
    cfm_model.load_state_dict(torch.load(cfm_model_path, map_location='cpu'))
    cfm_model.eval()

    wrapped_model = WrappedConditionalVectorField(cfm_model, c_warp)
    node = NeuralODE(wrapped_model, solver="dopri5", sensitivity="adjoint")

    with torch.no_grad():
        t_span = torch.linspace(0, 1, 100)
        traj = node.trajectory(x0, t_span)
        
        try:
            n = min(2000, traj.shape[1])
            selected_traj = traj[:, :n, :].cpu()
            
            x_min = torch.min(selected_traj[:, :, 0]).item() - 0.1
            x_max = torch.max(selected_traj[:, :, 0]).item() + 0.1
            y_min = torch.min(selected_traj[:, :, 1]).item() - 0.1
            y_max = torch.max(selected_traj[:, :, 1]).item() + 0.1
            
            grid_size = 20
            x_grid = torch.linspace(x_min, x_max, grid_size)
            y_grid = torch.linspace(y_min, y_max, grid_size)
            X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
            
            grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
            
            time_points = [0.0, 0.5, 1.0]
            
            fig_velocity, axes = plt.subplots(1, 3, figsize=(single_col_width*3, height))
            
            for i, t in enumerate(time_points):
                ax = axes[i]
                grid_tensor = grid_points
                
                c_example = c_warp[0:1].repeat(grid_tensor.shape[0], 1)
                
                x_features = x0[0:1, :].repeat(grid_tensor.shape[0], 1)
                
                x_features[:, 0:2] = grid_tensor
                
                t_tensor = torch.ones(grid_tensor.shape[0]) * t
                velocities = cfm_model(x_features, t_tensor, c_example)
                
                U = velocities[:, 0].reshape(grid_size, grid_size).cpu()
                V = velocities[:, 1].reshape(grid_size, grid_size).cpu()
                
                speed = torch.sqrt(U**2 + V**2)
                
                X_np = X.cpu().numpy()
                Y_np = Y.cpu().numpy()
                U_np = U.cpu().numpy()
                V_np = V.cpu().numpy()
                speed_np = speed.cpu().numpy()
                selected_traj_np = selected_traj.cpu().numpy()
                
                quiver = ax.quiver(X_np, Y_np, U_np, V_np, speed_np, cmap='coolwarm', scale=50.0, width=0.002, pivot='mid')
                
                if i == 0:
                    ax.scatter(selected_traj_np[0, :, 0], selected_traj_np[0, :, 1], s=3, alpha=0.7, c="black", label="Prior z(0)")
                elif i == 1:
                    midpoint_idx = len(t_span) // 2
                    ax.scatter(selected_traj_np[0, :, 0], selected_traj_np[0, :, 1], s=3, alpha=0.3, c="black")
                    ax.scatter(selected_traj_np[midpoint_idx, :, 0], selected_traj_np[midpoint_idx, :, 1], s=3, alpha=0.7, c="blue", label="z(0.5)")
                else:
                    ax.scatter(selected_traj_np[0, :, 0], selected_traj_np[0, :, 1], s=3, alpha=0.3, c="black")
                    ax.scatter(selected_traj_np[-1, :, 0], selected_traj_np[-1, :, 1], s=3, alpha=0.7, c="red", label="Target z(1)")
                
                ax.set_title(f"t = {t}", fontsize=8)
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.legend(fontsize=6)
                ax.grid(True, alpha=0.3)
                ax.set_axisbelow(True)
            
            cbar = fig_velocity.colorbar(quiver, ax=axes.ravel().tolist())
            cbar.set_label("Velocity magnitude", fontsize=7)
            
            fig_velocity.suptitle("CFM Vector Field Visualization", fontsize=10)
            plt.tight_layout()
            
            fig_velocity.savefig("figures/cfm_vector_field.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
            fig_velocity.savefig("figures/cfm_vector_field.pdf", bbox_inches="tight", pad_inches=0.01)
            plt.close(fig_velocity)
        
        except Exception as e:
            print(f"Warning: Vector field visualization failed with error: {e}")
            print("Continuing with the rest of the validation...")

        n = min(2000, traj.shape[1])
        
        fig_traj, ax_traj = plt.subplots(figsize=(single_col_width, height))
        
        ax_traj.scatter(traj[0, :n, 0].cpu().numpy(), traj[0, :n, 1].cpu().numpy(), s=10, alpha=0.8, c="black", label="Warp")
        ax_traj.scatter(traj[:, :n, 0].cpu().numpy().reshape(-1), traj[:, :n, 1].cpu().numpy().reshape(-1), s=0.2, alpha=0.2, c="olive", label="Flow")
        ax_traj.scatter(traj[-1, :n, 0].cpu().numpy(), traj[-1, :n, 1].cpu().numpy(), s=4, alpha=1, c="blue", label="SOFA")
        
        ax_traj.legend(loc="best", fontsize=6)
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        ax_traj.set_title("Flow Trajectories")
        ax_traj.grid(False)
        
        fig_traj.tight_layout()
        fig_traj.savefig("figures/flow_trajectories.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
        fig_traj.savefig("figures/flow_trajectories.pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig_traj)

        x1_pred = traj[-1]
        x1_true = x1
        
        mse = F.mse_loss(x1_pred, x1_true).item()
        mae = torch.mean(torch.abs(x1_pred - x1_true)).item()
        
        force_mse = F.mse_loss(x1_pred[:, 0], x1_true[:, 0]).item()
        force_mae = torch.mean(torch.abs(x1_pred[:, 0] - x1_true[:, 0])).item()
        
        print(f"\n[Warp→Sofa] CFM Validation (All {warp_data_norm.shape[0]} Samples)")
        print(f"全局 MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"力值 MSE: {force_mse:.4f}, MAE: {force_mae:.4f}")

    force_pred_norm = x1_pred[:, 0]
    force_true_norm = x1_true[:, 0]
    force_warp_norm = x0[:, 0]

    force_pred_raw = force_pred_norm * sofa_std[0] + sofa_mean[0]
    force_true_raw = force_true_norm * sofa_std[0] + sofa_mean[0]
    force_warp_raw = force_warp_norm * warp_std[0] + warp_mean[0]

    disp_norm = c_warp[:, 0]
    disp_raw = disp_norm * warp_std[-5] + warp_mean[-5]
    
    force_pred_raw_np = force_pred_raw.detach().cpu().numpy()
    force_true_raw_np = force_true_raw.detach().cpu().numpy()
    force_warp_raw_np = force_warp_raw.detach().cpu().numpy()
    disp_raw_np = disp_raw.detach().cpu().numpy()
    time_raw_np = np.linspace(0, 110, len(force_pred_raw_np))
    
    fig, ax = plt.subplots(figsize=(single_col_width, height))

    ax.plot(time_raw_np, force_true_raw_np, '-', linewidth=1.5, alpha=0.8, label="SOFA True Force", color='tab:blue')
    ax.plot(time_raw_np, force_pred_raw_np, '--', linewidth=1.5, alpha=0.8, label="SOFA Pred Force", color='tab:orange')
    ax.plot(time_raw_np, force_warp_raw_np, ':', linewidth=1.5, alpha=0.8, label="WARP Force", color='tab:green')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("CFM Predicted Force vs. Displacement")
    ax.grid(True)
    ax.legend(loc="best", fontsize=6)

    fig.tight_layout()
    fig.savefig("cfm_force_vs_disp.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig("cfm_force_vs_disp.pdf",  bbox_inches="tight", pad_inches=0.01)

    plt.close(fig)
    
    def denormalize_positions(positions, mean, std):
        return positions * std[1:1+num_points*2] + mean[1:1+num_points*2]
    
    with torch.no_grad():
        sofa_positions_pred = denormalize_positions(x1_pred[:, 1:1+num_points*2], sofa_mean, sofa_std)
        sofa_positions_true = denormalize_positions(x1_true[:, 1:1+num_points*2], sofa_mean, sofa_std)
        warp_positions = denormalize_positions(x0[:, 1:1+num_points*2], warp_mean, warp_std)
        
        sofa_positions_pred = sofa_positions_pred.reshape(-1, num_points, 2).cpu().numpy()
        sofa_positions_true = sofa_positions_true.reshape(-1, num_points, 2).cpu().numpy()
        warp_positions = warp_positions.reshape(-1, num_points, 2).cpu().numpy()
    
    index_labels = [3, 12, 68, 88, 82, 105, 540]
    
    sofa_true_color = 'tab:blue'
    sofa_pred_color = 'tab:orange'
    warp_color = 'tab:green'
    
    mm = 1 / 25.4
    width = 85 * mm
    height = 70 * mm
    fig, ax = plt.subplots(figsize=(width, height))

    step = 40

    for i in range(num_points):
        sofa_traj_true = sofa_positions_true[::step, i, :]
        ax.plot(
            sofa_traj_true[:, 0],
            sofa_traj_true[:, 1],
            '-',
            color=sofa_true_color,
            alpha=0.7,
            label=f"SOFA True {index_labels[i]}" if i == 0 else None
        )
        
        sofa_traj_pred = sofa_positions_pred[::step, i, :]
        ax.plot(
            sofa_traj_pred[:, 0],
            sofa_traj_pred[:, 1],
            ':',
            color=sofa_pred_color,
            alpha=0.6,
            marker='.',
            markersize=3,
            label=f"SOFA Pred {index_labels[i]}" if i == 0 else None
        )
        
        warp_traj = warp_positions[::step, i, :]
        ax.plot(
            warp_traj[:, 0],
            warp_traj[:, 1],
            '--',
            color=warp_color,
            alpha=0.5,
            label=f"WARP {index_labels[i]}" if i == 0 else None
        )

    from matplotlib.lines import Line2D
    source_legend = [
        Line2D([0], [0], color=sofa_true_color, linestyle='-', linewidth=1.5),
        Line2D([0], [0], color=sofa_pred_color, linestyle=':', linewidth=1.5, marker='.', markersize=5),
        Line2D([0], [0], color=warp_color, linestyle='--', linewidth=1.5)
    ]
    ax.legend(source_legend, ['SOFA True', 'SOFA Pred', 'WARP'], loc='upper right')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Point Trajectories Comparison")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("cfm_point_trajectories.png", dpi=300, bbox_inches="tight", pad_inches=0.01) 
    fig.savefig("cfm_point_trajectories.pdf",  bbox_inches="tight", pad_inches=0.01)    
   
    plt.close(fig)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(min(num_points, len(axes))):
        ax = axes[i]
        sofa_pt_true = sofa_positions_true[:, i, :]
        ax.plot(sofa_pt_true[:, 0], sofa_pt_true[:, 1], '-', linewidth=1, color=sofa_true_color, alpha=0.7, label="SOFA True")
        
        sofa_pt_pred = sofa_positions_pred[:, i, :]
        ax.plot(sofa_pt_pred[:, 0], sofa_pt_pred[:, 1], '--', linewidth=1, color=sofa_pred_color, alpha=0.7, label="SOFA Pred")
        
        warp_pt = warp_positions[:, i, :]
        ax.plot(warp_pt[:, 0], warp_pt[:, 1], ':', linewidth=1, color=warp_color, alpha=0.6, label="WARP")
        
        ax.set_title(f"Point {index_labels[i]}", fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xlabel("X Position", fontsize=7)
        ax.set_ylabel("Y Position", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        if i == 0:
            ax.legend(fontsize=6)
    
    for i in range(num_points, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.savefig("cfm_individual_point_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    sofa_val_data_raw = validate_and_clean_data(load_processed_data(is_sofa=True, validation=True), "SOFA Validation")
    warp_val_data_raw = validate_and_clean_data(load_processed_data(is_sofa=False, validation=True), "Warp Validation")

    checkpoint_sofa = torch.load("dataset_model_sim2sim/predictor_sofa.pth")
    num_points = checkpoint_sofa.get('max_points', 7)
    predictor_sofa = ForcePredictor(input_dim=5, num_points=num_points)
    predictor_sofa.load_state_dict(checkpoint_sofa['model_state_dict'])
    sofa_mean = checkpoint_sofa['mean']
    sofa_std = checkpoint_sofa['std']

    checkpoint_warp = torch.load("dataset_model_sim2sim/predictor_warp.pth")
    predictor_warp = ForcePredictor(input_dim=5, num_points=num_points)
    predictor_warp.load_state_dict(checkpoint_warp['model_state_dict'])
    warp_mean = checkpoint_warp['mean']
    warp_std = checkpoint_warp['std']

    print("\nValidating CFM Model...")
    validate_cfm_full(predictor_warp, predictor_sofa, warp_val_data_raw, sofa_val_data_raw, warp_mean, warp_std, sofa_mean, sofa_std, num_points=num_points)

if __name__ == "__main__":
    main()