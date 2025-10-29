import os
import json
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "font.family": "Arial",
})

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from data_funcs import load_geometry_data, validate_and_clean_data
from contact_encoder import ContactEncoder, ContactDataProcessor
from data_gen_model_train import GeometryConditionedPredictor, create_geometry_mapping
from cfm_finray_train import ContactConditionedVectorField, normalize_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mm = 1 / 25.4
single_col_width = 85 * mm
height = 70 * mm

def validate_cfm_on_pointy2_with_predictor(base_dir="finray-sim2real_ext_models", device='cuda'):
    
    save_dir = Path(base_dir) / "cfm_models" / "pointy2_validation"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("CFM Validation on Held-Out Geometry: pointy2")
    print("Using GeometryConditionedPredictor for paired data")
    print("="*60)
    print(f"Device: {device}")
    print("="*60)
    
    print("\nStep 1: Loading trained CFM model...")
    
    model_path = Path(base_dir) / "cfm_models" / "cfm_sim2real.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ContactConditionedVectorField(
        feature_dim=checkpoint['feature_dim'],
        displacement_dim=checkpoint['displacement_dim'],
        contact_encoding_dim=checkpoint['contact_encoding_dim'],
        hidden_dim=512
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    sim_mean = checkpoint['sim_mean'].to(device)
    sim_std = checkpoint['sim_std'].to(device)
    exp_mean = checkpoint['exp_mean'].to(device)
    exp_std = checkpoint['exp_std'].to(device)
    disp_mean = checkpoint['disp_mean'].to(device)
    disp_std = checkpoint['disp_std'].to(device)
    
    print(f"✓ Model loaded from {model_path}")
    print(f"  Feature dim: {checkpoint['feature_dim']}")
    print(f"  Displacement dim: {checkpoint['displacement_dim']}")
    print(f"  Contact encoding dim: {checkpoint['contact_encoding_dim']}")
    
    print("\nStep 2: Loading contact encoder...")
    
    contact_processor = ContactDataProcessor(base_dir=base_dir)
    
    encoder_path = Path(base_dir) / "dataset_models" / "contact_encoder.pth"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Contact encoder not found at {encoder_path}")
    
    encoder_checkpoint = torch.load(encoder_path, map_location=device)
    contact_encoder = ContactEncoder(
        num_top_surface_nodes=encoder_checkpoint['num_top_surface_nodes'],
        bottleneck_dim=encoder_checkpoint['bottleneck_dim'],
        hidden_dims=encoder_checkpoint['hidden_dims']
    )
    contact_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    contact_encoder = contact_encoder.to(device)
    contact_encoder.eval()
    
    print(f"✓ Contact encoder loaded")
    
    print("\nStep 3: Loading GeometryConditionedPredictor...")
    
    predictor_path = Path(base_dir) / "dataset_models" / "geometry_conditioned_predictor.pth"
    if not predictor_path.exists():
        raise FileNotFoundError(f"Predictor not found at {predictor_path}")
    
    predictor_checkpoint = torch.load(predictor_path, map_location=device)
    
    name_to_idx, idx_to_name = create_geometry_mapping()
    
    predictor = GeometryConditionedPredictor(
        input_dim=predictor_checkpoint['input_dim'],
        num_points=predictor_checkpoint['num_points'],
        num_geometries=predictor_checkpoint['num_geometries'],
        hidden_dim=256,
        use_embedding=True
    )
    predictor.load_state_dict(predictor_checkpoint['model_state_dict'])
    predictor.to(device)
    predictor.eval()
    
    pred_input_mean = predictor_checkpoint['input_mean'].to(device)
    pred_input_std = predictor_checkpoint['input_std'].to(device)
    pred_output_mean = predictor_checkpoint['output_mean'].to(device)
    pred_output_std = predictor_checkpoint['output_std'].to(device)
    
    print(f"✓ Predictor loaded: {predictor_checkpoint['input_dim']} inputs → "
          f"{predictor_checkpoint['num_points']} points")
    
    print("\nStep 4: Loading pointy2 data and generating paired data...")
    
    geometry_data = load_geometry_data(
        base_dir=base_dir,
        geometries=['pointy2'],
        use_normalized_sim=True,
        remap_sim_contact=False
    )
    
    sim_data, exp_data_real = geometry_data['pointy2']
    
    sim_data = validate_and_clean_data(sim_data, "pointy2_sim")
    exp_data_real = validate_and_clean_data(exp_data_real, "pointy2_exp_real")
    
    print(f"  Sim samples: {sim_data.shape[0]}")
    print(f"  Real exp samples: {exp_data_real.shape[0]}")
    
    sim_force = sim_data[:, 0:1]
    sim_positions = sim_data[:, 1:-2]
    sim_disp = sim_data[:, -2:-1]
    sim_phase = sim_data[:, -1:]
    
    disp_sq = sim_disp ** 2
    disp_cube = sim_disp ** 3
    predictor_input = torch.cat([sim_disp, disp_sq, disp_cube, sim_phase], dim=1)
    
    predictor_input = predictor_input.to(device)
    predictor_input_norm = (predictor_input - pred_input_mean) / pred_input_std
    
    geom_idx = name_to_idx['pointy2']
    geom_labels = torch.full((sim_data.shape[0],), geom_idx, dtype=torch.long).to(device)
    
    print("  Generating experimental data with predictor...")
    with torch.no_grad():
        exp_output_norm = predictor(predictor_input_norm, geom_labels)
        exp_output = exp_output_norm * pred_output_std + pred_output_mean
        exp_output = exp_output.cpu()
    
    exp_force_pred = exp_output[:, 0:1]
    exp_positions_pred = exp_output[:, 1:]
    
    sim_features = torch.cat([sim_force, sim_positions], dim=1)
    exp_features_pred = torch.cat([exp_force_pred, exp_positions_pred], dim=1)
    
    exp_force_real = exp_data_real[:, 0:1]
    exp_positions_real = exp_data_real[:, 1:-2]
    exp_features_real = torch.cat([exp_force_real, exp_positions_real], dim=1)
    
    n_samples = min(sim_features.shape[0], exp_features_real.shape[0])
    sim_features = sim_features[:n_samples]
    exp_features_pred = exp_features_pred[:n_samples]
    exp_features_real = exp_features_real[:n_samples]
    
    print(f"  Using {n_samples} samples")
    
    disp_conditions = torch.cat([sim_disp[:n_samples], 
                                 disp_sq[:n_samples], 
                                 disp_cube[:n_samples], 
                                 sim_phase[:n_samples]], dim=1)
    
    print("\nStep 5: Loading contact information...")
    
    sim_json_path = Path(base_dir) / "sim_data" / "data_pointy2" / "pointy2_sim_normalized.json"
    
    if not sim_json_path.exists():
        raise FileNotFoundError(f"pointy2 simulation JSON not found: {sim_json_path}")
    
    contact_arrays = contact_processor.process_json_file(str(sim_json_path), use_remapping=True)
    contact_arrays = contact_arrays[:n_samples].to(device)
    
    with torch.no_grad():
        contact_encodings = contact_encoder(contact_arrays)
    
    print(f"  Contact encodings: {contact_encodings.shape}")
    
    print("\nStep 6: Running CFM transformation...")
    
    sim_features = sim_features.to(device)
    exp_features_pred = exp_features_pred.to(device)
    exp_features_real = exp_features_real.to(device)
    disp_conditions = disp_conditions.to(device)
    
    x0 = (sim_features - sim_mean) / sim_std
    x1_pred_target = (exp_features_pred - exp_mean) / exp_std
    x1_real = (exp_features_real - exp_mean) / exp_std
    disp_cond = (disp_conditions - disp_mean) / disp_std
    
    with torch.no_grad():
        t_span = torch.linspace(0, 1, 100).to(device)
        x_current = x0.clone()
        
        for i in tqdm(range(len(t_span) - 1), desc="CFM integration"):
            t_current = t_span[i]
            dt = t_span[i+1] - t_span[i]
            
            t_batch = torch.full((n_samples, 1), t_current.item(), device=device)
            
            v = model(t_batch, x_current, disp_cond, contact_encodings)
            
            x_current = x_current + v * dt
        
        x1_cfm = x_current
    
    print("✓ CFM transformation complete")
    
    print("\nStep 7: Processing results...")
    
    force_sim = (x0[:, 0] * sim_std[0, 0] + sim_mean[0, 0]).cpu().numpy()
    force_pred_target = (x1_pred_target[:, 0] * exp_std[0, 0] + exp_mean[0, 0]).cpu().numpy()
    force_cfm = (x1_cfm[:, 0] * exp_std[0, 0] + exp_mean[0, 0]).cpu().numpy()
    force_real = (x1_real[:, 0] * exp_std[0, 0] + exp_mean[0, 0]).cpu().numpy()
    
    disp_np = sim_disp[:n_samples].cpu().numpy().squeeze()
    
    force_sim = np.abs(force_sim)
    force_pred_target = np.abs(force_pred_target)
    force_cfm = np.abs(force_cfm)
    force_real = np.abs(force_real)
    
    print("\nStep 8: Calculating metrics...")
    
    mse_cfm_vs_pred = np.mean((force_cfm - force_pred_target) ** 2)
    mae_cfm_vs_pred = np.mean(np.abs(force_cfm - force_pred_target))
    
    mse_cfm_vs_real = np.mean((force_cfm - force_real) ** 2)
    mae_cfm_vs_real = np.mean(np.abs(force_cfm - force_real))
    
    mse_pred_vs_real = np.mean((force_pred_target - force_real) ** 2)
    mae_pred_vs_real = np.mean(np.abs(force_pred_target - force_real))
    
    mse_sim_vs_real = np.mean((force_sim - force_real) ** 2)
    mae_sim_vs_real = np.mean(np.abs(force_sim - force_real))
    
    print(f"\n{'='*60}")
    print("Validation Results on pointy2 (Held-Out Geometry)")
    print(f"{'='*60}")
    print(f"CFM → Predictor Target:")
    print(f"  MSE: {mse_cfm_vs_pred:.6f} N²")
    print(f"  MAE: {mae_cfm_vs_pred:.6f} N")
    print(f"\nCFM → Real Exp:")
    print(f"  MSE: {mse_cfm_vs_real:.6f} N²")
    print(f"  MAE: {mae_cfm_vs_real:.6f} N")
    print(f"\nPredictor → Real Exp:")
    print(f"  MSE: {mse_pred_vs_real:.6f} N²")
    print(f"  MAE: {mae_pred_vs_real:.6f} N")
    print(f"\nBaseline (Sim → Real):")
    print(f"  MSE: {mse_sim_vs_real:.6f} N²")
    print(f"  MAE: {mae_sim_vs_real:.6f} N")
    print(f"{'='*60}")
    
    print("\nStep 9: Creating visualizations...")
    
    disp_plot = disp_np
    force_sim_plot = force_sim
    force_pred_plot = force_pred_target
    force_cfm_plot = force_cfm
    force_real_plot = force_real

    time_plot = np.linspace(0, 10, disp_plot.shape[0])
    
    fig, ax = plt.subplots(figsize=(single_col_width, height))
    
    ax.plot(time_plot, force_sim_plot, ':', linewidth=1.5, alpha=0.6,
           label="PNCG-IPC", color='tab:green')
    ax.plot(time_plot, force_pred_plot, '-',linewidth=1.5, alpha=0.6,
           label="Predictor", color='tab:blue')
    ax.plot(time_plot, force_cfm_plot, '--', linewidth=1.5, alpha=0.7,
           label="CFM", color='tab:orange')

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Force (N)", fontsize=10)
    ax.set_title("pointy2: Force Comparison (Raw Data)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "pointy2_all_predictions.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "pointy2_all_predictions.pdf", bbox_inches="tight")
    plt.close(fig)
    
    print(f"  ✓ Saved: pointy2_all_predictions.png/pdf")
    
    print("\n  Extracting position data for trajectory analysis...")
    
    num_points = (x0.shape[1] - 1) // 2
    
    pos_sim = (x0[:, 1:] * sim_std[0, 1:] + sim_mean[0, 1:]).cpu().numpy()
    pos_pred = (x1_pred_target[:, 1:] * exp_std[0, 1:] + exp_mean[0, 1:]).cpu().numpy()
    pos_cfm = (x1_cfm[:, 1:] * exp_std[0, 1:] + exp_mean[0, 1:]).cpu().numpy()
    pos_real = (x1_real[:, 1:] * exp_std[0, 1:] + exp_mean[0, 1:]).cpu().numpy()
    
    pos_sim = pos_sim.reshape(n_samples, num_points, 2)
    pos_pred = pos_pred.reshape(n_samples, num_points, 2)
    pos_cfm = pos_cfm.reshape(n_samples, num_points, 2)
    pos_real = pos_real.reshape(n_samples, num_points, 2)
    
    print(f"  Number of tracking points: {num_points}")
    print(f"  Position shapes: {pos_sim.shape}")
    
    pos_errors_sim = np.sqrt(np.sum((pos_sim - pos_real) ** 2, axis=2))
    pos_errors_pred = np.sqrt(np.sum((pos_pred - pos_real) ** 2, axis=2))
    pos_errors_cfm = np.sqrt(np.sum((pos_cfm - pos_real) ** 2, axis=2))
    
    mean_error_sim = np.mean(pos_errors_sim, axis=0)
    mean_error_pred = np.mean(pos_errors_pred, axis=0)
    mean_error_cfm = np.mean(pos_errors_cfm, axis=0)
    
    mae_pos_sim = np.mean(pos_errors_sim)
    mae_pos_pred = np.mean(pos_errors_pred)
    mae_pos_cfm = np.mean(pos_errors_cfm)
    
    print("\n  Creating node trajectories comparison...")
    
    exp_true_color = 'tab:purple'
    exp_pred_color = 'tab:orange'
    predictor_color = 'tab:blue'
    sim_color = 'tab:green'
    
    fig, ax = plt.subplots(figsize=(single_col_width, height))
    
    step = max(1, n_samples // 20)
    
    for i in range(num_points):
        exp_traj_cfm = pos_cfm[::step, i, :]
        ax.plot(exp_traj_cfm[:, 0], exp_traj_cfm[:, 1], ':', 
               color=exp_pred_color, alpha=0.5, linewidth=0.8, marker='.', markersize=2,
               label="Exp Pred" if i == 0 else None)
        
        exp_traj_pred = pos_pred[::step, i, :]
        ax.plot(exp_traj_pred[:, 0], exp_traj_pred[:, 1], '-.', 
               color=predictor_color, alpha=0.4, linewidth=0.8,
               label="Exp True" if i == 0 else None)
        
        sim_traj = pos_sim[::step, i, :]
        ax.plot(sim_traj[:, 0], sim_traj[:, 1], '--', 
               color=sim_color, alpha=0.4, linewidth=0.8,
               label="PNCG-IPC" if i == 0 else None)
    
    ax.set_xlabel("X Position (mm)", fontsize=10)
    ax.set_ylabel("Y Position (mm)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.axis('equal')
    fig.tight_layout()
    fig.savefig(save_dir / "pointy2_node_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "pointy2_node_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)
    
    print(f"  ✓ Saved: pointy2_node_trajectories.png/pdf")
    
    print(f"  ✓ Saved: pointy2_validation_results_with_predictor.json")
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)
    print(f"Results saved to: {save_dir}")
    print("\nGenerated files:")
    print("  - pointy2_all_predictions.png/pdf")
    print("  - pointy2_node_trajectories.png/pdf")
    print("  - pointy2_validation_results_with_predictor.json")
    print("="*60)



if __name__ == "__main__":
    validate_cfm_on_pointy2_with_predictor(
        base_dir="finray-sim2real_ext_models",
        device=device
    )
    