import os
import json
import time
import math

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdyn
from torchdyn.core import NeuralODE

from pathlib import Path
from tqdm import tqdm

from data_funcs import load_geometry_data, validate_and_clean_data
from contact_encoder import ContactEncoder, ContactDataProcessor
from data_gen_model_train import GeometryConditionedPredictor, create_geometry_mapping, augment_features_with_geometry

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.optimal_transport import OTPlanSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ContactConditionedVectorField(nn.Module):
    
    def __init__(self, feature_dim, displacement_dim=4, contact_encoding_dim=128, hidden_dim=512):
        super(ContactConditionedVectorField, self).__init__()
        
        self.feature_dim = feature_dim
        self.displacement_dim = displacement_dim
        self.contact_encoding_dim = contact_encoding_dim
        
        total_input_dim = 1 + feature_dim + displacement_dim + contact_encoding_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, t, x, displacement_condition, contact_encoding):
        batch_size = x.shape[0]

        if isinstance(t, (float, int)):
            t = torch.full((batch_size, 1), float(t), device=x.device, dtype=x.dtype)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(batch_size, 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.expand(batch_size).unsqueeze(1)
            elif t.shape[0] == batch_size:
                t = t.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected time shape {t.shape} for batch {batch_size}")
        elif t.dim() == 2:
            if t.shape[0] == 1:
                t = t.expand(batch_size, 1)
            elif t.shape[0] == batch_size and t.shape[1] == 1:
                pass
            else:
                raise ValueError(f"Time batch {t.shape[0]} != feature batch {batch_size}")
        else:
            raise ValueError(f"Unsupported time tensor shape {t.shape}")

        t = t.to(x.device, x.dtype)
        displacement_condition = displacement_condition.to(x.device, x.dtype)
        contact_encoding = contact_encoding.to(x.device, x.dtype)

        input_tensor = torch.cat([t, x, displacement_condition, contact_encoding], dim=1)
        return self.net(input_tensor)


class WrappedContactConditionedVectorField(nn.Module):
    
    def __init__(self, vector_field, displacement_condition, contact_encoding):
        super(WrappedContactConditionedVectorField, self).__init__()
        self.vector_field = vector_field
        self.displacement_condition = displacement_condition
        self.contact_encoding = contact_encoding
    
    def forward(self, t, x, args=None):
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        return self.vector_field(t, x, self.displacement_condition, self.contact_encoding)


def prepare_sim_exp_pairs(geometry_data, contact_processor, contact_encoder, 
                         geometries, name_to_idx, device):
    
    all_sim_features = []
    all_exp_features = []
    all_disp_conditions = []
    all_contact_encodings = []
    all_geom_labels = []
    
    print("\n" + "="*60)
    print("Preparing simulation-experiment pairs...")
    print("="*60)
    
    for geom_name in geometries:
        print(f"\nProcessing {geom_name}...")
        
        if geom_name not in geometry_data:
            print(f"  Skipping {geom_name}: not in geometry_data")
            continue
        
        sim_data, exp_data = geometry_data[geom_name]
        
        sim_data = validate_and_clean_data(sim_data, f"{geom_name}_sim")
        exp_data = validate_and_clean_data(exp_data, f"{geom_name}_exp")
        
        if sim_data.shape[0] == 0 or exp_data.shape[0] == 0:
            print(f"  Skipping {geom_name}: insufficient data")
            continue
        
        sim_disp = sim_data[:, -2].numpy()
        exp_disp = exp_data[:, -2].numpy()
        
        print(f"  Sim displacement range: [{sim_disp.min():.2f}, {sim_disp.max():.2f}]")
        print(f"  Exp displacement range: [{exp_disp.min():.2f}, {exp_disp.max():.2f}]")
        
        min_disp = max(sim_disp.min(), exp_disp.min())
        max_disp = min(sim_disp.max(), exp_disp.max())
        
        sim_mask = (sim_disp >= min_disp) & (sim_disp <= max_disp)
        exp_mask = (exp_disp >= min_disp) & (exp_disp <= max_disp)
        
        sim_data_filtered = sim_data[sim_mask]
        exp_data_filtered = exp_data[exp_mask]
        sim_disp_filtered = sim_disp[sim_mask]
        exp_disp_filtered = exp_disp[exp_mask]
        
        if sim_data_filtered.shape[0] == 0 or exp_data_filtered.shape[0] == 0:
            print(f"  Skipping {geom_name}: no overlapping displacement range")
            continue
        
        sim_sort_idx = np.argsort(sim_disp_filtered)
        exp_sort_idx = np.argsort(exp_disp_filtered)
        
        sim_data_sorted = sim_data_filtered[sim_sort_idx]
        exp_data_sorted = exp_data_filtered[exp_sort_idx]
        sim_disp_sorted = sim_disp_filtered[sim_sort_idx]
        exp_disp_sorted = exp_disp_filtered[exp_sort_idx]
        
        matched_exp_indices = []
        matched_sim_indices = []
        
        for i, s_disp in enumerate(sim_disp_sorted):
            distances = np.abs(exp_disp_sorted - s_disp)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < 0.5:
                matched_sim_indices.append(i)
                matched_exp_indices.append(nearest_idx)
        
        if len(matched_sim_indices) == 0:
            print(f"  Skipping {geom_name}: no close displacement matches found")
            continue
        
        matched_sim_data = sim_data_sorted[matched_sim_indices]
        matched_exp_data = exp_data_sorted[matched_exp_indices]
        
        min_samples = len(matched_sim_indices)
        print(f"  Using {min_samples} matched samples (displacement tolerance: 0.5mm)")
        
        matched_sim_disp = matched_sim_data[:, -2].numpy()
        matched_exp_disp = matched_exp_data[:, -2].numpy()
        disp_diff = np.abs(matched_sim_disp - matched_exp_disp)
        print(f"  Displacement matching error: mean={disp_diff.mean():.4f}, max={disp_diff.max():.4f}")
        
        sim_force = matched_sim_data[:, 0:1]
        sim_positions = matched_sim_data[:, 1:-2]
        sim_disp = matched_sim_data[:, -2:-1]
        sim_phase = matched_sim_data[:, -1:]
        
        exp_force = matched_exp_data[:, 0:1]
        exp_positions = matched_exp_data[:, 1:-2]
        
        sim_features = torch.cat([sim_force, sim_positions], dim=1)
        exp_features = torch.cat([exp_force, exp_positions], dim=1)
        
        disp_sq = sim_disp ** 2
        disp_cube = sim_disp ** 3
        disp_conditions = torch.cat([sim_disp, disp_sq, disp_cube, sim_phase], dim=1)
        
        base_path = Path("finray-sim2real_ext_models")
        sim_json_path = base_path / "sim_data" / f"data_{geom_name}" / f"{geom_name}_sim_normalized.json"
        
        if not sim_json_path.exists():
            print(f"  Warning: JSON not found for contact encoding: {sim_json_path}")
            contact_encodings = torch.zeros(min_samples, 128)
        else:
            contact_arrays = contact_processor.process_json_file(str(sim_json_path), use_remapping=True)
            
            contact_arrays_sorted = contact_arrays[sim_mask][sim_sort_idx]
            contact_arrays_matched = contact_arrays_sorted[matched_sim_indices].to(device)
            
            with torch.no_grad():
                contact_encodings = contact_encoder(contact_arrays_matched).cpu()
            
            print(f"  Contact encoding shape: {contact_encodings.shape}")
        
        geom_idx = name_to_idx[geom_name]
        geom_labels = torch.full((min_samples,), geom_idx, dtype=torch.long)
        
        all_sim_features.append(sim_features)
        all_exp_features.append(exp_features)
        all_disp_conditions.append(disp_conditions)
        all_contact_encodings.append(contact_encodings)
        all_geom_labels.append(geom_labels)
        
        print(f"  Sim features: {sim_features.shape}")
        print(f"  Exp features: {exp_features.shape}")
        print(f"  Displacement conditions: {disp_conditions.shape}")
        print(f"  Contact encodings: {contact_encodings.shape}")
    
    if not all_sim_features:
        raise ValueError("No valid simulation-experiment pairs found!")
    
    sim_features = torch.cat(all_sim_features, dim=0)
    exp_features = torch.cat(all_exp_features, dim=0)
    disp_conditions = torch.cat(all_disp_conditions, dim=0)
    contact_encodings = torch.cat(all_contact_encodings, dim=0)
    geom_labels = torch.cat(all_geom_labels, dim=0)
    
    print(f"\n{'='*60}")
    print(f"Combined dataset:")
    print(f"  Simulation features (x0): {sim_features.shape}")
    print(f"  Experimental features (x1): {exp_features.shape}")
    print(f"  Displacement conditions: {disp_conditions.shape}")
    print(f"  Contact encodings: {contact_encodings.shape}")
    print(f"  Geometry labels: {geom_labels.shape}")
    print(f"{'='*60}\n")
    
    return sim_features, exp_features, disp_conditions, contact_encodings, geom_labels


def prepare_sim_exp_pairs_with_predictor(geometry_data, predictor_checkpoint, 
                                         contact_processor, contact_encoder, 
                                         geometries, name_to_idx, device):
    
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
    
    input_mean = predictor_checkpoint['input_mean'].to(device)
    input_std = predictor_checkpoint['input_std'].to(device)
    output_mean = predictor_checkpoint['output_mean'].to(device)
    output_std = predictor_checkpoint['output_std'].to(device)
    
    all_sim_features = []
    all_exp_features = []
    all_disp_conditions = []
    all_contact_encodings = []
    all_geom_labels = []
    
    print("\n" + "="*60)
    print("Preparing simulation-experiment pairs using predictor...")
    print("="*60)
    
    for geom_name in geometries:
        print(f"\nProcessing {geom_name}...")
        
        if geom_name not in geometry_data:
            print(f"  Skipping {geom_name}: not in geometry_data")
            continue
        
        sim_data, _ = geometry_data[geom_name]
        
        sim_data = validate_and_clean_data(sim_data, f"{geom_name}_sim")
        
        if sim_data.shape[0] == 0:
            print(f"  Skipping {geom_name}: insufficient data")
            continue
        
        sim_force = sim_data[:, 0:1]
        sim_positions = sim_data[:, 1:-2]
        sim_disp = sim_data[:, -2:-1]
        sim_phase = sim_data[:, -1:]
        
        disp_sq = sim_disp ** 2
        disp_cube = sim_disp ** 3
        predictor_input = torch.cat([sim_disp, disp_sq, disp_cube, sim_phase], dim=1)
        
        predictor_input = predictor_input.to(device)
        
        predictor_input_norm = (predictor_input - input_mean) / input_std
        
        geom_idx = name_to_idx[geom_name]
        geom_labels = torch.full((sim_data.shape[0],), geom_idx, dtype=torch.long).to(device)
        
        with torch.no_grad():
            exp_output_norm = predictor(predictor_input_norm, geom_labels)
            
            exp_output = exp_output_norm * output_std + output_mean
            exp_output = exp_output.cpu()
        
        exp_force = exp_output[:, 0:1]
        exp_positions = exp_output[:, 1:]
        
        sim_features = torch.cat([sim_force, sim_positions], dim=1)
        exp_features = torch.cat([exp_force, exp_positions], dim=1)
        
        disp_conditions = torch.cat([sim_disp, disp_sq, disp_cube, sim_phase], dim=1)
        
        base_path = Path("finray-sim2real_ext_models")
        sim_json_path = base_path / "sim_data" / f"data_{geom_name}" / f"{geom_name}_sim_normalized.json"
        
        if not sim_json_path.exists():
            print(f"  Warning: JSON not found for contact encoding: {sim_json_path}")
            contact_encodings = torch.zeros(sim_data.shape[0], 128)
        else:
            contact_arrays = contact_processor.process_json_file(str(sim_json_path), use_remapping=True)
            contact_arrays = contact_arrays.to(device)
            
            with torch.no_grad():
                contact_encodings = contact_encoder(contact_arrays).cpu()
            
            print(f"  Contact encoding shape: {contact_encodings.shape}")
        
        all_sim_features.append(sim_features)
        all_exp_features.append(exp_features)
        all_disp_conditions.append(disp_conditions)
        all_contact_encodings.append(contact_encodings)
        all_geom_labels.append(torch.full((sim_data.shape[0],), geom_idx, dtype=torch.long))
        
        print(f"  Sim features: {sim_features.shape}")
        print(f"  Exp features (generated): {exp_features.shape}")
        print(f"  Displacement conditions: {disp_conditions.shape}")
        print(f"  Contact encodings: {contact_encodings.shape}")
    
    if not all_sim_features:
        raise ValueError("No valid simulation-experiment pairs found!")
    
    sim_features = torch.cat(all_sim_features, dim=0)
    exp_features = torch.cat(all_exp_features, dim=0)
    disp_conditions = torch.cat(all_disp_conditions, dim=0)
    contact_encodings = torch.cat(all_contact_encodings, dim=0)
    geom_labels = torch.cat(all_geom_labels, dim=0)
    
    print(f"\n{'='*60}")
    print(f"Combined dataset:")
    print(f"  Simulation features (x0): {sim_features.shape}")
    print(f"  Generated experimental features (x1): {exp_features.shape}")
    print(f"  Displacement conditions: {disp_conditions.shape}")
    print(f"  Contact encodings: {contact_encodings.shape}")
    print(f"  Geometry labels: {geom_labels.shape}")
    print(f"{'='*60}\n")
    
    return sim_features, exp_features, disp_conditions, contact_encodings, geom_labels


def normalize_features(features, mean=None, std=None):
    if mean is None:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std[std < 1e-6] = 1.0
    
    normalized = (features - mean) / std
    return normalized, mean, std


def train_cfm_sim2real(sim_features, exp_features, disp_conditions, contact_encodings,
                       num_epochs=10000, batch_size=128, lr=0.001, device='cuda',
                       save_dir="finray-sim2real_ext_models/cfm_models"):
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    sim_features = sim_features.to(device)
    exp_features = exp_features.to(device)
    disp_conditions = disp_conditions.to(device)
    contact_encodings = contact_encodings.to(device)
    
    sim_norm, sim_mean, sim_std = normalize_features(sim_features)
    exp_norm, exp_mean, exp_std = normalize_features(exp_features)
    disp_norm, disp_mean, disp_std = normalize_features(disp_conditions)
    
    feature_dim = sim_features.shape[1]
    displacement_dim = disp_conditions.shape[1]
    contact_encoding_dim = contact_encodings.shape[1]
    
    print("="*60)
    print("Training CFM Model for Sim-to-Real Transfer")
    print("="*60)
    print(f"Feature dimension: {feature_dim}")
    print(f"Displacement condition dimension: {displacement_dim}")
    print(f"Contact encoding dimension: {contact_encoding_dim}")
    print(f"Number of samples: {sim_features.shape[0]}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print("="*60 + "\n")
    
    model = ContactConditionedVectorField(
        feature_dim=feature_dim,
        displacement_dim=displacement_dim,
        contact_encoding_dim=contact_encoding_dim,
        hidden_dim=512
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    FM = ConditionalFlowMatcher(sigma=0.01)
    
    losses = []
    n_samples = sim_features.shape[0]
    
    print("Starting training...\n")
    
    for epoch in tqdm(range(num_epochs), desc="Training CFM"):
        idx = torch.randperm(n_samples)[:batch_size]
        
        x0 = sim_norm[idx]
        x1 = exp_norm[idx]
        disp_cond = disp_norm[idx]
        contact_enc = contact_encodings[idx]
        
        optimizer.zero_grad()
        
        t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
        
        v_t = model(t, x_t, disp_cond, contact_enc)
        
        loss = F.mse_loss(v_t, u_t)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            avg_loss = np.mean(losses[-1000:])
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Avg Loss (last 1000): {avg_loss:.6f}")
    
    print("\nTraining complete!")
    
    model_path = save_path / "cfm_sim2real.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': feature_dim,
        'displacement_dim': displacement_dim,
        'contact_encoding_dim': contact_encoding_dim,
        'sim_mean': sim_mean.cpu(),
        'sim_std': sim_std.cpu(),
        'exp_mean': exp_mean.cpu(),
        'exp_std': exp_std.cpu(),
        'disp_mean': disp_mean.cpu(),
        'disp_std': disp_std.cpu(),
        'losses': losses
    }, model_path)
    print(f"Model saved to {model_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.6, label='Loss')
    
    window = 100
    if len(losses) > window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg, 
                linewidth=2, color='red', label=f'Moving Avg ({window})')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('CFM Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path / "cfm_training_loss.pdf", dpi=150)
    plt.show()
    
    return model, sim_mean, sim_std, exp_mean, exp_std, disp_mean, disp_std


def main():
    base_dir = "finray-sim2real_ext_models"
    
    train_geometries = ['pointy1', 'wshape', 'pointy3', 'sphere1', 'sphere2']
    
    name_to_idx, idx_to_name = create_geometry_mapping()
    
    print("="*60)
    print("CFM Training for Sim-to-Real Transfer")
    print("="*60)
    print(f"Training geometries: {train_geometries}")
    print(f"Device: {device}")
    print("="*60)
    
    print("\n" + "="*60)
    print("Step 1: Loading geometry data")
    print("="*60)
    
    geometry_data = load_geometry_data(
        base_dir=base_dir,
        geometries=train_geometries,
        use_normalized_sim=True,
        remap_sim_contact=False
    )
    
    print("\n" + "="*60)
    print("Step 2: Loading contact encoder")
    print("="*60)
    
    contact_processor = ContactDataProcessor(base_dir=base_dir)
    
    encoder_path = Path(base_dir) / "dataset_models" / "contact_encoder.pth"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Contact encoder not found at {encoder_path}. "
                              "Please train it first using train_contact_encoder_all_geoms.py")
    
    checkpoint = torch.load(encoder_path, map_location=device)
    contact_encoder = ContactEncoder(
        num_top_surface_nodes=checkpoint['num_top_surface_nodes'],
        bottleneck_dim=checkpoint['bottleneck_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    contact_encoder.load_state_dict(checkpoint['model_state_dict'])
    contact_encoder = contact_encoder.to(device)
    contact_encoder.eval()
    
    print(f"Contact encoder loaded: {checkpoint['num_top_surface_nodes']} nodes → {checkpoint['bottleneck_dim']} dims")
    
    print("\n" + "="*60)
    print("Step 3: Loading GeometryConditionedPredictor")
    print("="*60)
    
    predictor_path = Path(base_dir) / "dataset_models" / "geometry_conditioned_predictor.pth"
    if not predictor_path.exists():
        raise FileNotFoundError(f"Predictor not found at {predictor_path}. "
                              "Please train it first using data_gen_model_train.py")
    
    predictor_checkpoint = torch.load(predictor_path, map_location=device)
    print(f"Predictor loaded: {predictor_checkpoint['input_dim']} inputs → "
          f"{predictor_checkpoint['num_points']} points")
    
    print("\n" + "="*60)
    print("Step 4: Preparing simulation-experiment pairs")
    print("="*60)
    
    sim_features, exp_features, disp_conditions, contact_encodings, geom_labels = prepare_sim_exp_pairs_with_predictor(
        geometry_data,
        predictor_checkpoint,
        contact_processor,
        contact_encoder,
        train_geometries,
        name_to_idx,
        device
    )
    
    print("\n" + "="*60)
    print("Step 5: Training CFM model")
    print("="*60)
    
    model, sim_mean, sim_std, exp_mean, exp_std, disp_mean, disp_std = train_cfm_sim2real(
        sim_features,
        exp_features,
        disp_conditions,
        contact_encodings,
        num_epochs=80000,
        batch_size=1024,
        lr=0.002,
        device=device,
        save_dir=Path(base_dir) / "cfm_models"
    )
    

    

if __name__ == "__main__":
    main()