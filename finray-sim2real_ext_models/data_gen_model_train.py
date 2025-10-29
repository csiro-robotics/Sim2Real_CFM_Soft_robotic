import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_funcs import load_geometry_data, load_combined_data, validate_and_clean_data, split_data_by_phase


class GeometryConditionedPredictor(nn.Module):
    
    def __init__(self, input_dim=4, num_points=9, num_geometries=6, hidden_dim=256, use_embedding=True):
        super(GeometryConditionedPredictor, self).__init__()
        self.num_points = num_points
        self.num_geometries = num_geometries
        self.use_embedding = use_embedding
        
        output_dim = 1 + num_points * 2
        
        if use_embedding:
            self.geometry_embedding = nn.Embedding(num_geometries, 32)
            actual_input_dim = input_dim + 32
        else:
            actual_input_dim = input_dim + num_geometries
        
        self.net = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
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
            
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.force_scale = nn.Parameter(torch.ones(num_geometries))
        self.force_bias = nn.Parameter(torch.zeros(num_geometries))
    
    def forward(self, x, geometry_ids):
        if self.use_embedding:
            geom_embed = self.geometry_embedding(geometry_ids)
            x_with_geom = torch.cat([x, geom_embed], dim=1)
        else:
            geom_onehot = F.one_hot(geometry_ids, self.num_geometries).float()
            x_with_geom = torch.cat([x, geom_onehot], dim=1)
        
        output = self.net(x_with_geom)
        
        force_pred = output[:, 0:1]
        positions_pred = output[:, 1:]
        
        scale = self.force_scale[geometry_ids].unsqueeze(1)
        bias = self.force_bias[geometry_ids].unsqueeze(1)
        force_scaled = force_pred * scale + bias
        
        return torch.cat([force_scaled, positions_pred], dim=1)


def create_geometry_mapping():
    geometry_names = ['pointy1', 'pointy2', 'pointy3', 'sphere1', 'sphere2', 'wshape']
    name_to_idx = {name: idx for idx, name in enumerate(geometry_names)}
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    return name_to_idx, idx_to_name


def augment_features_with_geometry(data, geometry_label):
    force = data[:, 0:1]
    positions = data[:, 1:-2]
    disp = data[:, -2:-1]
    phase = data[:, -1:]
    
    disp_sq = disp ** 2
    disp_cube = disp ** 3
    
    input_features = torch.cat([disp, disp_sq, disp_cube, phase], dim=1)
    output_features = torch.cat([force, positions], dim=1)
    geometry_labels = torch.full((data.shape[0],), geometry_label, dtype=torch.long)
    
    return input_features, output_features, geometry_labels


def prepare_combined_dataset(geometry_data, name_to_idx, use_experimental=False, equalize_counts=False):
    tensor_idx = 1 if use_experimental else 0
    per_geom_counts = []

    if equalize_counts:
        for geom_name, tensors in geometry_data.items():
            data_tensor = tensors[tensor_idx]
            data_clean = validate_and_clean_data(data_tensor, f"{geom_name}_{'exp' if use_experimental else 'sim'}")
            if data_clean.shape[0] > 0:
                per_geom_counts.append(data_clean.shape[0])
        target_count = min(per_geom_counts) if per_geom_counts else 0
    else:
        target_count = None

    all_inputs, all_outputs, all_geom_labels = [], [], []

    for geom_name, tensors in geometry_data.items():
        data_tensor = tensors[tensor_idx]
        data_clean = validate_and_clean_data(data_tensor, f"{geom_name}_{'exp' if use_experimental else 'sim'}")
        if data_clean.shape[0] == 0:
            continue

        if equalize_counts and data_clean.shape[0] > target_count:
            idx = torch.randperm(data_clean.shape[0])[:target_count]
            data_clean = data_clean[idx]

        geom_idx = name_to_idx[geom_name]
        inputs, outputs, geom_labels = augment_features_with_geometry(data_clean, geom_idx)

        all_inputs.append(inputs)
        all_outputs.append(outputs)
        all_geom_labels.append(geom_labels)

    if not all_inputs:
        return None, None, None

    combined_inputs = torch.cat(all_inputs, dim=0)
    combined_outputs = torch.cat(all_outputs, dim=0)
    combined_labels = torch.cat(all_geom_labels, dim=0)

    return combined_inputs, combined_outputs, combined_labels


def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        std[std < 1e-6] = 1.0
    return (data - mean) / std, mean, std


def train_conditioned_predictor(model, train_inputs, train_outputs, train_geom_labels,
                                val_inputs, val_outputs, val_geom_labels,
                                optimizer, scheduler, epochs=1000, name=""):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in tqdm(range(epochs), desc=f"Training {name}"):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(train_inputs, train_geom_labels)
        train_loss = F.mse_loss(predictions, train_outputs)
        
        reg_loss = 0.01 * (model.force_scale.pow(2).mean() + model.force_bias.pow(2).mean())
        total_loss = train_loss + reg_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(train_loss.item())
        
        if val_inputs is not None and epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(val_inputs, val_geom_labels)
                val_loss = F.mse_loss(val_predictions, val_outputs)
                val_losses.append(val_loss.item())
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 100 == 0:
            if val_inputs is not None:
                print(f"Epoch {epoch+1}: Train Loss={train_loss.item():.6f}, Val Loss={val_loss.item():.6f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={train_loss.item():.6f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_by_geometry(model, test_inputs, test_outputs, test_geom_labels,
                         input_mean, input_std, output_mean, output_std,
                         idx_to_name, num_points=9):
    model.eval()
    results_by_geometry = {}
    
    with torch.no_grad():
        predictions = model(test_inputs, test_geom_labels)
        
        inputs_denorm = test_inputs * input_std + input_mean
        outputs_denorm = test_outputs * output_std + output_mean
        predictions_denorm = predictions * output_std + output_mean
        
        unique_geoms = torch.unique(test_geom_labels)
        
        for geom_idx in unique_geoms:
            geom_idx = geom_idx.item()
            geom_name = idx_to_name[geom_idx]
            
            mask = test_geom_labels == geom_idx
            
            geom_outputs = outputs_denorm[mask]
            geom_predictions = predictions_denorm[mask]
            geom_inputs = inputs_denorm[mask]
            
            force_true = geom_outputs[:, 0].numpy()
            force_pred = geom_predictions[:, 0].numpy()
            
            force_mse = np.mean((force_true - force_pred) ** 2)
            force_mae = np.mean(np.abs(force_true - force_pred))
            force_mape = np.mean(np.abs((force_true - force_pred) / (force_true + 1e-8))) * 100
            
            print(f"\n{geom_name} Evaluation:")
            print(f"  Samples: {mask.sum().item()}")
            print(f"  Force MSE: {force_mse:.6f}, MAE: {force_mae:.6f}, MAPE: {force_mape:.2f}%")
            print(f"  Force range: [{force_true.min():.4f}, {force_true.max():.4f}] N")
            print(f"  Force scale factor: {model.force_scale[geom_idx].item():.3f}")
            print(f"  Force bias: {model.force_bias[geom_idx].item():.3f}")
            
            results_by_geometry[geom_name] = {
                'displacement': geom_inputs[:, 0].numpy(),
                'force_true': force_true,
                'force_pred': force_pred,
                'positions_true': geom_outputs[:, 1:].numpy().reshape(-1, num_points, 2),
                'positions_pred': geom_predictions[:, 1:].numpy().reshape(-1, num_points, 2),
                'metrics': {
                    'force_mse': force_mse,
                    'force_mae': force_mae,
                    'force_mape': force_mape
                }
            }
    
    return results_by_geometry


def visualize_geometry_specific_results(results_dict, save_path="geometry_predictions.pdf"):
    n_geoms = len(results_dict)
    fig, axes = plt.subplots(3, max(2, (n_geoms + 1) // 2), 
                             figsize=(7 * max(2, (n_geoms + 1) // 2), 15))
    axes = axes.flatten()
    
    for ax in axes[n_geoms:]:
        ax.set_visible(False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_geoms))
    
    for idx, (name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        sort_idx = np.argsort(results['displacement'])
        disp_sorted = results['displacement'][sort_idx]
        force_true_sorted = results['force_true'][sort_idx]
        force_pred_sorted = results['force_pred'][sort_idx]
        
        ax.scatter(disp_sorted, force_true_sorted, alpha=0.6, s=20, 
                  label='True', color=colors[idx])
        ax.plot(disp_sorted, force_pred_sorted, 'r-', alpha=0.8, 
               linewidth=2, label='Predicted')
        
        mse = results['metrics']['force_mse']
        mae = results['metrics']['force_mae']
        mape = results['metrics']['force_mape']
        
        ax.set_xlabel('Displacement (m)')
        ax.set_ylabel('Force (N)')
        ax.set_title(f'{name}\nMSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Geometry-Specific Force Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Results saved to {save_path}")


def plot_force_comparison_matrix(results_dict, save_path="force_comparison_matrix.pdf"):
    n_geoms = len(results_dict)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_geoms))
    
    for idx, (name, results) in enumerate(results_dict.items()):
        sort_idx = np.argsort(results['displacement'])
        disp = results['displacement'][sort_idx]
        force_true = results['force_true'][sort_idx]
        force_pred = results['force_pred'][sort_idx]
        
        ax.scatter(disp, force_true, alpha=0.3, s=15, color=colors[idx])
        ax.plot(disp, force_pred, '-', alpha=0.8, linewidth=2, 
               color=colors[idx], label=name)
    
    ax.set_xlabel('Displacement (m)', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.set_title('Force Profiles for Different Indentor Geometries', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Force comparison saved to {save_path}")


def main():
    base_dir = "finray-sim2real_ext_models"
    output_dir = Path(base_dir) / "dataset_models"
    output_dir.mkdir(exist_ok=True)
    
    name_to_idx, idx_to_name = create_geometry_mapping()
    geometries = list(name_to_idx.keys())
    
    print("="*60)
    print("Geometry-Conditioned Force Prediction Model")
    print("="*60)
    print(f"Geometries: {geometries}")
    print(f"Mapping: {name_to_idx}")
    
    print("\n" + "="*60)
    print("Loading geometry data...")
    print("="*60)
    
    geometry_data = load_geometry_data(
        base_dir=base_dir,
        geometries=geometries,
        use_normalized_sim=True
    )
    
    print("\n" + "="*60)
    print("Preparing combined dataset...")
    print("="*60)
    
    combined_inputs, combined_outputs, combined_geom_labels = prepare_combined_dataset(
        geometry_data, name_to_idx, use_experimental=True, equalize_counts=True
    )
    
    if combined_inputs is None:
        print("No valid data found!")
        return
    
    print(f"Combined dataset shape:")
    print(f"  Inputs: {combined_inputs.shape}")
    print(f"  Outputs: {combined_outputs.shape}")
    print(f"  Geometry labels: {combined_geom_labels.shape}")
    
    for geom_idx in range(len(geometries)):
        count = (combined_geom_labels == geom_idx).sum().item()
        print(f"  {idx_to_name[geom_idx]}: {count} samples")
    
    n_samples = combined_inputs.shape[0]
    indices = torch.randperm(n_samples)
    n_train = int(0.8 * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_inputs = combined_inputs[train_idx]
    train_outputs = combined_outputs[train_idx]
    train_geom_labels = combined_geom_labels[train_idx]
    
    val_inputs = combined_inputs[val_idx]
    val_outputs = combined_outputs[val_idx]
    val_geom_labels = combined_geom_labels[val_idx]
    
    train_inputs_norm, input_mean, input_std = normalize_data(train_inputs)
    val_inputs_norm, _, _ = normalize_data(val_inputs, input_mean, input_std)
    
    train_outputs_norm, output_mean, output_std = normalize_data(train_outputs)
    val_outputs_norm, _, _ = normalize_data(val_outputs, output_mean, output_std)
    
    print(f"\nTraining set: {n_train} samples")
    print(f"Validation set: {n_samples - n_train} samples")
    
    num_points = (combined_outputs.shape[1] - 1) // 2
    input_dim = combined_inputs.shape[1]
    num_geometries = len(geometries)
    
    print(f"\nModel configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of tracked points: {num_points}")
    print(f"  Number of geometries: {num_geometries}")
    
    model = GeometryConditionedPredictor(
        input_dim=input_dim,
        num_points=num_points,
        num_geometries=num_geometries,
        hidden_dim=256,
        use_embedding=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)
    
    print("\n" + "="*60)
    print("Training geometry-conditioned model...")
    print("="*60)
    
    model, train_losses, val_losses = train_conditioned_predictor(
        model,
        train_inputs_norm, train_outputs_norm, train_geom_labels,
        val_inputs_norm, val_outputs_norm, val_geom_labels,
        optimizer, scheduler,
        epochs=1500,
        name="Geometry-Conditioned"
    )
    
    print("\n" + "="*60)
    print("Evaluating on validation set...")
    print("="*60)
    
    results = evaluate_by_geometry(
        model,
        val_inputs_norm, val_outputs_norm, val_geom_labels,
        input_mean, input_std, output_mean, output_std,
        idx_to_name,
        num_points=num_points
    )
    
    visualize_geometry_specific_results(results, save_path=output_dir / "geometry_predictions.pdf")
    plot_force_comparison_matrix(results, save_path=output_dir / "force_comparison.pdf")
    
    model_path = output_dir / "geometry_conditioned_predictor.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'num_points': num_points,
        'input_dim': input_dim,
        'num_geometries': num_geometries,
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name,
        'force_scales': model.force_scale.detach().cpu(),
        'force_biases': model.force_bias.detach().cpu()
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "="*60)
    print("Learned geometry-specific force parameters:")
    print("="*60)
    for idx, name in idx_to_name.items():
        scale = model.force_scale[idx].item()
        bias = model.force_bias[idx].item()
        print(f"  {name}: scale={scale:.3f}, bias={bias:.3f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if val_losses:
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(val_losses)*10, 10), val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.pdf", dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()