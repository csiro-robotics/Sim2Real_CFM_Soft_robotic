import json
import glob
import os
import torch
import numpy as np
from pathlib import Path

def load_top_surface_indices(base_dir="finray-sim2real_ext_models"):
    indices_path = Path(base_dir) / "sim_data" / "top_surface_indices.npy"
    
    if not indices_path.exists():
        print(f"Warning: Top surface indices file not found at {indices_path}")
        print("Run extract_top_surface_indices.py first to generate this file.")
        return np.array([]), {}
    
    top_indices = np.load(indices_path)
    
    index_mapping = {int(orig_idx): seq_idx for seq_idx, orig_idx in enumerate(top_indices)}
    
    print(f"Loaded {len(top_indices)} top surface indices")
    print(f"Index range: {top_indices.min()} to {top_indices.max()}")
    
    return top_indices, index_mapping

def remap_contact_indices(contact_indices, index_mapping):
    remapped = []
    skipped = []
    
    for orig_idx in contact_indices:
        if orig_idx in index_mapping:
            remapped.append(index_mapping[orig_idx])
        else:
            skipped.append(orig_idx)
    
    if skipped:
        print(f"  Warning: {len(skipped)} contact indices not in top surface: {skipped[:5]}...")
    
    return remapped

def get_contact_info(entry):
    contact = entry.get("contact_info", {})
    indices = contact.get("indices", [])
    positions = contact.get("positions", [])
    if not isinstance(indices, list):
        indices = []
    if not isinstance(positions, list):
        positions = []
    return indices, positions

def read_geometry_json(file_path, geometry_name=None, contact_mapping=None):
    with open(file_path, 'r') as f:
        data_entries = json.load(f)
    
    if not data_entries:
        return torch.empty((0, 0), dtype=torch.float32)
    
    all_indices = set()
    for entry in data_entries:
        tracked_nodes = entry.get('tracked_nodes', {})
        indices = tracked_nodes.get('indices', [])
        
        if not indices:
            contact_indices, _ = get_contact_info(entry)
            indices = contact_indices
            
            if contact_mapping is not None and indices:
                indices = remap_contact_indices(indices, contact_mapping)
        
        all_indices.update(indices)
    
    tracked_indices = sorted(all_indices) if all_indices else list(range(9))
    
    print(f"  Tracked indices for {geometry_name}: {tracked_indices}")
    
    processed_data = []
    
    for entry in data_entries:
        force = entry.get('finger_force', 0.0)
        if force == 0.0:
            force = entry.get('net_forces', {}).get('total_z', 0.0)
        
        pointy_pos = entry.get('pointy_position', [0.25, 0.45, 0.0])
        if isinstance(pointy_pos, list):
            if len(pointy_pos) == 1:
                displacement = abs(pointy_pos[0])
            elif len(pointy_pos) >= 2:
                displacement = abs(0.45 - pointy_pos[1])
        else:
            displacement = 0.0
        
        phase = entry.get('phase', 'compression')
        phase_encoded = 0.0 if phase == 'compression' else 1.0
        
        tracked_nodes = entry.get('tracked_nodes', {})
        indices = tracked_nodes.get('indices', [])
        positions = tracked_nodes.get('positions', [])
        
        if not indices:
            contact_indices, contact_positions = get_contact_info(entry)
            indices = contact_indices
            positions = contact_positions
            
            if contact_mapping is not None and indices:
                original_indices = indices
                indices = remap_contact_indices(indices, contact_mapping)
                
                remapped_positions = []
                for i, orig_idx in enumerate(original_indices):
                    if orig_idx in contact_mapping:
                        remapped_positions.append(positions[i])
                positions = remapped_positions
        
        pos_dict = {}
        for idx, pos in zip(indices, positions):
            if len(pos) >= 2:
                pos_dict[idx] = pos[:3]
        
        flattened_xy = []
        for idx in tracked_indices:
            if idx in pos_dict:
                flattened_xy.extend([pos_dict[idx][0], pos_dict[idx][1]])
            else:
                flattened_xy.extend([0.0, 0.0])
        
        features = [force] + flattened_xy + [displacement, phase_encoded]
        processed_data.append(features)
    
    return torch.tensor(processed_data, dtype=torch.float32)


def load_geometry_data(base_dir="finray-sim2real_ext_models", 
                       geometries=None,
                       use_normalized_sim=True,
                       remap_sim_contact=False):
    base_path = Path(base_dir)
    
    if geometries is None:
        geometries = ['pointy1', 'pointy2', 'pointy3', 'sphere1', 'sphere2', 'wshape']
    
    contact_mapping = None
    if remap_sim_contact:
        _, contact_mapping = load_top_surface_indices(base_dir)
        if not contact_mapping:
            print("Warning: Could not load top surface mapping, proceeding without remapping")
            remap_sim_contact = False
    
    data_dict = {}
    
    for geometry in geometries:
        print(f"\nLoading {geometry} data...")
        
        if use_normalized_sim:
            sim_file = base_path / "sim_data" / f"data_{geometry}" / f"{geometry}_sim_normalized.json"
        else:
            sim_file = base_path / "sim_data" / f"data_{geometry}" / f"{geometry}_sim.json"
        
        exp_file = base_path / "exp_data" / f"data_{geometry}" / f"{geometry}_exp_aligned.json"
        
        if not exp_file.exists():
            exp_file = base_path / "exp_data" / f"data_{geometry}" / f"{geometry}_exp_formatted.json"
            print(f"  Warning: Using formatted data (not aligned)")
        
        if sim_file.exists():
            sim_mapping = contact_mapping if remap_sim_contact else None
            sim_data = read_geometry_json(str(sim_file), f"{geometry}_sim", sim_mapping)
            print(f"  Loaded simulation data: {sim_data.shape}")
            if remap_sim_contact:
                print(f"    (contact indices remapped to sequential)")
        else:
            print(f"  Simulation file not found: {sim_file}")
            sim_data = torch.empty((0, 0), dtype=torch.float32)
        
        if exp_file.exists():
            exp_data = read_geometry_json(str(exp_file), f"{geometry}_exp", contact_mapping=None)
            print(f"  Loaded experimental data: {exp_data.shape}")
            print(f"    (uses tracked_nodes, no remapping)")
        else:
            print(f"  Experimental file not found: {exp_file}")
            exp_data = torch.empty((0, 0), dtype=torch.float32)
        
        data_dict[geometry] = (sim_data, exp_data)
    
    return data_dict


def load_combined_data(base_dir="finray-sim2real_ext_models",
                      geometries=None,
                      use_normalized_sim=True,
                      remap_sim_contact=False):
    data_dict = load_geometry_data(base_dir, geometries, use_normalized_sim, remap_sim_contact)
    
    all_sim_data = []
    all_exp_data = []
    sim_labels = []
    exp_labels = []
    
    for i, (geometry, (sim_data, exp_data)) in enumerate(data_dict.items()):
        if sim_data.numel() > 0:
            all_sim_data.append(sim_data)
            sim_labels.extend([i] * sim_data.shape[0])
        
        if exp_data.numel() > 0:
            all_exp_data.append(exp_data)
            exp_labels.extend([i] * exp_data.shape[0])
    
    if all_sim_data:
        combined_sim = torch.cat(all_sim_data, dim=0)
        sim_labels = torch.tensor(sim_labels, dtype=torch.long)
    else:
        combined_sim = torch.empty((0, 0), dtype=torch.float32)
        sim_labels = torch.tensor([], dtype=torch.long)
    
    if all_exp_data:
        combined_exp = torch.cat(all_exp_data, dim=0)
        exp_labels = torch.tensor(exp_labels, dtype=torch.long)
    else:
        combined_exp = torch.empty((0, 0), dtype=torch.float32)
        exp_labels = torch.tensor([], dtype=torch.long)
    
    print(f"\nCombined data shapes:")
    print(f"  Simulation: {combined_sim.shape}, Labels: {sim_labels.shape}")
    print(f"  Experimental: {combined_exp.shape}, Labels: {exp_labels.shape}")
    
    label_mapping = {i: geom for i, geom in enumerate(data_dict.keys())}
    print(f"\nLabel mapping: {label_mapping}")
    
    return combined_sim, combined_exp, (sim_labels, exp_labels, label_mapping)


def validate_and_clean_data(tensor_data, name=""):
    if tensor_data.numel() == 0:
        print(f"{name} is empty")
        return tensor_data
    
    has_nan = torch.isnan(tensor_data)
    has_inf = torch.isinf(tensor_data)
    
    print(f"\nValidating {name}:")
    print(f"  Shape: {tensor_data.shape}")
    
    for i in range(tensor_data.shape[1]):
        nan_count = has_nan[:, i].sum().item()
        inf_count = has_inf[:, i].sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"  Feature {i}: {nan_count} NaN, {inf_count} Inf values")
    
    valid_rows = torch.all(torch.isfinite(tensor_data), dim=1)
    tensor_cleaned = tensor_data[valid_rows]
    
    removed_count = (~valid_rows).sum().item()
    if removed_count > 0:
        print(f"  Removed {removed_count} invalid entries")
    
    print(f"  Remaining entries: {tensor_cleaned.shape[0]}")
    
    if tensor_cleaned.shape[0] > 0:
        mean = tensor_cleaned.mean(dim=0)
        std = tensor_cleaned.std(dim=0)
        min_vals = tensor_cleaned.min(dim=0)[0]
        max_vals = tensor_cleaned.max(dim=0)[0]
        
        print(f"\n{name} statistics:")
        print(f"  Force (feature 0): mean={mean[0]:.4f}, std={std[0]:.4f}, range=[{min_vals[0]:.4f}, {max_vals[0]:.4f}]")
        print(f"  Displacement (feature -2): mean={mean[-2]:.4f}, std={std[-2]:.4f}, range=[{min_vals[-2]:.4f}, {max_vals[-2]:.4f}]")
        print(f"  Phase (feature -1): mean={mean[-1]:.4f}, std={std[-1]:.4f}")
    
    return tensor_cleaned


def get_feature_names(num_tracked_nodes=9):
    feature_names = ['force']
    
    for i in range(num_tracked_nodes):
        feature_names.extend([f'node_{i}_x', f'node_{i}_y'])
    
    feature_names.extend(['displacement', 'phase'])
    
    return feature_names


def split_data_by_phase(data_tensor):
    if data_tensor.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0, 0), dtype=torch.float32)
    
    compression_mask = data_tensor[:, -1] == 0
    decompression_mask = data_tensor[:, -1] == 1
    
    compression_data = data_tensor[compression_mask]
    decompression_data = data_tensor[decompression_mask]
    
    print(f"Split data by phase:")
    print(f"  Compression: {compression_data.shape[0]} samples")
    print(f"  Decompression: {decompression_data.shape[0]} samples")
    
    return compression_data, decompression_data


if __name__ == "__main__":
    print("="*60)
    print("Example 1: Loading without contact remapping")
    print("="*60)
    
    geometry_data = load_geometry_data(
        base_dir="finray-sim2real_ext_models",
        geometries=['pointy1', 'sphere2'],
        remap_sim_contact=False
    )
    
    print("\n" + "="*60)
    print("Example 2: Loading with contact remapping for simulation")
    print("="*60)
    
    geometry_data_remapped = load_geometry_data(
        base_dir="finray-sim2real_ext_models",
        geometries=['pointy1', 'sphere2'],
        remap_sim_contact=True
    )
    
    for geom_name, (sim_data, exp_data) in geometry_data_remapped.items():
        print(f"\n{geom_name}:")
        print(f"  Simulation shape: {sim_data.shape} (contact remapped)")
        print(f"  Experimental shape: {exp_data.shape} (tracked_nodes, no remapping)")