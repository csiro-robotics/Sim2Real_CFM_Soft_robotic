import json
import glob
import os
import torch


def read_json_file(file_path, is_sofa):
    with open(file_path, 'r') as f:
        data_entries = json.load(f)
    selected_idx_set = {3, 12, 68, 88, 82, 105, 540}
    num_points = len(selected_idx_set)
    sorted_indices = sorted(selected_idx_set)

    processed_data = []
    for entry in data_entries:
        force = entry.get('processed_force', 0)
        disp = entry.get('processed_disp', 0)
        speed = entry.get('speed', 0)
        
        indices = entry.get('indices', [0])
        mesh = entry.get('positions', [[0, 0, 0]])
        pos_dict = {i: pos for i, pos in zip(indices, mesh) if i in selected_idx_set}
        
        flattened_xy = []
        if is_sofa:
            for idx in sorted_indices:
                if idx in pos_dict:
                    flattened_xy.extend([pos_dict[idx][1], pos_dict[idx][2]])
                else:
                    flattened_xy.extend([0, 0])
        else:
            for idx in sorted_indices:
                if idx in pos_dict:
                    flattened_xy.extend([pos_dict[idx][0], pos_dict[idx][1]])
                else:
                    flattened_xy.extend([0, 0])
        features = [force] + flattened_xy + [disp, speed]
        processed_data.append(features)
    
    return torch.tensor(processed_data, dtype=torch.float32)

def load_processed_data(is_sofa=True, validation=False):
    if not validation:
        pattern = os.path.join("tensile_sim2sim_src\data_sofa_processed" if is_sofa else "tensile_sim2sim_src\data_warp_processed", "**", "*.json")
        all_entries = [read_json_file(file_path, is_sofa) for file_path in glob.glob(pattern, recursive=True)]

    else:
        pattern = os.path.join("tensile_sim2sim_src\data_validation_sofa" if is_sofa else "tensile_sim2sim_src\data_validation_warp", "**", "*.json")
        all_entries = [read_json_file(file_path, is_sofa) for file_path in glob.glob(pattern, recursive=True)]

    if not all_entries:
        selected_idx_set = {3, 12, 68, 88, 82, 105, 540}
        dim = 1 + 2 * len(selected_idx_set) + 2
        return torch.empty((0, dim))
    return torch.cat(all_entries, dim=0)


def validate_and_clean_data(tensor_data, name=""):
    has_nan = torch.isnan(tensor_data)
    has_inf = torch.isinf(tensor_data)
    print(f"\nValidating {name}:")
    for i in range(tensor_data.shape[1]):
        print(f"Feature {i}: {has_nan[:, i].sum().item()} NaN, {has_inf[:, i].sum().item()} Inf values")
    valid_rows = torch.all(torch.isfinite(tensor_data), dim=1)
    tensor_cleaned = tensor_data[valid_rows]
    print(f"\nRemoved {(~valid_rows).sum().item()} invalid entries")
    print(f"Remaining entries: {tensor_cleaned.shape[0]}")
    mean = tensor_cleaned.mean(dim=0)
    std = tensor_cleaned.std(dim=0)
    print(f"\nCleaned {name} statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return tensor_cleaned