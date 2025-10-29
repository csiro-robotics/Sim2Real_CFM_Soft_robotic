import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from contact_encoder import ContactEncoder, ContactDataProcessor, train_contact_encoder

def load_all_simulation_contact_data(base_dir="finray-sim2real_ext_models", geometries=None):
    """
    Load contact data from all simulation geometries.
    
    Args:
        base_dir (str): Base directory
        geometries (list): List of geometry names. If None, uses default.
    
    Returns:
        dict: Dictionary with geometry names as keys and contact arrays as values
        torch.Tensor: Combined contact arrays from all geometries
    """
    if geometries is None:
        geometries = ['pointy1', 'pointy2', 'pointy3', 'sphere1', 'sphere2', 'wshape']
    
    base_path = Path(base_dir)
    processor = ContactDataProcessor(base_dir=base_dir)
    
    all_contact_data = {}
    all_arrays = []
    
    print("="*60)
    print("Loading simulation contact data for all geometries")
    print("="*60)
    
    for geom_name in geometries:
        # Path to simulation data
        json_path = base_path / "sim_data" / f"data_{geom_name}" / f"{geom_name}_sim_normalized.json"
        
        if not json_path.exists():
            print(f"\nWarning: {geom_name} data not found at {json_path}")
            continue
        
        print(f"\nProcessing {geom_name}...")
        print(f"  File: {json_path}")
        
        # Process JSON file to contact arrays
        contact_arrays = processor.process_json_file(str(json_path), use_remapping=True)
        
        print(f"  Shape: {contact_arrays.shape}")
        print(f"  Samples: {contact_arrays.shape[0]}")
        
        # Check contact statistics
        num_in_contact = (contact_arrays != 0).any(dim=1).sum().item()
        print(f"  Timesteps with contact: {num_in_contact}/{contact_arrays.shape[0]}")
        
        # Store data
        all_contact_data[geom_name] = contact_arrays
        all_arrays.append(contact_arrays)
    
    # Combine all data
    if all_arrays:
        combined_data = torch.cat(all_arrays, dim=0)
        print(f"\n{'='*60}")
        print(f"Combined dataset statistics:")
        print(f"  Total samples: {combined_data.shape[0]}")
        print(f"  Feature dimension: {combined_data.shape[1]}")
        print(f"  Samples per geometry:")
        for geom_name, data in all_contact_data.items():
            print(f"    {geom_name}: {data.shape[0]}")
        print(f"{'='*60}\n")
    else:
        combined_data = torch.empty((0, processor.num_nodes))
        print("Warning: No data loaded!")
    
    return all_contact_data, combined_data, processor

def visualize_contact_statistics(all_contact_data, processor, save_path="contact_statistics.pdf"):
    """
    Visualize contact statistics for all geometries.
    
    Args:
        all_contact_data (dict): Dictionary of contact arrays per geometry
        processor (ContactDataProcessor): Processor instance
        save_path (str): Path to save figure
    """
    n_geoms = len(all_contact_data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_geoms))
    
    for idx, (geom_name, contact_data) in enumerate(all_contact_data.items()):
        ax = axes[idx]
        
        # Calculate average contact pattern
        contact_data_np = contact_data.numpy()
        
        # Average y-coordinate at each node (excluding zeros)
        avg_contact = []
        for node_idx in range(contact_data.shape[1]):
            node_values = contact_data_np[:, node_idx]
            non_zero = node_values[node_values != 0]
            if len(non_zero) > 0:
                avg_contact.append(non_zero.mean())
            else:
                avg_contact.append(0.0)
        
        avg_contact = np.array(avg_contact)
        
        # Count how often each node is in contact
        contact_frequency = (contact_data_np != 0).sum(axis=0) / contact_data.shape[0] * 100
        
        # Plot
        ax2 = ax.twinx()
        
        # Bar plot for contact frequency
        ax.bar(range(len(contact_frequency)), contact_frequency, 
               alpha=0.3, color=colors[idx], label='Contact frequency')
        ax.set_ylabel('Contact Frequency (%)', color=colors[idx])
        ax.tick_params(axis='y', labelcolor=colors[idx])
        
        # Line plot for average y-coordinate
        ax2.plot(avg_contact, 'o-', color='red', markersize=2, 
                linewidth=1, alpha=0.7, label='Avg Y-coord')
        ax2.set_ylabel('Average Y-coordinate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Node Index')
        ax.set_title(f'{geom_name}\n({contact_data.shape[0]} samples)')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if n_geoms < len(axes):
        for ax in axes[n_geoms:]:
            ax.set_visible(False)
    
    plt.suptitle('Contact Patterns Across All Geometries', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Contact statistics saved to {save_path}")

def train_and_evaluate_contact_encoder(combined_data, processor, 
                                       bottleneck_dim=128,
                                       hidden_dims=[256, 512, 256],
                                       num_epochs=500,
                                       batch_size=32,
                                       lr=0.001,
                                       device='cuda',
                                       output_dir="finray-sim2real_ext_models/dataset_models"):
    """
    Train contact encoder on combined data and evaluate.
    
    Args:
        combined_data (torch.Tensor): Combined contact arrays
        processor (ContactDataProcessor): Data processor
        bottleneck_dim (int): Size of bottleneck encoding
        hidden_dims (list): Hidden layer dimensions
        num_epochs (int): Training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        device (str): Device to use
        output_dir (str): Directory to save outputs
    
    Returns:
        ContactEncoder: Trained encoder
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Training Contact Encoder on All Geometries")
    print("="*60)
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Bottleneck dimension: {bottleneck_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Split data (80/20 train/val)
    n_samples = combined_data.shape[0]
    indices = torch.randperm(n_samples)
    n_train = int(0.8 * n_samples)
    
    train_data = combined_data[indices[:n_train]]
    val_data = combined_data[indices[n_train:]]
    
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}\n")
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create encoder
    encoder = ContactEncoder(
        num_top_surface_nodes=processor.num_nodes,
        bottleneck_dim=bottleneck_dim,
        hidden_dims=hidden_dims
    )
    
    print(f"Encoder architecture:")
    print(f"  Input size: {processor.num_nodes}")
    print(f"  Bottleneck size: {bottleneck_dim}")
    print(f"  Hidden layers: {hidden_dims}")
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}\n")
    
    # Train encoder with validation
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    print("Starting training...\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for batch_data in train_loader:
            if isinstance(batch_data, (list, tuple)):
                contact_arrays = batch_data[0]
            else:
                contact_arrays = batch_data
            
            contact_arrays = contact_arrays.to(device)
            
            # Forward pass
            encoding, reconstruction = encoder.encode_decode(contact_arrays)
            loss = criterion(reconstruction, contact_arrays)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        scheduler.step()
        
        # Validation phase
        encoder.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    contact_arrays = batch_data[0]
                else:
                    contact_arrays = batch_data
                
                contact_arrays = contact_arrays.to(device)
                
                encoding, reconstruction = encoder.encode_decode(contact_arrays)
                loss = criterion(reconstruction, contact_arrays)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = encoder.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Best Val Loss: {best_val_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        encoder.load_state_dict(best_model_state)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "contact_encoder_training_curves.pdf", dpi=150)
    plt.show()
    
    # Test reconstruction on validation samples
    print("\nTesting reconstruction quality...")
    encoder.eval()
    with torch.no_grad():
        # Take first batch of validation data
        test_samples = val_data[:8].to(device)
        encodings, reconstructions = encoder.encode_decode(test_samples)
        
        # Calculate metrics
        mse = torch.mean((test_samples - reconstructions) ** 2, dim=1)
        mae = torch.mean(torch.abs(test_samples - reconstructions), dim=1)
        
        print(f"  MSE per sample: {mse.cpu().numpy()}")
        print(f"  MAE per sample: {mae.cpu().numpy()}")
        print(f"  Average MSE: {mse.mean().item():.6f}")
        print(f"  Average MAE: {mae.mean().item():.6f}")
        
        # Visualize some reconstructions
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        
        for i in range(min(4, test_samples.shape[0])):
            # Original
            ax1 = axes[i, 0]
            original = test_samples[i].cpu().numpy()
            processor.visualize_contact_array(original, title=f"Original Sample {i+1}")
            plt.close()  # Close individual plots
            
            ax1.plot(original, 'o-', markersize=3, label='Original')
            contact_mask = original != 0
            if contact_mask.any():
                ax1.scatter(np.where(contact_mask)[0], original[contact_mask], 
                          color='red', s=50, alpha=0.6, label='Contact')
            ax1.set_xlabel('Node Index')
            ax1.set_ylabel('Y-coordinate')
            ax1.set_title(f'Original Sample {i+1}\nMSE: {mse[i].item():.6f}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Reconstruction
            ax2 = axes[i, 1]
            reconstruction = reconstructions[i].cpu().numpy()
            
            ax2.plot(reconstruction, 'o-', markersize=3, label='Reconstructed', color='green')
            ax2.plot(original, '--', alpha=0.5, label='Original', color='blue')
            ax2.set_xlabel('Node Index')
            ax2.set_ylabel('Y-coordinate')
            ax2.set_title(f'Reconstructed Sample {i+1}\nMAE: {mae[i].item():.6f}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.suptitle('Contact Encoder Reconstruction Quality', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / "contact_encoder_reconstructions.pdf", dpi=150)
        plt.show()
    
    # Save encoder
    model_path = output_path / "contact_encoder.pth"
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'num_top_surface_nodes': processor.num_nodes,
        'bottleneck_dim': bottleneck_dim,
        'hidden_dims': hidden_dims,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"\nContact encoder saved to {model_path}")
    
    return encoder

def main():
    """Main training pipeline."""
    base_dir = "finray-sim2real_ext_models"
    output_dir = Path(base_dir) / "dataset_models"
    
    # Configuration
    geometries = ['pointy1', 'pointy2', 'pointy3', 'sphere1', 'sphere2', 'wshape']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load all simulation contact data
    all_contact_data, combined_data, processor = load_all_simulation_contact_data(
        base_dir=base_dir,
        geometries=geometries
    )
    
    if combined_data.shape[0] == 0:
        print("Error: No data loaded!")
        return
    
    # Visualize contact statistics
    visualize_contact_statistics(
        all_contact_data, 
        processor, 
        save_path=output_dir / "contact_statistics.pdf"
    )
    
    # Train contact encoder
    encoder = train_and_evaluate_contact_encoder(
        combined_data=combined_data,
        processor=processor,
        bottleneck_dim=128,
        hidden_dims=[256, 512, 256],
        num_epochs=500,
        batch_size=32,
        lr=0.001,
        device=device,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("Contact Encoder Training Complete!")
    print("="*60)
    print(f"Model saved in: {output_dir}")
    print(f"Files generated:")
    print(f"  - contact_encoder.pth")
    print(f"  - contact_encoder_training_curves.pdf")
    print(f"  - contact_encoder_reconstructions.pdf")
    print(f"  - contact_statistics.pdf")
    print("="*60)

if __name__ == "__main__":
    main()