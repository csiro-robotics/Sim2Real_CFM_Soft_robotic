import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from data_funcs import load_top_surface_indices, get_contact_info, remap_contact_indices

class ContactEncoder(nn.Module):
    """
    Encoder that converts variable-length contact information into a fixed-size 128-dimensional vector.
    
    Architecture:
    1. Contact info → Fixed-size array (size = num_top_surface_nodes)
    2. Each position filled with y-coordinate if in contact, 0 otherwise
    3. Feed through neural network to compress to 128-dim bottleneck
    """
    
    def __init__(self, num_top_surface_nodes, bottleneck_dim=128, hidden_dims=[256, 512, 256]):
        """
        Args:
            num_top_surface_nodes (int): Total number of nodes on top surface
            bottleneck_dim (int): Size of the bottleneck encoding (default: 128)
            hidden_dims (list): Hidden layer dimensions for the encoder
        """
        super(ContactEncoder, self).__init__()
        
        self.num_top_surface_nodes = num_top_surface_nodes
        self.bottleneck_dim = bottleneck_dim
        
        # Build encoder layers
        layers = []
        input_dim = num_top_surface_nodes
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Final bottleneck layer
        layers.append(nn.Linear(input_dim, bottleneck_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Optional: Decoder for reconstruction (useful for pre-training)
        self.decoder = None
        self._build_decoder(hidden_dims)
    
    def _build_decoder(self, hidden_dims):
        """Build decoder for reconstruction (optional, for pre-training)"""
        layers = []
        input_dim = self.bottleneck_dim
        
        # Reverse hidden dimensions
        reversed_dims = list(reversed(hidden_dims))
        
        for hidden_dim in reversed_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Final reconstruction layer
        layers.append(nn.Linear(input_dim, self.num_top_surface_nodes))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, contact_array):
        """
        Forward pass through encoder.
        
        Args:
            contact_array (torch.Tensor): Shape [batch_size, num_top_surface_nodes]
                                         Each element is y-coordinate or 0
        
        Returns:
            torch.Tensor: Encoded representation [batch_size, bottleneck_dim]
        """
        return self.encoder(contact_array)
    
    def decode(self, encoding):
        """
        Decode from bottleneck back to contact array (for reconstruction).
        
        Args:
            encoding (torch.Tensor): Shape [batch_size, bottleneck_dim]
        
        Returns:
            torch.Tensor: Reconstructed contact array [batch_size, num_top_surface_nodes]
        """
        if self.decoder is None:
            raise ValueError("Decoder not built. Set build_decoder=True in __init__")
        return self.decoder(encoding)
    
    def encode_decode(self, contact_array):
        """
        Full autoencoder pass (encode + decode).
        
        Args:
            contact_array (torch.Tensor): Input contact array
        
        Returns:
            tuple: (encoding, reconstruction)
        """
        encoding = self.forward(contact_array)
        reconstruction = self.decode(encoding)
        return encoding, reconstruction


class ContactDataProcessor:
    """
    Utility class to convert raw contact info to fixed-size arrays.
    """
    
    def __init__(self, base_dir="finray-sim2real_ext_models"):
        """
        Args:
            base_dir (str): Base directory containing top_surface_indices.npy
        """
        self.top_indices, self.index_mapping = load_top_surface_indices(base_dir)
        self.num_nodes = len(self.top_indices)
        
        if self.num_nodes == 0:
            raise ValueError("No top surface indices found. Run extract_top_surface_indices.py first.")
        
        print(f"ContactDataProcessor initialized with {self.num_nodes} top surface nodes")
    
    def process_contact_entry(self, entry, use_remapping=True):
        """
        Convert a single data entry to fixed-size contact array.
        
        Args:
            entry (dict): Data entry with contact_info
            use_remapping (bool): Whether to remap original mesh indices to sequential
        
        Returns:
            numpy.ndarray: Array of shape [num_nodes] with y-coordinates or 0
        """
        contact_array = np.zeros(self.num_nodes, dtype=np.float32)
        
        # Extract contact info
        contact_indices, contact_positions = get_contact_info(entry)
        
        if not contact_indices or not contact_positions:
            return contact_array
        
        # Remap indices if requested
        if use_remapping:
            original_indices = contact_indices
            remapped_indices = remap_contact_indices(contact_indices, self.index_mapping)
            
            # Match positions with remapped indices
            for i, orig_idx in enumerate(original_indices):
                if orig_idx in self.index_mapping:
                    seq_idx = self.index_mapping[orig_idx]
                    if i < len(contact_positions) and len(contact_positions[i]) >= 2:
                        # Fill with y-coordinate (index 1)
                        contact_array[seq_idx] = contact_positions[i][1]
        else:
            # Use original indices directly
            for idx, pos in zip(contact_indices, contact_positions):
                if 0 <= idx < self.num_nodes and len(pos) >= 2:
                    contact_array[idx] = pos[1]  # y-coordinate
        
        return contact_array
    
    def process_json_file(self, json_path, use_remapping=True):
        """
        Process entire JSON file to contact arrays.
        
        Args:
            json_path (str): Path to JSON file
            use_remapping (bool): Whether to remap indices
        
        Returns:
            torch.Tensor: Shape [num_entries, num_nodes]
        """
        import json
        
        with open(json_path, 'r') as f:
            data_entries = json.load(f)
        
        contact_arrays = []
        for entry in data_entries:
            contact_array = self.process_contact_entry(entry, use_remapping)
            contact_arrays.append(contact_array)
        
        return torch.tensor(np.array(contact_arrays), dtype=torch.float32)
    
    def visualize_contact_array(self, contact_array, title="Contact Array"):
        """
        Visualize a contact array.
        
        Args:
            contact_array (numpy.ndarray or torch.Tensor): Contact array to visualize
            title (str): Plot title
        """
        import matplotlib.pyplot as plt
        
        if isinstance(contact_array, torch.Tensor):
            contact_array = contact_array.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Line plot of y-coordinates
        ax1.plot(contact_array, 'o-', markersize=3)
        ax1.set_xlabel('Sequential Node Index')
        ax1.set_ylabel('Y-coordinate (0 if no contact)')
        ax1.set_title(f'{title} - Y Coordinates')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bar plot showing contact vs non-contact
        contact_mask = contact_array != 0
        colors = ['red' if c else 'gray' for c in contact_mask]
        ax2.bar(range(len(contact_array)), np.abs(contact_array), color=colors, alpha=0.6)
        ax2.set_xlabel('Sequential Node Index')
        ax2.set_ylabel('|Y-coordinate|')
        ax2.set_title(f'{title} - Contact Points (red) vs Non-contact (gray)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        num_contact = contact_mask.sum()
        print(f"\nContact Statistics:")
        print(f"  Total nodes: {len(contact_array)}")
        print(f"  In contact: {num_contact} ({100*num_contact/len(contact_array):.1f}%)")
        print(f"  Not in contact: {len(contact_array) - num_contact}")
        if num_contact > 0:
            print(f"  Y-coord range: [{contact_array[contact_mask].min():.4f}, {contact_array[contact_mask].max():.4f}]")


def train_contact_encoder(encoder, dataloader, num_epochs=50, lr=0.001, device='cuda'):
    """
    Pre-train the contact encoder as an autoencoder.
    
    Args:
        encoder (ContactEncoder): Encoder model
        dataloader (DataLoader): DataLoader with contact arrays
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to train on
    
    Returns:
        ContactEncoder: Trained encoder
    """
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("Pre-training ContactEncoder as autoencoder...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        encoder.train()
        for batch_data in dataloader:
            # Unpack batch - TensorDataset returns tuple
            if isinstance(batch_data, (list, tuple)):
                contact_arrays = batch_data[0]
            else:
                contact_arrays = batch_data
            
            contact_arrays = contact_arrays.to(device)
            
            # Forward pass
            encoding, reconstruction = encoder.encode_decode(contact_arrays)
            
            # Compute loss
            loss = criterion(reconstruction, contact_arrays)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print("Pre-training complete!")
    return encoder


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    
    print("="*60)
    print("Contact Encoder Example")
    print("="*60)
    
    # Initialize processor
    processor = ContactDataProcessor(base_dir="finray-sim2real_ext_models")
    
    # Process a JSON file
    json_path = "finray-sim2real_ext_models/sim_data/data_pointy1/pointy1_sim_normalized.json"
    if Path(json_path).exists():
        print(f"\nProcessing: {json_path}")
        contact_arrays = processor.process_json_file(json_path, use_remapping=True)
        print(f"Processed shape: {contact_arrays.shape}")
        
        # Visualize first entry
        print("\nVisualizing first entry...")
        processor.visualize_contact_array(contact_arrays[0], title="First Entry Contact")
        
        # Create encoder
        print(f"\nCreating ContactEncoder...")
        encoder = ContactEncoder(
            num_top_surface_nodes=processor.num_nodes,
            bottleneck_dim=128,
            hidden_dims=[256, 512, 256]
        )
        print(f"Encoder input size: {processor.num_nodes}")
        print(f"Encoder output size: 128")
        print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        sample_batch = contact_arrays[:4]  # Take first 4 samples
        with torch.no_grad():
            encoding = encoder(sample_batch)
            print(f"Input shape: {sample_batch.shape}")
            print(f"Encoding shape: {encoding.shape}")
            
            # Test autoencoder
            encoding, reconstruction = encoder.encode_decode(sample_batch)
            print(f"Reconstruction shape: {reconstruction.shape}")
            
            # Compute reconstruction error
            mse = torch.mean((sample_batch - reconstruction) ** 2)
            print(f"Reconstruction MSE (untrained): {mse:.6f}")
        
        # Optional: Pre-train the encoder
        print("\n" + "="*60)
        print("Pre-training encoder (optional)")
        print("="*60)
        
        # Create dataloader
        dataset = TensorDataset(contact_arrays)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train for a few epochs as demonstration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = train_contact_encoder(encoder, dataloader, num_epochs=500, lr=0.001, device=device)
        
        # Test after training
        print("\nTesting after training...")
        encoder.eval()
        with torch.no_grad():
            sample_batch = contact_arrays[:4].to(device)
            encoding, reconstruction = encoder.encode_decode(sample_batch)
            mse = torch.mean((sample_batch - reconstruction) ** 2)
            print(f"Reconstruction MSE (trained): {mse:.6f}")
        
        # Visualize reconstruction
        print("\nVisualizing reconstruction...")
        processor.visualize_contact_array(sample_batch[0].cpu(), title="Original")
        processor.visualize_contact_array(reconstruction[0].cpu(), title="Reconstructed")
    
    else:
        print(f"JSON file not found: {json_path}")
        print("Please check the file path.")