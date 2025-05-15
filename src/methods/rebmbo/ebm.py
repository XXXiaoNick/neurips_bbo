import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EnergyNetwork(nn.Module):
    """Energy network for EBM model"""
    def __init__(self, input_dim, hidden_dims=[128, 128]):
        super(EnergyNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Compute energy of input"""
        return self.network(x).squeeze()


class EnergyBasedModel:
    """
    Energy-Based Model
    
    Uses short-run MCMC to train an energy network that captures the global structure
    of the target function and provides exploration signals for BO.
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dims=[128, 128], 
                 mcmc_steps=20, 
                 step_size=0.01, 
                 noise_scale=0.01,
                 learning_rate=0.001,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize energy-based model
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            mcmc_steps: Number of MCMC steps
            step_size: MCMC step size
            noise_scale: MCMC noise scale
            learning_rate: Learning rate
            device: Computation device
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.mcmc_steps = mcmc_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.device = device
        
        self.network = EnergyNetwork(input_dim, hidden_dims).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.losses = []
        
    def energy(self, x):
        """
        Compute energy value for given input
        
        Args:
            x: Input point with shape (batch_size, input_dim) or (input_dim,)
        
        Returns:
            Energy value
        """
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            energy_values = self.network(x)
            return energy_values.cpu().numpy()
    
    def train(self, data_points, bounds, epochs=10, batch_size=32):
        """
        Train energy model using short-run MCMC
        
        Args:
            data_points: Training data points as [(x_1, y_1), ..., (x_n, y_n)]
            bounds: Bounds for each dimension [(lower_1, upper_1), ..., (lower_d, upper_d)]
            epochs: Number of training epochs
            batch_size: Batch size
        """
        X = np.array([x for x, _ in data_points])
        y = np.array([y for _, y in data_points])
        
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        weights = 1.0 - y_normalized
        weights = torch.FloatTensor(weights).to(self.device)
        
        for epoch in range(epochs):
            self.network.train()
            
            all_indices = np.arange(len(X))
            np.random.shuffle(all_indices)
            
            total_loss = 0.0
            num_batches = int(np.ceil(len(X) / batch_size))
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                batch_indices = all_indices[start_idx:end_idx]
                
                pos_samples = X_tensor[batch_indices]
                pos_weights = weights[batch_indices]
                
                neg_samples = self._short_run_mcmc(pos_samples.clone(), bounds)
                
                pos_energy = self.network(pos_samples)
                neg_energy = self.network(neg_samples)
                
                pos_loss = torch.mean(pos_energy * pos_weights)
                neg_loss = -torch.mean(neg_energy)
                
                loss = pos_loss + neg_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            self.losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.losses
    
    def _short_run_mcmc(self, initial_samples, bounds):
        """
        Perform short-run MCMC sampling
        
        Args:
            initial_samples: Initial sampling points
            bounds: Boundaries
        
        Returns:
            Points after MCMC sampling
        """
        samples = initial_samples.clone().detach().requires_grad_(True)
        
        for _ in range(self.mcmc_steps):
            if samples.grad is not None:
                samples.grad.zero_()
                
            energy = self.network(samples)
            energy.sum().backward()
            
            with torch.no_grad():
                samples = samples - self.step_size * samples.grad
                
                noise = torch.randn_like(samples) * self.noise_scale
                samples = samples + noise
                
                for dim in range(samples.shape[1]):
                    lower, upper = bounds[dim]
                    samples[:, dim].clamp_(lower, upper)
            
            samples.requires_grad_(True)
        
        return samples.detach()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'losses': self.losses
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.network = EnergyNetwork(self.input_dim, self.hidden_dims).to(self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']