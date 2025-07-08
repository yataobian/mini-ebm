"""
NCE Implementation Comparison and Usage Examples
NCE实现比较和使用示例

This file demonstrates the differences between various NCE implementations
and provides usage examples for training EBMs.
这个文件演示了各种NCE实现之间的差异，并提供了训练EBM的使用示例。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal, Independent
import matplotlib.pyplot as plt
import numpy as np
from nce_variants import NCELoss, AdaptiveNCELoss


class SimpleEnergyNetwork(nn.Module):
    """Simple energy network for demonstration."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def generate_toy_data(n_samples=1000, dim=2):
    """Generate toy 2D data from a mixture of Gaussians."""
    # Mixture of two Gaussians
    component = torch.randint(0, 2, (n_samples,))
    
    means = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
    covs = torch.stack([torch.eye(2) * 0.5, torch.eye(2) * 0.5])
    
    samples = torch.zeros(n_samples, dim)
    for i in range(n_samples):
        dist = MultivariateNormal(means[component[i]], covs[component[i]])
        samples[i] = dist.sample()
    
    return samples


def train_ebm_comparison():
    """Compare different NCE implementations on toy data."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    data = generate_toy_data(1000, 2).to(device)
    base_dist = Normal(torch.zeros(2).to(device), torch.ones(2).to(device))
    noise_dist = Independent(base_dist, 1)
    
    # Models
    models = {
        'Correct NCE': (
            SimpleEnergyNetwork(2).to(device),
            NCELoss(
                energy_network=None,  # Will be set later
                noise_dist=noise_dist,
                noise_ratio=1,
                self_normalized=True
            )
        ),

        'Adaptive NCE': (
            SimpleEnergyNetwork(2).to(device),
            AdaptiveNCELoss(
                energy_network=None,  # Will be set later
                initial_noise_dist=noise_dist,
                noise_ratio=1,
                noise_lr=0.01,
                self_normalized=True
            )
        )
    }
    
    # Set energy networks in loss functions
    for name, (model, loss_fn) in models.items():
        loss_fn.energy_network = model
    
    # Training
    results = {}
    for name, (model, loss_fn) in models.items():
        print(f"\nTraining {name}...")
        
        # Collect all parameters for optimizer
        all_params = list(model.parameters())
        
        # Add partition function parameter if it exists
        if hasattr(loss_fn, 'log_partition') and loss_fn.log_partition is not None:
            all_params.append(loss_fn.log_partition)
        
        # Add noise distribution parameters if they exist
        if hasattr(loss_fn, 'noise_mean') and hasattr(loss_fn, 'noise_log_std'):
            all_params.extend([loss_fn.noise_mean, loss_fn.noise_log_std])
        
        optimizer = optim.Adam(all_params, lr=0.001)
        
        losses = []
        for epoch in range(200):
            # Mini-batch training
            batch_size = 64
            idx = torch.randperm(len(data))[:batch_size]
            batch = data[idx]
            
            optimizer.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 50 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
        
        results[name] = {
            'model': model,
            'loss_fn': loss_fn,
            'losses': losses
        }
    
    return results, data


def visualize_results(results, data):
    """Visualize the learned energy functions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot original data
    axes[0].scatter(data[:, 0].cpu(), data[:, 1].cpu(), alpha=0.6, s=20)
    axes[0].set_title('Original Data')
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    
    # Plot energy functions
    x = torch.linspace(-5, 5, 50)
    y = torch.linspace(-5, 5, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(data.device)
    
    for i, (name, result) in enumerate(results.items(), 1):
        model = result['model']
        model.eval()
        
        with torch.no_grad():
            energies = model(grid).cpu().reshape(50, 50)
        
        im = axes[i].contourf(X.cpu(), Y.cpu(), energies, levels=20, cmap='viridis')
        axes[i].scatter(data[:, 0].cpu(), data[:, 1].cpu(), 
                       alpha=0.6, s=10, c='red', marker='.')
        axes[i].set_title(f'{name} - Energy Function')
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-5, 5)
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('nce_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result['losses'], label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('nce_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def performance_benchmark():
    """Benchmark computational performance of different implementations."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup
    data = torch.randn(1000, 10).to(device)
    base_dist = Normal(torch.zeros(10).to(device), torch.ones(10).to(device))
    noise_dist = Independent(base_dist, 1)
    energy_net = SimpleEnergyNetwork(10).to(device)
    
    implementations = {
        'Correct': NCELoss(energy_net, noise_dist, noise_ratio=1),
        'Adaptive': AdaptiveNCELoss(energy_net, noise_dist, noise_ratio=1)
    }
    
    print("=== Performance Benchmark ===\n")
    
    import time
    
    for name, loss_fn in implementations.items():
        # Warmup
        for _ in range(10):
            _ = loss_fn(data[:64])
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            loss = loss_fn(data[:64])
            loss.backward()
        
        elapsed = time.time() - start_time
        print(f"{name:>12}: {elapsed:.4f}s (100 iterations)")


if __name__ == "__main__":
    
    # Performance benchmark
    performance_benchmark()
    
    
    # Train and compare models
    print("\n=== Training Comparison ===")
    results, data = train_ebm_comparison()
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, data)
    
    print("\nComparison complete! Check the generated plots.") 