import torch
import numpy as np
import matplotlib.pyplot as plt
from visualize import plot_energy_landscape

def redraw_energy_landscape(fig, ax, energy_network, data_samples, neg_samples=None, title="", range_limit=8):
    """
    Clear and redraw an energy landscape on a given matplotlib axis.
    This is the key utility for widget callbacks to update plots smoothly in-place.

    在给定的 matplotlib 坐标轴上清除并重绘能量曲面。
    这是 widget 回调中原地平滑更新图像的关键工具。

    Args:
        fig: matplotlib figure object
        ax: matplotlib axes object
        energy_network: the EBM energy model
        data_samples: real data points
        neg_samples (optional): model-generated negative samples
        title: plot title
        range_limit: axis range for contour plot
    """
    ax.clear()
    plot_energy_landscape(energy_network, data_samples, neg_samples=neg_samples, ax=ax,
                          title=title, range_limit=range_limit)
    fig.canvas.draw_idle()


def plot_langevin_trajectory(trajectory, energy_network=None, data_samples=None, ax=None,
                             title="Langevin Sampling Trajectory", range_limit=8):
    """
    Plot the trajectory of particles undergoing Langevin dynamics.

    绘制 Langevin 动力学下粒子的运动轨迹。

    Args:
        trajectory: tensor of shape (n_steps+1, n_particles, 2) from LangevinSampler.sample(return_trajectory=True)
        energy_network (optional): if provided, plot energy contour background
        data_samples (optional): real data points to display
        ax: matplotlib axes to draw on (creates new if None)
        title: plot title
        range_limit: axis range
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Plot energy contour if energy_network is provided
    if energy_network is not None:
        plot_energy_landscape(energy_network, data_samples if data_samples is not None else torch.randn(10, 2),
                            ax=ax, title=title, range_limit=range_limit)

    # Extract trajectory as numpy; shape (n_steps+1, n_particles, 2)
    traj_np = trajectory.cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    n_steps, n_particles, dim = traj_np.shape

    # Plot each particle's path
    colors = plt.cm.Spectral(np.linspace(0, 1, n_particles))
    for p in range(n_particles):
        path = traj_np[:, p, :]  # shape (n_steps+1, 2)
        # Fade color over time (alpha increases)
        for t in range(n_steps):
            alpha = (t + 1) / (n_steps + 1)
            ax.plot(path[t:t+2, 0], path[t:t+2, 1], color=colors[p], alpha=alpha, linewidth=1.5)
        # Mark final position
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[p], s=50, marker='*',
                  edgecolors='black', linewidth=0.5)

    ax.set_xlim([-range_limit, range_limit])
    ax.set_ylim([-range_limit, range_limit])
    ax.set_aspect('equal', adjustable='box')


def plot_trajectory_step(trajectory, step, ax, energy_network=None, range_limit=8,
                        title="Langevin Chain (step-by-step)"):
    """
    Plot only up to a specific step in the trajectory (for scrubber interaction).

    只绘制轨迹中到特定步数的部分（用于滑块交互）。

    Args:
        trajectory: tensor of shape (n_steps+1, n_particles, 2)
        step: integer, which step to show up to (0 to n_steps)
        ax: matplotlib axes
        energy_network (optional): energy model for contour background
        range_limit: axis range
        title: plot title
    """
    ax.clear()

    if energy_network is not None:
        plot_energy_landscape(energy_network, torch.randn(10, 2), ax=ax,
                            title=title, range_limit=range_limit)

    traj_np = trajectory.cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    n_steps, n_particles, dim = traj_np.shape

    colors = plt.cm.Spectral(np.linspace(0, 1, n_particles))
    for p in range(n_particles):
        path = traj_np[:step+1, p, :]  # Only up to requested step
        # Plot path up to current step
        for t in range(step):
            alpha = 0.3 + 0.7 * (t + 1) / (step + 1)  # Fade from light to dark
            ax.plot(path[t:t+2, 0], path[t:t+2, 1], color=colors[p], alpha=alpha, linewidth=1.5)
        # Highlight current position
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[p], s=100, marker='o',
                  edgecolors='white', linewidth=1.5, zorder=5)

    ax.set_xlim([-range_limit, range_limit])
    ax.set_ylim([-range_limit, range_limit])
    ax.set_aspect('equal', adjustable='box')


def plot_score_field(energy_network, range_limit=8, grid_n=15, ax=None, title="Score Field (-∇E)"):
    """
    Plot the score field (negative gradient of energy) as a quiver plot.

    绘制分数场（能量的负梯度）作为箭头图。

    Args:
        energy_network: EBM model
        range_limit: axis range
        grid_n: number of grid points per side
        ax: matplotlib axes
        title: plot title
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Create grid
    x = np.linspace(-range_limit, range_limit, grid_n)
    y = np.linspace(-range_limit, range_limit, grid_n)
    xx, yy = np.meshgrid(x, y)
    grid_points = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32))
    grid_points = grid_points.to(next(energy_network.parameters()).device)
    grid_points.requires_grad = True

    # Compute energy at grid points
    energy_vals = energy_network(grid_points).sum()
    energy_vals.backward()

    # Extract gradients (direction of steepest ascent; we want negative gradient = descent direction)
    grad = grid_points.grad.cpu().numpy().reshape(grid_n, grid_n, 2)
    score = -grad  # Score = -∇E

    # Plot quiver field
    ax.quiver(xx, yy, score[:, :, 0], score[:, :, 1], alpha=0.7)
    ax.set_xlim([-range_limit, range_limit])
    ax.set_ylim([-range_limit, range_limit])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')


def plot_noise_vs_data(data_samples, noise_samples, ax=None, title="Real Data vs Noise Distribution"):
    """
    Scatter plot comparing real data and noise samples side by side.

    散点图对比真实数据和噪声样本。

    Args:
        data_samples: real data points
        noise_samples: samples from noise distribution
        ax: matplotlib axes
        title: plot title
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    data_np = data_samples.cpu().numpy() if isinstance(data_samples, torch.Tensor) else data_samples
    noise_np = noise_samples.cpu().numpy() if isinstance(noise_samples, torch.Tensor) else noise_samples

    ax.scatter(data_np[:, 0], data_np[:, 1], s=20, alpha=0.6, c='blue', label='Real Data')
    ax.scatter(noise_np[:, 0], noise_np[:, 1], s=20, alpha=0.6, c='red', label='Noise Distribution')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')


def plot_loss_curve(loss_history, ax=None, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
    """
    Plot a line graph of loss over epochs.

    绘制 epoch 上的损失曲线。

    Args:
        loss_history: list of loss values
        ax: matplotlib axes
        title: plot title
        xlabel, ylabel: axis labels
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(loss_history, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_methods_grid(energy_networks_dict, data_samples, range_limit=8, ncols=3, figsize=None):
    """
    Plot multiple energy landscapes in a grid for comparison.

    在网格中绘制多个能量曲面以便比较。

    Args:
        energy_networks_dict: dict of {title: energy_network}
        data_samples: real data points
        range_limit: axis range for each subplot
        ncols: number of columns in grid
        figsize: figure size (default auto)

    Returns:
        fig, axes
    """
    n_methods = len(energy_networks_dict)
    nrows = (n_methods + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols) if axes.ndim > 1 else axes.reshape(-1, 1)
    else:
        axes = axes.reshape(nrows, ncols)

    axes_flat = axes.flatten()

    for idx, (title, energy_net) in enumerate(energy_networks_dict.items()):
        ax = axes_flat[idx]
        plot_energy_landscape(energy_net, data_samples, ax=ax, title=title, range_limit=range_limit)

    # Hide unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    return fig, axes
