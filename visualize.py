import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_landscape(energy_network, data_samples, neg_samples=None, ax=None, title="", range_limit=8):
    """
    Plots the 2D energy landscape learned by the model, along with data samples.
    绘制模型学习到的二维能量曲面，以及数据样本。

    Args:
        energy_network (torch.nn.Module): The trained EBM.
                                          训练好的 EBM。
        data_samples (torch.Tensor): The ground truth data points.
                                     真实的（正）数据点。
        neg_samples (torch.Tensor, optional): Model-generated negative samples. Defaults to None.
                                              模型生成的负样本。默认为 None。
        ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot on. Defaults to None.
                                             用于绘图的 matplotlib 坐标轴。默认为 None。
        title (str, optional): The title for the plot. Defaults to "".
                               图表的标题。默认为 ""。
        range_limit (int, optional): The limit for the x and y axes. Defaults to 4.
                                     x 和 y 轴的范围限制。默认为 4。
    """
    if ax is None:
        _, ax = plt.subplots()

    # Create a grid of points to evaluate the energy function on
    # 创建一个点网格，用于评估能量函数
    x = np.linspace(-range_limit, range_limit, 100)
    y = np.linspace(-range_limit, range_limit, 100)
    xx, yy = np.meshgrid(x, y)
    grid_points = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32))
    grid_points = grid_points.to(next(energy_network.parameters()).device)

    # Evaluate the energy function on the grid
    # 在网格上评估能量函数
    with torch.no_grad():
        energy_values = energy_network(grid_points).cpu().numpy().reshape(xx.shape)

    # Plot the energy landscape
    # 绘制能量曲面
    contour = ax.contourf(xx, yy, energy_values, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Energy')

    # Plot the positive (real) data samples
    # 绘制正（真实）数据样本
    ax.scatter(
        data_samples[:, 0],
        data_samples[:, 1],
        s=10, alpha=0.7, c='white', edgecolors='black', label='Positive Samples'
    )

    # Plot the negative (generated) samples if provided
    # 如果提供了负（生成）样本，则绘制它们
    if neg_samples is not None:
        ax.scatter(
            neg_samples[:, 0],
            neg_samples[:, 1],
            s=10, alpha=0.7, c='red', edgecolors='black', label='Negative Samples'
        )

    ax.set_xlim([-range_limit, range_limit])
    ax.set_ylim([-range_limit, range_limit])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
