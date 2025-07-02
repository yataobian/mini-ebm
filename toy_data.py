import torch
import numpy as np
from sklearn.datasets import make_moons, make_blobs

def get_toy_data(name, n_samples):
    """
    Generates and returns specified 2D toy data.
    生成并返回指定的 2D 玩具数据。

    Args:
        name (str): The name of the dataset ('gmm', 'two_moons', 'checkerboard').
                    数据集的名称 ('gmm', 'two_moons', 'checkerboard')。
        n_samples (int): The number of samples to generate.
                         要生成的样本数量。

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the data.
                      一个形状为 (n_samples, 2) 的张量，包含数据。
    """
    if name == 'gmm':
        # Gaussian Mixture Model with 8 modes
        # 具有 8 个模式的混合高斯模型
        centers = [
            (1, 2.5), (-1, 2.5), (2.5, 1), (2.5, -1),
            (1, -2.5), (-1, -2.5), (-2.5, 1), (-2.5, -1)
        ]
        data, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.3, random_state=42)
    elif name == 'two_moons':
        # Two Moons dataset
        # 双月数据集
        data, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
        data = data * 2 # Scale the data for better visualization
    elif name == 'checkerboard':
        # Checkerboard pattern
        # 棋盘格模式
        n_points = int(np.ceil(np.sqrt(n_samples)))
        x = np.linspace(-4, 4, n_points)
        y = np.linspace(-4, 4, n_points)
        xv, yv = np.meshgrid(x, y)
        data = np.stack([xv.ravel(), yv.ravel()], axis=-1)
        # Select points based on the checkerboard pattern
        # 根据棋盘格模式选择点
        ix = (np.floor(data[:, 0]) % 2 == 0) & (np.floor(data[:, 1]) % 2 == 0)
        iy = (np.floor(data[:, 0]) % 2!= 0) & (np.floor(data[:, 1]) % 2!= 0)
        data = data[ix | iy]
        # Subsample to get the desired number of points
        # 子采样以获得所需数量的点
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            data = data[indices]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return torch.from_numpy(data.astype(np.float32))
