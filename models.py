import torch
import torch.nn as nn

class EnergyNet(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) to model the energy function.
    It takes a 2D coordinate as input and outputs a single scalar energy value.
    一个简单的多层感知机（MLP），用于建模能量函数。
    它接收一个二维坐标作为输入，并输出一个标量能量值。
    """
    def __init__(self, input_dim=2, hidden_dim=128):
        """
        Args:
            input_dim (int): The dimensionality of the input data (should be 2 for toy data).
                             输入数据的维度（对于玩具数据应为 2）。
            hidden_dim (int): The number of neurons in the hidden layers.
                              隐藏层中的神经元数量。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Computes the energy for a given input.
        为给定的输入计算能量。

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2).
                              形状为 (batch_size, 2) 的输入张量。

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the energy values.
                          一个形状为 (batch_size, 1) 的张量，包含能量值。
        """
        return self.net(x)
