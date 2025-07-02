import torch
import torch.nn.functional as F
import numpy as np

class DenoisingScoreMatchingLoss:
    """
    Implements the Denoising Score Matching (DSM) loss.
    This loss trains the model to predict the score (gradient of log-probability)
    of a noised data distribution, which avoids calculating the partition function.
    实现了去噪分数匹配 (DSM) 损失。
    该损失训练模型来预测带噪数据分布的分数（对数概率的梯度），从而避免了计算配分函数。
    
    References / 参考文献:
    - Vincent, P. (2011). A connection between score matching and denoising autoencoders. 
      Neural computation, 23(7), 1661-1674.
    - Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. 
      Journal of machine learning research, 6, 695-709.
    - Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. 
      In Advances in neural information processing systems (pp. 11918-11930).
    """
    def __init__(self, energy_network, sigma=0.1):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
                                              EBM 模型。
            sigma (float): The standard deviation of the Gaussian noise for corruption.
                           用于损坏数据的高斯噪声的标准差。
        """
        self.energy_network = energy_network
        self.sigma = sigma

    def __call__(self, samples):
        """
        Calculates the DSM loss.
        计算 DSM 损失。

        Args:
            samples (torch.Tensor): A batch of clean data samples.
                                    一批干净的数据样本。

        Returns:
            torch.Tensor: The scalar DSM loss.
                          标量 DSM 损失。
        """
        # Add Gaussian noise to the clean samples.
        # 向干净的样本中添加高斯噪声。
        noisy_samples = samples + torch.randn_like(samples) * self.sigma
        noisy_samples.requires_grad_(True)

        # The model's score is -∇x E(x). We compute this gradient.
        # 模型的得分是 -∇_x E(x)。我们计算这个梯度。
        energy = self.energy_network(noisy_samples).sum()
        
        # create_graph=True is essential for this method.
        # create_graph=True 对此方法至关重要。
        grad_energy = torch.autograd.grad(energy, noisy_samples, create_graph=True)[0]
        
        # The target score for a Gaussian noise distribution is -(noisy_x - clean_x) / sigma^2.
        # The loss is the squared L2 norm of the difference between the model's score
        # and the target score.
        # 对于高斯噪声分布，目标分数是 -(noisy_x - clean_x) / sigma^2。
        # 损失是模型分数与目标分数之差的 L2 范数的平方。
        target = (samples - noisy_samples) / (self.sigma**2)
        
        # We want our model's score component (-grad_energy) to match the target score component.
        # 我们希望模型的分数部分 (-grad_energy) 与目标分数部分匹配。
        loss = 0.5 * ((grad_energy + target)**2).sum(dim=-1).mean()
        
        return loss 