import torch
import torch.nn.functional as F
import numpy as np

class NCELoss:
    """
    Implements the Noise Contrastive Estimation (NCE) loss.
    This reframes density estimation as a binary classification problem: discriminating
    between real data and samples from a known noise distribution.
    实现了噪声对比估计 (NCE) 损失。
    这将密度估计问题重构为一个二元分类问题：区分真实数据和来自已知噪声分布的样本。
    
    References / 参考文献:
    - Gutmann, M., & Hyvärinen, A. (2010). Noise-contrastive estimation: A new estimation 
      principle for unnormalized statistical models. In Proceedings of the thirteenth 
      international conference on artificial intelligence and statistics (pp. 297-304).
    - Gutmann, M. U., & Hyvärinen, A. (2012). Noise-contrastive estimation of unnormalized 
      statistical models, with applications to natural image statistics. Journal of machine 
      learning research, 13(Feb), 307-361.
    - Mnih, A., & Teh, Y. W. (2012). A fast and simple algorithm for training neural 
      probabilistic language models. In Proceedings of the 29th international conference 
      on international conference on machine learning (pp. 1751-1758).
    - Pihlaja, M., Gutmann, M., & Hyvärinen, A. (2010). A family of computationally efficient 
      and simple estimators for unnormalized statistical models. In Proceedings of the twenty-sixth 
      conference on uncertainty in artificial intelligence (pp. 442-449).
    - Ma, Z., & Collins, M. (2018). Noise contrastive estimation and negative sampling for 
      conditional models: Consistency and statistical efficiency. In Proceedings of the 2018 
      conference on empirical methods in natural language processing (pp. 3698-3707).
    """
    def __init__(self, energy_network, noise_dist, noise_ratio=1, self_normalized=True):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
                                              EBM 模型。
            noise_dist (torch.distributions.Distribution): A known noise distribution
                                                           to sample from (e.g., Normal).
                                                           一个已知的用于采样的噪声分布（例如，正态分布）。
            noise_ratio (int): The number of noise samples per data sample.
                               每个数据样本对应的噪声样本数量。
            self_normalized (bool): Whether to assume the model is self-normalized (Z=1).
                                   是否假设模型是自归一化的（Z=1）。
        """
        self.energy_network = energy_network
        self.noise_dist = noise_dist
        self.noise_ratio = noise_ratio
        self.self_normalized = self_normalized
        
        # Learnable log partition function (if not self-normalized)
        # 可学习的对数配分函数（如果不是自归一化）
        if not self_normalized:
            self.log_partition = torch.nn.Parameter(torch.zeros(1))

    def __call__(self, data_samples):
        """
        Calculates the NCE loss.
        计算 NCE 损失。

        Args:
            data_samples (torch.Tensor): A batch of real data samples.
                                         一批真实数据样本。

        Returns:
            torch.Tensor: The scalar NCE loss.
                          标量 NCE 损失。
        """
        batch_size = data_samples.shape[0]
        device = data_samples.device

        # Generate noise samples from the known noise distribution.
        # 从已知的噪声分布中生成噪声样本。
        noise_samples = self.noise_dist.sample((batch_size * self.noise_ratio,)).to(device)
        
        # Compute energy values
        # 计算能量值
        data_energies = self.energy_network(data_samples)
        noise_energies = self.energy_network(noise_samples)
        
        # Ensure energies are 1D tensors
        # 确保能量值是一维tensor
        if data_energies.dim() > 1:
            data_energies = data_energies.squeeze(-1)
        if noise_energies.dim() > 1:
            noise_energies = noise_energies.squeeze(-1)
        
        # Compute noise distribution log probabilities
        # 计算噪声分布的对数概率
        data_noise_logprobs = self.noise_dist.log_prob(data_samples)
        if len(data_noise_logprobs.shape) > 1:
            data_noise_logprobs = data_noise_logprobs.sum(dim=tuple(range(1, len(data_noise_logprobs.shape))))
        
        noise_noise_logprobs = self.noise_dist.log_prob(noise_samples)
        if len(noise_noise_logprobs.shape) > 1:
            noise_noise_logprobs = noise_noise_logprobs.sum(dim=tuple(range(1, len(noise_noise_logprobs.shape))))
        
        # Compute model log probabilities (unnormalized)
        # 计算模型对数概率（未归一化）
        if self.self_normalized:
            data_model_logprobs = -data_energies
            noise_model_logprobs = -noise_energies
        else:
            data_model_logprobs = -data_energies + self.log_partition
            noise_model_logprobs = -noise_energies + self.log_partition
        
        # Compute NCE logits: log(p_model(x) / (nu * p_noise(x)))
        # 计算NCE logits: log(p_model(x) / (nu * p_noise(x)))
        data_logits = data_model_logprobs - data_noise_logprobs - np.log(self.noise_ratio)
        noise_logits = noise_model_logprobs - noise_noise_logprobs - np.log(self.noise_ratio)
        
        # Compute NCE probabilities
        # 计算NCE概率
        # P(D=1|x) = p_model(x) / (p_model(x) + nu * p_noise(x))
        data_nce_probs = torch.sigmoid(data_logits)
        # P(D=0|x) = nu * p_noise(x) / (p_model(x) + nu * p_noise(x))
        noise_nce_probs = torch.sigmoid(-noise_logits)
        
        # Compute NCE loss
        # 计算NCE损失
        data_loss = -torch.log(data_nce_probs + 1e-8).mean()
        noise_loss = -torch.log(noise_nce_probs + 1e-8).mean()
        
        total_loss = data_loss + noise_loss
        return total_loss



class AdaptiveNCELoss:
    """
    Adaptive NCE that learns the noise distribution parameters.
    自适应NCE，学习噪声分布参数。
    """
    def __init__(self, energy_network, initial_noise_dist, noise_ratio=1, 
                 noise_lr=0.01, self_normalized=True):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
            initial_noise_dist: Initial noise distribution.
            noise_ratio (int): The number of noise samples per data sample.
            noise_lr (float): Learning rate for noise distribution parameters.
            self_normalized (bool): Whether to assume self-normalization.
        """
        self.energy_network = energy_network
        self.noise_ratio = noise_ratio
        self.noise_lr = noise_lr
        self.self_normalized = self_normalized
        
        # Initialize learnable noise distribution parameters
        # 初始化可学习的噪声分布参数
        self.noise_mean = torch.nn.Parameter(torch.zeros_like(initial_noise_dist.mean))
        self.noise_log_std = torch.nn.Parameter(torch.log(initial_noise_dist.stddev))
        
        if not self_normalized:
            self.log_partition = torch.nn.Parameter(torch.zeros(1))
    
    def get_noise_dist(self):
        """Get current noise distribution."""
        from torch.distributions import Normal
        return Normal(self.noise_mean, torch.exp(self.noise_log_std))
    
    def __call__(self, data_samples):
        """
        Calculates the adaptive NCE loss.
        计算自适应NCE损失。
        """
        batch_size = data_samples.shape[0]
        device = data_samples.device
        
        # Get current noise distribution
        noise_dist = self.get_noise_dist()
        
        # Generate noise samples
        noise_samples = noise_dist.sample((batch_size * self.noise_ratio,)).to(device)
        
        # Compute energies
        data_energies = self.energy_network(data_samples)
        noise_energies = self.energy_network(noise_samples)
        
        # Ensure energies are 1D tensors
        # 确保能量值是一维tensor
        if data_energies.dim() > 1:
            data_energies = data_energies.squeeze(-1)
        if noise_energies.dim() > 1:
            noise_energies = noise_energies.squeeze(-1)
        
        # Compute noise log probabilities
        data_noise_logprobs = noise_dist.log_prob(data_samples)
        if len(data_noise_logprobs.shape) > 1:
            data_noise_logprobs = data_noise_logprobs.sum(dim=tuple(range(1, len(data_noise_logprobs.shape))))
        
        noise_noise_logprobs = noise_dist.log_prob(noise_samples)
        if len(noise_noise_logprobs.shape) > 1:
            noise_noise_logprobs = noise_noise_logprobs.sum(dim=tuple(range(1, len(noise_noise_logprobs.shape))))
        
        # Compute model log probabilities
        if self.self_normalized:
            data_model_logprobs = -data_energies
            noise_model_logprobs = -noise_energies
        else:
            data_model_logprobs = -data_energies + self.log_partition
            noise_model_logprobs = -noise_energies + self.log_partition
        
        # NCE logits
        data_logits = data_model_logprobs - data_noise_logprobs - np.log(self.noise_ratio)
        noise_logits = noise_model_logprobs - noise_noise_logprobs - np.log(self.noise_ratio)
        
        # NCE probabilities and loss
        data_nce_probs = torch.sigmoid(data_logits)
        noise_nce_probs = torch.sigmoid(-noise_logits)
        
        nce_loss = -torch.log(data_nce_probs + 1e-8).mean() - torch.log(noise_nce_probs + 1e-8).mean()
        
        # Additional loss to adapt noise distribution (minimize KL divergence)
        # 附加损失以适应噪声分布（最小化KL散度）
        noise_adaptation_loss = -data_noise_logprobs.mean()
        
        total_loss = nce_loss + self.noise_lr * noise_adaptation_loss
        
        return total_loss 