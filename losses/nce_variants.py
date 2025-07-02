import torch
import torch.nn.functional as F
import numpy as np

class NoiseContrastiveEstimationLoss:
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
    def __init__(self, energy_network, noise_dist, noise_ratio=1):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
                                              EBM 模型。
            noise_dist (torch.distributions.Distribution): A known noise distribution
                                                           to sample from (e.g., Normal).
                                                           一个已知的用于采样的噪声分布（例如，正态分布）。
            noise_ratio (int): The number of noise samples per data sample.
                               每个数据样本对应的噪声样本数量。
        """
        self.energy_network = energy_network
        self.noise_dist = noise_dist
        self.noise_ratio = noise_ratio

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
        sample_shape = (batch_size * self.noise_ratio,) + data_samples.shape[1:]
        noise_samples = self.noise_dist.sample(sample_shape).to(device)
        
        # We use -Energy as the logit for the binary classifier.
        # 我们使用 -Energy 作为二元分类器的 logit。
        data_logits = -self.energy_network(data_samples).squeeze(-1)
        noise_logits = -self.energy_network(noise_samples).squeeze(-1)
        
        # Concatenate data and noise logits for binary classification.
        # 连接数据和噪声的 logits 以进行二元分类。
        all_logits = torch.cat([data_logits, noise_logits])
        
        # Create labels: 1 for real data, 0 for noise.
        # 创建标签：真实数据为 1，噪声为 0。
        data_labels = torch.ones_like(data_logits)
        noise_labels = torch.zeros_like(noise_logits)
        all_labels = torch.cat([data_labels, noise_labels])
        
        # Use Binary Cross-Entropy with Logits loss, which is numerically stable.
        # 使用带 Logits 的二元交叉熵损失，这在数值上更稳定。
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        
        return loss 