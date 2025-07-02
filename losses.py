import torch
import torch.nn.functional as F

class ContrastiveDivergenceLoss:
    """
    Implements the Contrastive Divergence (CD-k) loss.
    This loss pushes down the energy of real data (positive samples) and pushes up
    the energy of model-generated data (negative samples).
    实现了对比散度 (CD-k) 损失。
    该损失降低真实数据（正样本）的能量，并提升模型生成数据（负样本）的能量。
    """
    def __init__(self, energy_network, sampler, k=1):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
                                              EBM 模型。
            sampler (LangevinSampler): The MCMC sampler to generate negative samples.
                                       用于生成负样本的 MCMC 采样器。
            k (int): The number of MCMC steps for CD (k in CD-k).
                     CD 的 MCMC 步数 (CD-k 中的 k)。
        """
        self.energy_network = energy_network
        self.sampler = sampler
        self.k = k
        self.last_negative_samples = None # For visualization / 用于可视化

    def __call__(self, positive_samples):
        """
        Calculates the CD loss.
        计算 CD 损失。

        Args:
            positive_samples (torch.Tensor): A batch of real data samples.
                                             一批真实数据样本。

        Returns:
            torch.Tensor: The scalar CD loss.
                          标量 CD 损失。
        """
        # Positive phase: compute energy of real data.
        # 正向阶段：计算真实数据的能量。
        positive_energy = self.energy_network(positive_samples)

        # Negative phase: generate "fantasy" samples via MCMC.
        # We detach positive_samples to prevent gradients from flowing back to the network
        # through the MCMC initialization.
        # 负向阶段：通过 MCMC 生成"幻想"样本。
        # 我们 detach positive_samples 以防止梯度通过 MCMC 的初始化回传到网络。
        negative_samples_start = positive_samples.detach()
        
        # The sampler runs k steps of Langevin Dynamics.
        # 采样器运行 k 步朗之万动力学。
        negative_samples = self.sampler.sample(negative_samples_start, n_steps=self.k)
        self.last_negative_samples = negative_samples # Save for visualization / 保存用于可视化
        
        # Compute energy of the fantasy data.
        # 计算幻想数据的能量。
        negative_energy = self.energy_network(negative_samples)

        # The CD loss is the difference between the mean energies.
        # We want to minimize positive_energy and maximize negative_energy.
        # CD 损失是平均能量之差。
        # 我们希望最小化 positive_energy 并最大化 negative_energy。
        loss = positive_energy.mean() - negative_energy.mean()
        
        return loss

class DenoisingScoreMatchingLoss:
    """
    Implements the Denoising Score Matching (DSM) loss.
    This loss trains the model to predict the score (gradient of log-probability)
    of a noised data distribution, which avoids calculating the partition function.
    实现了去噪分数匹配 (DSM) 损失。
    该损失训练模型来预测带噪数据分布的分数（对数概率的梯度），从而避免了计算配分函数。
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

class NoiseContrastiveEstimationLoss:
    """
    Implements the Noise Contrastive Estimation (NCE) loss.
    This reframes density estimation as a binary classification problem: discriminating
    between real data and samples from a known noise distribution.
    实现了噪声对比估计 (NCE) 损失。
    这将密度估计问题重构为一个二元分类问题：区分真实数据和来自已知噪声分布的样本。
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
