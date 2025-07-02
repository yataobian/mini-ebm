import torch
import torch.nn.functional as F
import numpy as np

class ContrastiveDivergenceLoss:
    """
    Implements the Contrastive Divergence (CD-k) loss.
    This loss pushes down the energy of real data (positive samples) and pushes up
    the energy of model-generated data (negative samples).
    实现了对比散度 (CD-k) 损失。
    该损失降低真实数据（正样本）的能量，并提升模型生成数据（负样本）的能量。
    
    References / 参考文献:
    - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. 
      Neural computation, 14(8), 1771-1800.
    - Carreira-Perpinan, M. A., & Hinton, G. E. (2005). On contrastive divergence learning. 
      In Proceedings of the tenth international workshop on artificial intelligence and statistics 
      (pp. 33-40).
    - Tieleman, T. (2008). Training restricted Boltzmann machines using approximations 
      to the likelihood gradient. In Proceedings of the 25th international conference 
      on Machine learning (pp. 1064-1071).
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

class PersistentContrastiveDivergenceLoss:
    """
    Implements Persistent Contrastive Divergence (PCD) loss.
    Unlike standard CD, PCD maintains a persistent set of negative samples across training steps,
    allowing the Markov chain to mix better and explore the energy landscape more thoroughly.
    
    实现了持久对比散度 (PCD) 损失。
    与标准 CD 不同，PCD 在训练步骤中维护一组持久的负样本，
    允许马尔可夫链更好地混合并更彻底地探索能量景观。
    
    References / 参考文献:
    - Tieleman, T. (2008). Training restricted Boltzmann machines using approximations 
      to the likelihood gradient. In Proceedings of the 25th international conference 
      on Machine learning (pp. 1064-1071).
    - Tieleman, T., & Hinton, G. (2009). Using fast weights to improve persistent 
      contrastive divergence. In Proceedings of the 26th annual international conference 
      on machine learning (pp. 1033-1040).
    """
    def __init__(self, energy_network, sampler, k=1, n_persistent=100, buffer_init_std=1.0):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
                                              EBM 模型。
            sampler (LangevinSampler): The MCMC sampler to generate negative samples.
                                       用于生成负样本的 MCMC 采样器。
            k (int): The number of MCMC steps for each update.
                     每次更新的 MCMC 步数。
            n_persistent (int): Number of persistent particles to maintain.
                                要维护的持久粒子数量。
            buffer_init_std (float): Standard deviation for initializing the persistent buffer.
                                     初始化持久缓冲区的标准差。
        """
        self.energy_network = energy_network
        self.sampler = sampler
        self.k = k
        self.n_persistent = n_persistent
        self.buffer_init_std = buffer_init_std
        
        # Persistent particle buffer - initialized when first called
        # 持久粒子缓冲区 - 首次调用时初始化
        self.persistent_particles = None
        self.last_negative_samples = None  # For visualization / 用于可视化

    def _initialize_buffer(self, sample_shape, device):
        """
        Initialize the persistent particle buffer.
        初始化持久粒子缓冲区。
        """
        # Buffer Initialization: The code initializes the buffer from a standard normal distribution. 
        # For some datasets, it might be better to initialize it from a random subset of the training data to start the chains in a more plausible region.
        # 缓冲区初始化：代码从标准正态分布初始化缓冲区。
        # 对于某些数据集，最好从训练数据的一个随机子集初始化，以在更合理区域启动链。
        buffer_shape = (self.n_persistent,) + sample_shape[1:]
        self.persistent_particles = torch.randn(buffer_shape, device=device) * self.buffer_init_std

    def __call__(self, positive_samples):
        """
        Calculates the PCD loss.
        计算 PCD 损失。

        Args:
            positive_samples (torch.Tensor): A batch of real data samples.
                                             一批真实数据样本。

        Returns:
            torch.Tensor: The scalar PCD loss.
                          标量 PCD 损失。
        """
        device = positive_samples.device
        batch_size = positive_samples.shape[0]
        
        # Initialize persistent buffer if not already done
        # 如果尚未完成，则初始化持久缓冲区
        if self.persistent_particles is None:
            self._initialize_buffer(positive_samples.shape, device)
        
        # Positive phase: compute energy of real data
        # 正向阶段：计算真实数据的能量
        positive_energy = self.energy_network(positive_samples)
        
        # Negative phase: sample from persistent particles
        # 负向阶段：从持久粒子中采样
        # Randomly select particles from the buffer
        # 从缓冲区随机选择粒子
        indices = torch.randint(0, self.n_persistent, (batch_size,))
        assert self.persistent_particles is not None  # Type hint for linter
        selected_particles = self.persistent_particles[indices].clone()
        
        # Run k steps of MCMC from selected particles
        # 从选定的粒子运行 k 步 MCMC
        negative_samples = self.sampler.sample(selected_particles, n_steps=self.k)
        self.last_negative_samples = negative_samples  # Save for visualization / 保存用于可视化
        
        # Update the persistent buffer with new samples
        # 用新样本更新持久缓冲区
        assert self.persistent_particles is not None  # Type hint for linter
        self.persistent_particles[indices] = negative_samples.detach()
        
        # Compute energy of the fantasy data
        # 计算幻想数据的能量
        negative_energy = self.energy_network(negative_samples)
        
        # The PCD loss is the difference between the mean energies
        # PCD 损失是平均能量之差
        loss = positive_energy.mean() - negative_energy.mean()
        
        return loss


class FastPersistentContrastiveDivergenceLoss:
    """
    Implements Fast Persistent Contrastive Divergence (Fast PCD) loss.
    This variant uses multiple parallel chains and occasional random restarts
    to improve mixing and avoid getting stuck in local modes.
    
    实现了快速持久对比散度 (Fast PCD) 损失。
    该变体使用多个并行链和偶尔的随机重启来改善混合并避免陷入局部模式。
    
    References / 参考文献:
    - Tieleman, T., & Hinton, G. (2009). Using fast weights to improve persistent 
      contrastive divergence. In Proceedings of the 26th annual international conference 
      on machine learning (pp. 1033-1040).
    """
    def __init__(self, energy_network, sampler, k=1, n_chains=20, restart_prob=0.01, buffer_init_std=1.0):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model.
                                              EBM 模型。
            sampler (LangevinSampler): The MCMC sampler.
                                       MCMC 采样器。
            k (int): Number of MCMC steps per update.
                     每次更新的 MCMC 步数。
            n_chains (int): Number of parallel chains to maintain.
                            要维护的并行链数量。
            restart_prob (float): Probability of randomly restarting a chain.
                                  随机重启链的概率。
            buffer_init_std (float): Standard deviation for buffer initialization.
                                     缓冲区初始化的标准差。
        """
        self.energy_network = energy_network
        self.sampler = sampler
        self.k = k
        self.n_chains = n_chains
        self.restart_prob = restart_prob
        self.buffer_init_std = buffer_init_std
        
        self.parallel_chains = None
        self.last_negative_samples = None

    def _initialize_chains(self, sample_shape, device):
        """
        Initialize parallel chains.
        初始化并行链。
        """
        chain_shape = (self.n_chains,) + sample_shape[1:]
        self.parallel_chains = torch.randn(chain_shape, device=device) * self.buffer_init_std

    def __call__(self, positive_samples):
        """
        Calculates the Fast PCD loss.
        计算 Fast PCD 损失。
        
        Comment: Restart Strategy: Instead of re-initializing to pure random noise (torch.randn_like), restarts could be initialized from the current batch of positive samples, which can sometimes be more effective.
        """
        device = positive_samples.device
        batch_size = positive_samples.shape[0]
        
        # Initialize chains if needed
        # 如果需要，初始化链
        if self.parallel_chains is None:
            self._initialize_chains(positive_samples.shape, device)
        
        # Positive phase
        # 正向阶段
        positive_energy = self.energy_network(positive_samples)
        
        # Random restarts for some chains
        # 某些链的随机重启
        restart_mask = torch.rand(self.n_chains) < self.restart_prob
        if restart_mask.any():
            assert self.parallel_chains is not None  # Type hint for linter
            # Create restart samples by selecting relevant chains and randomizing them
            # 通过选择相关链并随机化来创建重启样本
            restart_samples = torch.randn_like(self.parallel_chains[restart_mask]) * self.buffer_init_std
            self.parallel_chains[restart_mask] = restart_samples
        
        # Sample from all chains
        # 从所有链中采样
        updated_chains = self.sampler.sample(self.parallel_chains, n_steps=self.k)
        self.parallel_chains = updated_chains.detach()
        
        # Select samples for this batch
        # 为此批次选择样本
        indices = torch.randint(0, self.n_chains, (batch_size,))
        negative_samples = updated_chains[indices]
        self.last_negative_samples = negative_samples
        
        # Negative phase
        # 负向阶段
        negative_energy = self.energy_network(negative_samples)
        
        loss = positive_energy.mean() - negative_energy.mean()
        return loss