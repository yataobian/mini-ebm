import torch

class LangevinSampler:
    """
    Implements Langevin Dynamics sampling for EBMs.
    This sampler generates negative samples (fantasy particles) by following the gradient
    of the energy function, with added noise.
    实现了用于 EBM 的朗之万动力学采样。
    该采样器通过跟随能量函数的梯度并添加噪声来生成负样本（幻想粒子）。
    """
    def __init__(self, energy_network, step_size, noise_std):
        """
        Args:
            energy_network (torch.nn.Module): The EBM model that computes energy.
                                              计算能量的 EBM 模型。
            step_size (float): The step size for the Langevin update (alpha).
                               朗之万更新的步长 (alpha)。
            noise_std (float): The standard deviation of the Gaussian noise.
                               高斯噪声的标准差。
        """
        self.energy_network = energy_network
        self.step_size = step_size
        self.noise_std = noise_std

    def sample(self, x_init, n_steps):
        """
        Generates samples using k-step Langevin Dynamics.
        使用 k 步朗之万动力学生成样本。

        Args:
            x_init (torch.Tensor): Initial points to start the MCMC chain from.
                                   MCMC 链的起始点。
            n_steps (int): The number of MCMC steps (k in CD-k).
                           MCMC 的步数 (CD-k 中的 k)。

        Returns:
            torch.Tensor: The final samples after n_steps.
                          n_steps 后的最终样本。
        """
        # Clone the initial tensor to avoid modifying it in place.
        # 克隆初始张量以避免原地修改。
        x = x_init.clone().detach()

        for _ in range(n_steps):
            # We need gradients of the energy with respect to the samples.
            # 我们需要能量关于样本的梯度。
            x.requires_grad = True
            
            # Calculate the energy. The sum() is to get a scalar for backward().
            # 计算能量。sum() 是为了得到一个标量以便进行 backward()。
            energy = self.energy_network(x).sum()
            
            # Compute the gradient of the energy with respect to x.
            # 计算能量关于 x 的梯度。
            energy.backward()
            grad = x.grad

            # Langevin update rule:
            # x_t+1 = x_t - (step_size / 2) * grad + sqrt(step_size) * noise
            # For simplicity, we use a common variant:
            # x_t+1 = x_t - step_size * grad + noise_std * noise
            # 朗之万更新规则：
            # x_t+1 = x_t - (step_size / 2) * grad + sqrt(step_size) * noise
            # 为简单起见，我们使用一个常见的变体：
            # x_t+1 = x_t - step_size * grad + noise_std * noise
            with torch.no_grad():
                x = x - self.step_size * grad + self.noise_std * torch.randn_like(x)
            
            # Reset gradients for the next iteration.
            # 为下一次迭代重置梯度。
            if x.grad is not None:
                x.grad.zero_()
        
        return x.detach()
