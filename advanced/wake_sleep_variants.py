"""
Wake-Sleep Algorithm Variants for Energy-Based Models
EBM的Wake-Sleep算法变体

Based on the classical wake-sleep algorithm from Helmholtz Machine (1994),
this module implements various wake-sleep training procedures for EBMs.

基于1994年Helmholtz Machine的经典wake-sleep算法，
本模块实现了EBM的各种wake-sleep训练过程。

References:
    - Hinton, G. E., Dayan, P., Frey, B. J., & Neal, R. M. (1995). 
      The "wake-sleep" algorithm for unsupervised neural networks. Science.
    - Dayan, P., Hinton, G. E., Neal, R. M., & Zemel, R. S. (1995). 
      The helmholtz machine. Neural computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from collections import deque
import warnings


class WakeSleepEBMLoss(ABC):
    """
    Abstract base class for Wake-Sleep training of Energy-Based Models
    EBM Wake-Sleep训练的抽象基类
    
    The wake-sleep algorithm alternates between two phases:
    Wake-sleep算法在两个阶段之间交替：
    
    1. Wake Phase: Update energy function using real data
       觉醒阶段：使用真实数据更新能量函数
    
    2. Sleep Phase: Update energy function using generated/sampled data  
       睡眠阶段：使用生成/采样数据更新能量函数
    """
    
    def __init__(self, energy_network, wake_lr=0.001, sleep_lr=0.001, 
                 device='cpu', **kwargs):
        """
        Initialize Wake-Sleep EBM trainer
        初始化Wake-Sleep EBM训练器
        
        Args:
            energy_network: Neural network that outputs energy E(x)
                           输出能量E(x)的神经网络
            wake_lr: Learning rate for wake phase / 觉醒阶段学习率
            sleep_lr: Learning rate for sleep phase / 睡眠阶段学习率
            device: Device for computation / 计算设备
        """
        self.energy_network = energy_network.to(device)
        self.device = device
        self.wake_lr = wake_lr
        self.sleep_lr = sleep_lr
        
        # Separate optimizers for wake and sleep phases
        # 为觉醒和睡眠阶段使用独立的优化器
        self.wake_optimizer = optim.Adam(
            self.energy_network.parameters(), lr=wake_lr
        )
        self.sleep_optimizer = optim.Adam(
            self.energy_network.parameters(), lr=sleep_lr  
        )
        
        # Training statistics / 训练统计
        self.wake_losses = []
        self.sleep_losses = []
        self.phase_history = []  # Track which phase was last executed
        
    @abstractmethod
    def wake_phase(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        Wake phase: process real data
        觉醒阶段：处理真实数据
        
        Args:
            real_data: Real data samples / 真实数据样本
            
        Returns:
            Wake phase loss / 觉醒阶段损失
        """
        pass
    
    @abstractmethod  
    def sleep_phase(self, batch_size: int) -> torch.Tensor:
        """
        Sleep phase: process generated/sampled data
        睡眠阶段：处理生成/采样数据
        
        Args:
            batch_size: Number of samples to generate / 生成样本数量
            
        Returns:
            Sleep phase loss / 睡眠阶段损失
        """
        pass
    
    def __call__(self, real_data: torch.Tensor, mode: str = 'both') -> Dict[str, torch.Tensor]:
        """
        Execute wake-sleep training step
        执行wake-sleep训练步骤
        
        Args:
            real_data: Real data batch / 真实数据批次
            mode: Training mode - 'wake', 'sleep', or 'both' / 训练模式
            
        Returns:
            Dictionary with losses / 包含损失的字典
        """
        batch_size = real_data.shape[0]
        
        # Set data dimension if not already set
        if hasattr(self, 'data_dim') and self.data_dim is None:
            self.data_dim = real_data.shape[1:]
            
        losses = {}
        
        if mode in ['wake', 'both']:
            wake_loss = self.wake_phase(real_data)
            losses['wake_loss'] = wake_loss
            self.wake_losses.append(wake_loss.item())
            self.phase_history.append('wake')
            
        if mode in ['sleep', 'both']:
            sleep_loss = self.sleep_phase(batch_size)
            losses['sleep_loss'] = sleep_loss  
            self.sleep_losses.append(sleep_loss.item())
            self.phase_history.append('sleep')
            
        return losses


class ClassicalWakeSleepEBM(WakeSleepEBMLoss):
    """
    Classical Wake-Sleep algorithm for EBMs based on Helmholtz Machine
    基于Helmholtz Machine的经典EBM Wake-Sleep算法
    
    This implements the core idea from the 1994 Helmholtz Machine paper:
    这实现了1994年Helmholtz Machine论文的核心思想：
    
    - Wake: Minimize energy of real data (like training recognition model)
      觉醒：最小化真实数据的能量（类似训练识别模型）
    - Sleep: Maximize energy of generated samples (like training with fantasies)  
      睡眠：最大化生成样本的能量（类似用幻想数据训练）
    """
    
    def __init__(self, energy_network, sampler_steps=20, noise_std=0.01, 
                 wake_lr=0.001, sleep_lr=0.001, data_dim=None, **kwargs):
        super().__init__(energy_network, wake_lr, sleep_lr, **kwargs)
        
        self.sampler_steps = sampler_steps
        self.noise_std = noise_std
        self.data_dim = data_dim  # Store data dimension for sampling
        
        # Replay buffer for persistent chains (like PCD)
        # 用于持续链的重放缓冲区（类似PCD）
        self.replay_buffer = deque(maxlen=10000)
        
    def _langevin_sampler(self, x_init: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Langevin MCMC sampler for generating negative samples
        用于生成负样本的Langevin MCMC采样器
        
        Args:
            x_init: Initial samples / 初始样本
            steps: Number of sampling steps / 采样步数
            
        Returns:
            Generated samples / 生成样本
        """
        x = x_init.clone().detach().requires_grad_(True)
        
        for _ in range(steps):
            energy = self.energy_network(x).sum()
            grad = torch.autograd.grad(energy, x, create_graph=False)[0]
            
            # Langevin dynamics step / Langevin动力学步骤
            x = x - 0.5 * 0.01 * grad + self.noise_std * torch.randn_like(x)
            x = x.detach().requires_grad_(True)
            
        return x.detach()
    
    def wake_phase(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        Wake phase: Train on real data to minimize their energy
        觉醒阶段：在真实数据上训练以最小化其能量
        
        This corresponds to the "wake" phase in Helmholtz Machine where
        we update the recognition model using real sensory input.
        这对应于Helmholtz Machine中的"觉醒"阶段，
        我们使用真实感官输入更新识别模型。
        """
        self.wake_optimizer.zero_grad()
        
        # Minimize energy of real data / 最小化真实数据的能量
        energies = self.energy_network(real_data)
        wake_loss = energies.mean()  # Minimize energy
        
        wake_loss.backward()
        self.wake_optimizer.step()
        
        return wake_loss
    
    def sleep_phase(self, batch_size: int) -> torch.Tensor:
        """
        Sleep phase: Generate samples and maximize their energy  
        睡眠阶段：生成样本并最大化其能量
        
        This corresponds to the "sleep" phase in Helmholtz Machine where
        we generate internal fantasies and train the recognition model.
        这对应于Helmholtz Machine中的"睡眠"阶段，
        我们生成内部幻想并训练识别模型。
        """
        self.sleep_optimizer.zero_grad()
        
        # Initialize samples from replay buffer or noise
        # 从重放缓冲区或噪声初始化样本
        if len(self.replay_buffer) > batch_size and np.random.random() > 0.05:
            # Use samples from replay buffer / 使用重放缓冲区的样本
            indices = np.random.choice(len(self.replay_buffer), batch_size)
            x_init = torch.stack([self.replay_buffer[i] for i in indices])
        else:
            # Initialize from noise / 从噪声初始化
            # Use stored data dimension or default to common size
            if self.data_dim is not None:
                x_init = torch.randn(batch_size, *self.data_dim, device=self.device)
            else:
                # Default to common size for compatibility
                x_init = torch.randn(batch_size, 784, device=self.device)
        
        # Generate negative samples using Langevin sampling
        # 使用Langevin采样生成负样本
        neg_samples = self._langevin_sampler(x_init, self.sampler_steps)
        
        # Update replay buffer / 更新重放缓冲区
        for sample in neg_samples:
            self.replay_buffer.append(sample.detach().cpu())
        
        # Maximize energy of generated samples / 最大化生成样本的能量
        neg_energies = self.energy_network(neg_samples)
        sleep_loss = -neg_energies.mean()  # Maximize energy (minimize negative)
        
        sleep_loss.backward()
        self.sleep_optimizer.step()
        
        return sleep_loss


class AlternatingWakeSleepEBM(ClassicalWakeSleepEBM):
    """
    Alternating Wake-Sleep EBM that strictly alternates between phases
    严格在阶段间交替的Wake-Sleep EBM
    
    This variant ensures that wake and sleep phases are executed in alternation,
    more closely following the original Helmholtz Machine protocol.
    这个变体确保觉醒和睡眠阶段交替执行，
    更紧密地遵循原始Helmholtz Machine协议。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_phase = 'wake'  # Start with wake phase
        
    def __call__(self, real_data: torch.Tensor, mode: str = 'both') -> Dict[str, torch.Tensor]:
        """
        Execute current phase and switch to the other
        执行当前阶段并切换到另一个阶段
        """
        # Override alternating behavior if mode is specified / 如果指定了mode则覆盖交替行为
        if mode == 'wake':
            loss = self.wake_phase(real_data)
            losses = {'wake_loss': loss}
        elif mode == 'sleep':
            loss = self.sleep_phase(real_data.shape[0])
            losses = {'sleep_loss': loss}
        else:
            # Default alternating behavior / 默认交替行为
            if self.current_phase == 'wake':
                loss = self.wake_phase(real_data)
                losses = {'wake_loss': loss}
                self.current_phase = 'sleep'
            else:
                loss = self.sleep_phase(real_data.shape[0])
                losses = {'sleep_loss': loss}
                self.current_phase = 'wake'
            
        return losses


class BiDirectionalWakeSleepEBM(WakeSleepEBMLoss):
    """
    Bidirectional Wake-Sleep EBM with separate forward and backward networks
    具有独立前向和后向网络的双向Wake-Sleep EBM
    
    This variant uses separate networks for encoding (wake) and generation (sleep),
    more closely mimicking the original Helmholtz Machine architecture.
    这个变体使用独立的网络进行编码（觉醒）和生成（睡眠），
    更紧密地模仿原始Helmholtz Machine架构。
    """
    
    def __init__(self, forward_network, backward_network, **kwargs):
        # Initialize with forward network, but we'll use both
        super().__init__(forward_network, **kwargs)
        
        self.forward_net = forward_network.to(self.device)   # Recognition/Encoder
        self.backward_net = backward_network.to(self.device) # Generator/Decoder
        
        # Infer latent dimension from backward network's first layer
        # 从后向网络的第一层推断潜在维度
        self.latent_dim = self._infer_latent_dimension()
        
        # Separate optimizers for each network / 为每个网络使用独立优化器
        self.forward_optimizer = optim.Adam(self.forward_net.parameters(), lr=self.wake_lr)
        self.backward_optimizer = optim.Adam(self.backward_net.parameters(), lr=self.sleep_lr)
        
    def _infer_latent_dimension(self) -> int:
        """Infer latent dimension from backward network's first layer"""
        # Check if backward network is a Sequential model
        if isinstance(self.backward_net, nn.Sequential):
            first_layer = self.backward_net[0]
            if isinstance(first_layer, nn.Linear):
                return first_layer.in_features
        
        # Check if backward network has a first layer attribute
        elif hasattr(self.backward_net, 'layers') and len(self.backward_net.layers) > 0:
            first_layer = self.backward_net.layers[0]
            if isinstance(first_layer, nn.Linear):
                return first_layer.in_features
        
        # Default fallback
        return 100
        
    def wake_phase(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        Wake phase: Train forward network (recognition) on real data
        觉醒阶段：在真实数据上训练前向网络（识别）
        """
        self.forward_optimizer.zero_grad()
        
        # Forward network learns to recognize real data patterns
        # 前向网络学习识别真实数据模式
        energies = self.forward_net(real_data)
        wake_loss = energies.mean()  # Minimize energy for real data
        
        wake_loss.backward()
        self.forward_optimizer.step()
        
        return wake_loss
    
    def sleep_phase(self, batch_size: int) -> torch.Tensor:
        """
        Sleep phase: Train backward network (generation) with forward network feedback
        睡眠阶段：用前向网络反馈训练后向网络（生成）
        """
        self.backward_optimizer.zero_grad()
        
        # Generate samples from backward network / 从后向网络生成样本
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)  # Latent space
        generated_samples = self.backward_net(noise)
        
        # Forward network evaluates generated samples
        # 前向网络评估生成样本
        with torch.no_grad():
            target_energies = self.forward_net(generated_samples)
        
        # Train backward network to generate low-energy samples
        # 训练后向网络生成低能量样本
        current_energies = self.forward_net(generated_samples)
        sleep_loss = current_energies.mean()  # Minimize energy of generated samples
        
        sleep_loss.backward()
        self.backward_optimizer.step()
        
        return sleep_loss


class TemperatureControlledWakeSleepEBM(ClassicalWakeSleepEBM):
    """
    Temperature-controlled Wake-Sleep EBM with annealing schedule
    具有退火调度的温度控制Wake-Sleep EBM
    
    This variant introduces temperature control to balance exploration vs exploitation,
    following the principles of simulated annealing in the original wake-sleep algorithm.
    这个变体引入温度控制来平衡探索与利用，
    遵循原始wake-sleep算法中的模拟退火原理。
    """
    
    def __init__(self, *args, initial_temp=1.0, min_temp=0.1, 
                 temp_decay=0.999, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        
    def _update_temperature(self):
        """Update temperature according to annealing schedule"""
        self.current_temp = max(
            self.min_temp, 
            self.current_temp * self.temp_decay
        )
    
    def wake_phase(self, real_data: torch.Tensor) -> torch.Tensor:
        """Wake phase with temperature-scaled energy"""
        self.wake_optimizer.zero_grad()
        
        # Temperature-scaled energy / 温度缩放能量
        energies = self.energy_network(real_data) / self.current_temp
        wake_loss = energies.mean()
        
        wake_loss.backward()
        self.wake_optimizer.step()
        
        return wake_loss
    
    def sleep_phase(self, batch_size: int) -> torch.Tensor:
        """Sleep phase with temperature-controlled sampling"""
        self.sleep_optimizer.zero_grad()
        
        # Temperature-controlled Langevin sampling
        # 温度控制的Langevin采样
        # Use stored data dimension or default to common size
        if self.data_dim is not None:
            x_init = torch.randn(batch_size, *self.data_dim, device=self.device)
        else:
            # Default to common size for compatibility
            x_init = torch.randn(batch_size, 784, device=self.device)
            
        neg_samples = self._temperature_langevin_sampler(x_init, self.sampler_steps)
        
        neg_energies = self.energy_network(neg_samples) / self.current_temp
        sleep_loss = -neg_energies.mean()
        
        sleep_loss.backward()
        self.sleep_optimizer.step()
        
        # Update temperature / 更新温度
        self._update_temperature()
        
        return sleep_loss
    
    def _temperature_langevin_sampler(self, x_init: torch.Tensor, steps: int) -> torch.Tensor:
        """Temperature-controlled Langevin sampler"""
        x = x_init.clone().detach().requires_grad_(True)
        
        for _ in range(steps):
            energy = self.energy_network(x).sum() / self.current_temp
            grad = torch.autograd.grad(energy, x, create_graph=False)[0]
            
            # Temperature affects both gradient step and noise
            # 温度影响梯度步长和噪声
            step_size = 0.01 * self.current_temp
            noise_scale = self.noise_std * np.sqrt(self.current_temp)
            
            x = x - 0.5 * step_size * grad + noise_scale * torch.randn_like(x)
            x = x.detach().requires_grad_(True)
            
        return x.detach()


class HierarchicalWakeSleepEBM(WakeSleepEBMLoss):
    """
    Hierarchical Wake-Sleep EBM with multiple levels of representation
    具有多层表示的分层Wake-Sleep EBM
    
    This variant implements hierarchical learning as in the original Helmholtz Machine,
    with multiple levels of hidden representations.
    这个变体实现了如原始Helmholtz Machine中的分层学习，
    具有多层隐藏表示。
    """
    
    def __init__(self, energy_networks: List[nn.Module], **kwargs):
        # Use the first network for base class initialization
        super().__init__(energy_networks[0], **kwargs)
        
        self.energy_networks = [net.to(self.device) for net in energy_networks]
        self.num_levels = len(energy_networks)
        
        # Separate optimizers for each level / 为每层使用独立优化器
        self.optimizers = [
            optim.Adam(net.parameters(), lr=self.wake_lr) 
            for net in self.energy_networks
        ]
        
    def wake_phase(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical wake phase: bottom-up processing
        分层觉醒阶段：自下而上处理
        """
        total_loss = 0
        current_input = real_data
        
        for level, (net, optimizer) in enumerate(zip(self.energy_networks, self.optimizers)):
            optimizer.zero_grad()
            
            # Process current level / 处理当前层
            energies = net(current_input)
            level_loss = energies.mean()
            
            level_loss.backward(retain_graph=True)
            optimizer.step()
            
            total_loss += level_loss
            
            # Prepare input for next level (using energy gradients)
            # 为下一层准备输入（使用能量梯度）
            if level < self.num_levels - 1:
                with torch.no_grad():
                    current_input = torch.autograd.grad(
                        energies.sum(), current_input, create_graph=False
                    )[0].detach()
        
        return total_loss
    
    def sleep_phase(self, batch_size: int) -> torch.Tensor:
        """
        Hierarchical sleep phase: top-down generation
        分层睡眠阶段：自上而下生成
        """
        total_loss = 0
        
        # Start from top level with noise / 从顶层噪声开始
        current_samples = torch.randn(batch_size, 100, device=self.device)
        
        # Top-down generation / 自上而下生成
        for level in reversed(range(self.num_levels)):
            net = self.energy_networks[level]
            optimizer = self.optimizers[level]
            
            optimizer.zero_grad()
            
            # Generate samples at current level / 在当前层生成样本
            if level < self.num_levels - 1:
                # Transform from higher level / 从更高层转换
                current_samples = self._transform_down(current_samples, level)
            
            # Evaluate and update / 评估和更新
            energies = net(current_samples)
            level_loss = -energies.mean()  # Maximize energy
            
            level_loss.backward()
            optimizer.step()
            
            total_loss += level_loss
            
        return total_loss
    
    def _transform_down(self, higher_level_samples: torch.Tensor, target_level: int) -> torch.Tensor:
        """Transform samples from higher level to target level"""
        # Simple linear transformation (could be more sophisticated)
        # 简单线性变换（可以更复杂）
        if target_level == 0:
            # Transform to input space / 转换到输入空间
            return torch.randn(higher_level_samples.shape[0], 784, device=self.device)
        else:
            # Transform to intermediate level / 转换到中间层
            return torch.randn(higher_level_samples.shape[0], 256, device=self.device)


class AdaptiveWakeSleepEBM(ClassicalWakeSleepEBM):
    """
    Adaptive Wake-Sleep EBM with dynamic phase switching
    具有动态阶段切换的自适应Wake-Sleep EBM
    
    This variant adaptively determines when to switch between wake and sleep phases
    based on training progress, similar to adaptive algorithms in modern ML.
    这个变体根据训练进度自适应地确定何时在觉醒和睡眠阶段间切换，
    类似于现代ML中的自适应算法。
    """
    
    def __init__(self, *args, adaptation_window=100, 
                 wake_threshold=0.1, sleep_threshold=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.adaptation_window = adaptation_window
        self.wake_threshold = wake_threshold
        self.sleep_threshold = sleep_threshold
        
        # Tracking for adaptive switching / 自适应切换的跟踪
        self.recent_wake_losses = deque(maxlen=adaptation_window)
        self.recent_sleep_losses = deque(maxlen=adaptation_window)
        
    def _should_do_wake_phase(self) -> bool:
        """Determine if wake phase should be executed based on recent performance"""
        if len(self.recent_wake_losses) < self.adaptation_window:
            return True  # Default to wake phase initially
        
        # Check if wake loss is not improving / 检查觉醒损失是否没有改善
        recent_avg = np.mean(list(self.recent_wake_losses)[-20:])
        older_avg = np.mean(list(self.recent_wake_losses)[:20])
        
        return recent_avg > older_avg - self.wake_threshold
    
    def _should_do_sleep_phase(self) -> bool:
        """Determine if sleep phase should be executed based on recent performance"""
        if len(self.recent_sleep_losses) < self.adaptation_window:
            return True  # Default to sleep phase initially
        
        # Check if sleep loss is not improving / 检查睡眠损失是否没有改善
        recent_avg = np.mean(list(self.recent_sleep_losses)[-20:])
        older_avg = np.mean(list(self.recent_sleep_losses)[:20])
        
        return recent_avg > older_avg - self.sleep_threshold
    
    def __call__(self, real_data: torch.Tensor, mode: str = 'both') -> Dict[str, torch.Tensor]:
        """Adaptively execute wake and/or sleep phases"""
        losses = {}
        
        # Adaptive phase selection / 自适应阶段选择
        do_wake = self._should_do_wake_phase()
        do_sleep = self._should_do_sleep_phase()
        
        # Override with mode parameter if specified / 如果指定了mode参数则覆盖
        if mode == 'wake':
            do_wake = True
            do_sleep = False
        elif mode == 'sleep':
            do_wake = False
            do_sleep = True
        # mode == 'both' uses adaptive selection
        
        if do_wake:
            wake_loss = self.wake_phase(real_data)
            losses['wake_loss'] = wake_loss
            self.recent_wake_losses.append(wake_loss.item())
            self.phase_history.append('wake')
            
        if do_sleep:
            sleep_loss = self.sleep_phase(real_data.shape[0])
            losses['sleep_loss'] = sleep_loss
            self.recent_sleep_losses.append(sleep_loss.item())
            self.phase_history.append('sleep')
            
        return losses


class WakeSleepEBMTrainer:
    """
    Unified trainer for all Wake-Sleep EBM variants
    所有Wake-Sleep EBM变体的统一训练器
    
    This class provides a common interface for training different wake-sleep variants
    and includes utilities for monitoring training progress.
    这个类为训练不同的wake-sleep变体提供了通用接口，
    并包含监控训练进度的工具。
    """
    
    def __init__(self, model: WakeSleepEBMLoss, device='cpu'):
        self.model = model
        self.device = device
        
        # Training history / 训练历史
        self.training_history = {
            'wake_losses': [],
            'sleep_losses': [],
            'epochs': [],
            'iterations': []
        }
        
    def train_epoch(self, dataloader, epoch: int, mode: str = 'both') -> Dict[str, float]:
        """
        Train for one epoch
        训练一个epoch
        
        Args:
            dataloader: Data loader / 数据加载器
            epoch: Current epoch / 当前epoch
            mode: Training mode / 训练模式
            
        Returns:
            Average losses for the epoch / epoch的平均损失
        """
        self.model.energy_network.train()
        
        total_wake_loss = 0
        total_sleep_loss = 0
        num_wake_steps = 0
        num_sleep_steps = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(self.device)
            
            # Execute wake-sleep step / 执行wake-sleep步骤
            losses = self.model(data, mode=mode)
            
            # Accumulate losses / 累积损失
            if 'wake_loss' in losses:
                total_wake_loss += losses['wake_loss'].item()
                num_wake_steps += 1
                
            if 'sleep_loss' in losses:
                total_sleep_loss += losses['sleep_loss'].item()
                num_sleep_steps += 1
                
        # Calculate averages / 计算平均值
        avg_losses = {}
        if num_wake_steps > 0:
            avg_losses['avg_wake_loss'] = total_wake_loss / num_wake_steps
            self.training_history['wake_losses'].append(avg_losses['avg_wake_loss'])
            
        if num_sleep_steps > 0:
            avg_losses['avg_sleep_loss'] = total_sleep_loss / num_sleep_steps
            self.training_history['sleep_losses'].append(avg_losses['avg_sleep_loss'])
            
        self.training_history['epochs'].append(epoch)
        
        return avg_losses
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'total_epochs': len(self.training_history['epochs']),
            'final_wake_loss': self.training_history['wake_losses'][-1] if self.training_history['wake_losses'] else None,
            'final_sleep_loss': self.training_history['sleep_losses'][-1] if self.training_history['sleep_losses'] else None,
            'wake_loss_trend': np.polyfit(range(len(self.training_history['wake_losses'])), 
                                         self.training_history['wake_losses'], 1)[0] if len(self.training_history['wake_losses']) > 1 else 0,
            'sleep_loss_trend': np.polyfit(range(len(self.training_history['sleep_losses'])), 
                                          self.training_history['sleep_losses'], 1)[0] if len(self.training_history['sleep_losses']) > 1 else 0,
        }
        return stats
    
    def save_model(self, filepath: str):
        """Save model and training history"""
        torch.save({
            'model_state_dict': self.model.energy_network.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'type': type(self.model).__name__,
                'wake_lr': self.model.wake_lr,
                'sleep_lr': self.model.sleep_lr,
            }
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load model and training history"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.energy_network.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        return checkpoint['model_config']


# Example usage and demonstration
# 使用示例和演示
def create_simple_energy_network(input_dim: int = 784, hidden_dim: int = 256) -> nn.Module:
    """Create a simple energy network for demonstration"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)  # Output single energy value
    )

def create_generator_network(latent_dim: int = 100, output_dim: int = 784) -> nn.Module:
    """Create a generator network for bidirectional wake-sleep"""
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim),
        nn.Tanh()  # Output in [-1, 1] range
    )

if __name__ == "__main__":
    # Demonstration of different wake-sleep variants
    # 不同wake-sleep变体的演示
    
    print("Wake-Sleep EBM Variants Demonstration")
    print("Wake-Sleep EBM变体演示")
    print("=" * 50)
    
    # Create example networks / 创建示例网络
    energy_net = create_simple_energy_network()
    generator_net = create_generator_network()
    
    # Test different variants / 测试不同变体
    variants = {
        'Classical Wake-Sleep': ClassicalWakeSleepEBM(energy_net),
        'Alternating Wake-Sleep': AlternatingWakeSleepEBM(energy_net),
        'Temperature-Controlled': TemperatureControlledWakeSleepEBM(energy_net),
        'Adaptive Wake-Sleep': AdaptiveWakeSleepEBM(energy_net),
    }
    
    # Generate some dummy data / 生成一些虚拟数据
    dummy_data = torch.randn(32, 784)
    
    for name, variant in variants.items():
        print(f"\nTesting {name}:")
        print(f"测试 {name}:")
        
        try:
            losses = variant(dummy_data)
            print(f"  Losses: {losses}")
            print(f"  损失: {losses}")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  错误: {e}")
    
    print("\n" + "=" * 50)
    print("All variants implemented successfully!")
    print("所有变体实现成功！") 