"""
Example usage of various Contrastive Divergence variants for EBM training.
演示如何使用各种对比散度变体进行EBM训练的示例。

This file demonstrates how to use the different CD variants implemented in losses/cd_variants.py.
该文件演示了如何使用在 cd_variants.py 中实现的不同 CD 变体。
"""

import torch
import torch.distributions as dist
from models import EnergyNet
from samplers import LangevinSampler
from toy_data import get_toy_data
from losses import (
    PersistentContrastiveDivergenceLoss,
    FastPersistentContrastiveDivergenceLoss
    # TemperedContrastiveDivergenceLoss,
    # ParallelTemperingContrastiveDivergenceLoss,
    # AdaptiveContrastiveDivergenceLoss
)

def create_temperature_schedule(initial_temp=2.0, final_temp=1.0, total_steps=1000):
    """
    Create a temperature annealing schedule.
    创建温度退火调度。
    
    Args:
        initial_temp (float): Starting temperature / 起始温度
        final_temp (float): Final temperature / 最终温度
        total_steps (int): Total training steps / 总训练步数
    
    Returns:
        callable: Temperature schedule function / 温度调度函数
    """
    def schedule(step):
        # Linear annealing / 线性退火
        progress = min(step / total_steps, 1.0)
        return initial_temp + (final_temp - initial_temp) * progress
    
    return schedule

def example_persistent_cd():
    """
    Example using Persistent Contrastive Divergence.
    使用持久对比散度的示例。
    """
    print("=== Persistent Contrastive Divergence Example ===")
    print("=== 持久对比散度示例 ===")
    
    # Setup / 设置
    device = torch.device('cpu')
    energy_net = EnergyNet(input_dim=2, hidden_dim=64)
    sampler = LangevinSampler(energy_net, step_size=0.01, noise_std=0.01)
    
    # PCD Loss with 50 persistent particles
    # 带有50个持久粒子的PCD损失
    loss_fn = PersistentContrastiveDivergenceLoss(
        energy_network=energy_net,
        sampler=sampler,
        k=5,  # 5 MCMC steps per update / 每次更新5步MCMC
        n_persistent=50,  # 50 persistent particles / 50个持久粒子
        buffer_init_std=1.0
    )
    
    # Create toy data / 创建玩具数据
    data = get_toy_data('two_moons', n_samples=100)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    print("Training with PCD...")
    print("使用PCD训练...")
    for epoch in range(3):  # Just 3 epochs for demo / 演示只用3个epoch
        for batch_data in data_loader:
            optimizer.zero_grad()
            # Extract data from tuple (TensorDataset returns tuples)
            # 从元组中提取数据（TensorDataset 返回元组）
            batch_data = batch_data[0]
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            break  # Just one batch per epoch for demo / 演示每个epoch只用一个batch

def example_fast_persistent_cd():
    """
    Example using Fast Persistent Contrastive Divergence.
    使用快速持久对比散度的示例。
    """
    print("\n=== Fast Persistent Contrastive Divergence Example ===")
    print("=== 快速持久对比散度示例 ===")
    
    device = torch.device('cpu')
    energy_net = EnergyNet(input_dim=2, hidden_dim=64)
    sampler = LangevinSampler(energy_net, step_size=0.01, noise_std=0.01)
    
    # Fast PCD with multiple parallel chains and random restarts
    # 具有多个并行链和随机重启的快速PCD
    loss_fn = FastPersistentContrastiveDivergenceLoss(
        energy_network=energy_net,
        sampler=sampler,
        k=3,  # Fewer steps due to parallel chains / 由于并行链，步数较少
        n_chains=30,  # 30 parallel chains / 30个并行链
        restart_prob=0.05,  # 5% chance of restart per chain / 每个链5%的重启概率
        buffer_init_std=1.0
    )
    
    data = get_toy_data('checkerboard', n_samples=100)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    print("Training with Fast PCD...")
    print("使用快速PCD训练...")
    for epoch in range(3):
        for batch_data in data_loader:
            optimizer.zero_grad()
            batch_data = batch_data[0]
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            break

def example_tempered_cd():
    """
    Example using Tempered Contrastive Divergence with temperature scheduling.
    使用带温度调度的有温度对比散度的示例。
    """
    print("\n=== Tempered Contrastive Divergence Example ===")
    print("=== 有温度对比散度示例 ===")
    
    device = torch.device('cpu')
    energy_net = EnergyNet(input_dim=2, hidden_dim=64)
    sampler = LangevinSampler(energy_net, step_size=0.01, noise_std=0.01)
    
    # Temperature schedule: start hot, cool down over time
    # 温度调度：开始时温度高，随时间冷却
    temp_schedule = create_temperature_schedule(
        initial_temp=3.0,  # Start hot for better mixing / 开始时温度高以获得更好的混合
        final_temp=1.0,    # Cool to normal temperature / 冷却到正常温度
        total_steps=100
    )
    
    loss_fn = TemperedContrastiveDivergenceLoss(
        energy_network=energy_net,
        sampler=sampler,
        k=8,  # More steps needed with temperature / 使用温度时需要更多步数
        temperature=3.0,  # Will be overridden by schedule / 将被调度覆盖
        temp_schedule=temp_schedule
    )
    
    data = get_toy_data('gmm', n_samples=100)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    print("Training with Tempered CD...")
    print("使用有温度CD训练...")
    for epoch in range(3):
        for batch_data in data_loader:
            optimizer.zero_grad()
            batch_data = batch_data[0]
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            
            current_temp = loss_fn.temperature
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Temperature: {current_temp:.2f}")
            break

def example_parallel_tempering_cd():
    """
    Example using Parallel Tempering Contrastive Divergence.
    使用并行回火对比散度的示例。
    """
    print("\n=== Parallel Tempering Contrastive Divergence Example ===")
    print("=== 并行回火对比散度示例 ===")
    
    device = torch.device('cpu')
    energy_net = EnergyNet(input_dim=2, hidden_dim=64)
    sampler = LangevinSampler(energy_net, step_size=0.01, noise_std=0.01)
    
    # Multiple temperature levels for better exploration
    # 多个温度级别以获得更好的探索
    loss_fn = ParallelTemperingContrastiveDivergenceLoss(
        energy_network=energy_net,
        sampler=sampler,
        k=4,
        temperatures=[1.0, 1.5, 2.0, 3.0],  # Four temperature levels / 四个温度级别
        swap_prob=0.1,  # 10% chance of temperature swaps / 10%的温度交换概率
        n_particles_per_temp=15  # 15 particles per temperature / 每个温度15个粒子
    )
    
    data = get_toy_data('two_moons', n_samples=100)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    print("Training with Parallel Tempering CD...")
    print("使用并行回火CD训练...")
    for epoch in range(3):
        for batch_data in data_loader:
            optimizer.zero_grad()
            batch_data = batch_data[0]
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            break

def example_adaptive_cd():
    """
    Example using Adaptive Contrastive Divergence.
    使用自适应对比散度的示例。
    """
    print("\n=== Adaptive Contrastive Divergence Example ===")
    print("=== 自适应对比散度示例 ===")
    
    device = torch.device('cpu')
    energy_net = EnergyNet(input_dim=2, hidden_dim=64)
    sampler = LangevinSampler(energy_net, step_size=0.01, noise_std=0.01)
    
    # Adaptive CD that adjusts steps based on convergence
    # 根据收敛性调整步数的自适应CD
    loss_fn = AdaptiveContrastiveDivergenceLoss(
        energy_network=energy_net,
        sampler=sampler,
        k_min=2,    # Minimum 2 steps / 最少2步
        k_max=15,   # Maximum 15 steps / 最多15步
        convergence_threshold=0.01,  # Energy variance threshold / 能量方差阈值
        adaptation_rate=0.1  # 10% adaptation rate / 10%的适应率
    )
    
    data = get_toy_data('checkerboard', n_samples=100)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    print("Training with Adaptive CD...")
    print("使用自适应CD训练...")
    for epoch in range(3):
        for batch_data in data_loader:
            optimizer.zero_grad()
            batch_data = batch_data[0]
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            
            current_k = loss_fn.current_k
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Adaptive k: {current_k}")
            break

def main():
    """
    Run all CD variant examples.
    运行所有CD变体示例。
    """
    print("Contrastive Divergence Variants Examples")
    print("对比散度变体示例")
    print("=" * 50)
    
    # Run all examples / 运行所有示例
    example_persistent_cd()
    example_fast_persistent_cd()
    # example_tempered_cd()
    # example_parallel_tempering_cd()
    # example_adaptive_cd()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("所有示例成功完成！")

if __name__ == "__main__":
    main() 