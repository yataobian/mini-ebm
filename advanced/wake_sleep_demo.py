"""
Wake-Sleep EBM Variants Demonstration Script
Wake-Sleep EBM变体演示脚本

This script demonstrates the usage of different wake-sleep algorithm variants
for training Energy-Based Models, based on the classical Helmholtz Machine.

这个脚本演示了基于经典Helmholtz Machine的不同wake-sleep算法变体
用于训练能量基模型的使用方法。
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import wake-sleep variants from the same directory
try:
    from wake_sleep_variants import (
        WakeSleepEBMLoss, ClassicalWakeSleepEBM, AlternatingWakeSleepEBM,
        BiDirectionalWakeSleepEBM, TemperatureControlledWakeSleepEBM,
        HierarchicalWakeSleepEBM, AdaptiveWakeSleepEBM, WakeSleepEBMTrainer
    )
except ImportError:
    # If running as a script, try importing from current directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from wake_sleep_variants import (
        WakeSleepEBMLoss, ClassicalWakeSleepEBM, AlternatingWakeSleepEBM,
        BiDirectionalWakeSleepEBM, TemperatureControlledWakeSleepEBM,
        HierarchicalWakeSleepEBM, AdaptiveWakeSleepEBM, WakeSleepEBMTrainer
    )

def create_synthetic_dataset(n_samples=1000, dim=784, n_modes=3):
    """
    Create synthetic multi-modal dataset for demonstration
    创建用于演示的合成多模态数据集
    """
    data = []
    
    for i in range(n_modes):
        # Create mode-specific data / 创建模态特定数据
        center = torch.randn(dim) * 2
        mode_data = torch.randn(n_samples // n_modes, dim) * 0.5 + center
        data.append(mode_data)
    
    # Combine all modes / 合并所有模态
    full_data = torch.cat(data, dim=0)
    
    # Add some noise and normalize / 添加噪声并归一化
    full_data += torch.randn_like(full_data) * 0.1
    full_data = torch.tanh(full_data)  # Normalize to [-1, 1]
    
    return full_data

def create_2d_visualization_dataset(n_samples=500):
    """
    Create 2D dataset for visualization
    创建用于可视化的2D数据集
    """
    # Create spiral dataset / 创建螺旋数据集
    t = torch.linspace(0, 4*np.pi, n_samples)
    x = t * torch.cos(t) * 0.1
    y = t * torch.sin(t) * 0.1
    
    # Add noise / 添加噪声
    x += torch.randn_like(x) * 0.02
    y += torch.randn_like(y) * 0.02
    
    data = torch.stack([x, y], dim=1)
    return data

def create_energy_network_2d():
    """Create energy network for 2D data"""
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1)
    )

def plot_energy_landscape_2d(model, data, title="Energy Landscape", save_path=None):
    """
    Plot 2D energy landscape for visualization
    绘制2D能量景观进行可视化
    """
    # Create grid for plotting / 创建绘图网格
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )
    
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    )
    
    # Compute energy for grid points / 计算网格点的能量
    with torch.no_grad():
        energies = model.energy_network(grid_points).numpy()
    
    energies = energies.reshape(xx.shape)
    
    # Plot / 绘图
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, energies, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Energy')
    
    # Plot data points / 绘制数据点
    plt.scatter(data[:, 0], data[:, 1], c='red', s=20, alpha=0.6, label='Data')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved energy landscape to {save_path}")
    
    plt.show()

def demonstrate_classical_wake_sleep():
    """
    Demonstrate Classical Wake-Sleep EBM
    演示经典Wake-Sleep EBM
    """
    print("\n" + "="*60)
    print("Classical Wake-Sleep EBM Demonstration")
    print("经典Wake-Sleep EBM演示")
    print("="*60)
    
    # Create 2D dataset for visualization / 创建2D数据集用于可视化
    data = create_2d_visualization_dataset(1000)
    dataset = TensorDataset(data, torch.zeros(len(data)))  # Dummy labels
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create model / 创建模型
    energy_net = create_energy_network_2d()
    model = ClassicalWakeSleepEBM(
        energy_net, 
        sampler_steps=10, 
        wake_lr=0.001, 
        sleep_lr=0.001
    )
    
    trainer = WakeSleepEBMTrainer(model)
    
    print("Training Classical Wake-Sleep EBM...")
    print("训练经典Wake-Sleep EBM...")
    
    # Train for a few epochs / 训练几个epoch
    for epoch in range(5):
        losses = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1}: {losses}")
        print(f"第{epoch+1}轮: {losses}")
    
    # Plot energy landscape / 绘制能量景观
    plot_energy_landscape_2d(
        model, data, 
        "Classical Wake-Sleep EBM Energy Landscape",
        "classical_wake_sleep_energy.png"
    )
    
    return model, trainer

def demonstrate_temperature_controlled():
    """
    Demonstrate Temperature-Controlled Wake-Sleep EBM
    演示温度控制Wake-Sleep EBM
    """
    print("\n" + "="*60)
    print("Temperature-Controlled Wake-Sleep EBM Demonstration")
    print("温度控制Wake-Sleep EBM演示")
    print("="*60)
    
    # Create dataset / 创建数据集
    data = create_2d_visualization_dataset(1000)
    dataset = TensorDataset(data, torch.zeros(len(data)))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create model with temperature control / 创建带温度控制的模型
    energy_net = create_energy_network_2d()
    model = TemperatureControlledWakeSleepEBM(
        energy_net,
        initial_temp=2.0,
        min_temp=0.1,
        temp_decay=0.95,
        sampler_steps=10
    )
    
    trainer = WakeSleepEBMTrainer(model)
    
    print("Training Temperature-Controlled Wake-Sleep EBM...")
    print("训练温度控制Wake-Sleep EBM...")
    
    # Track temperature over training / 跟踪训练过程中的温度
    temperatures = []
    
    for epoch in range(8):
        losses = trainer.train_epoch(dataloader, epoch)
        temperatures.append(model.current_temp)
        print(f"Epoch {epoch+1}: {losses}, Temperature: {model.current_temp:.3f}")
        print(f"第{epoch+1}轮: {losses}, 温度: {model.current_temp:.3f}")
    
    # Plot temperature decay / 绘制温度衰减
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(temperatures)
    plt.title('Temperature Decay / 温度衰减')
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.grid(True)
    
    # Plot energy landscape / 绘制能量景观
    plt.subplot(1, 2, 2)
    # We'll create a simplified version since we can't easily embed the landscape here
    plt.text(0.5, 0.5, 'Energy Landscape\n能量景观\n(See separate plot)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig("temperature_controlled_training.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot final energy landscape / 绘制最终能量景观
    plot_energy_landscape_2d(
        model, data,
        "Temperature-Controlled Wake-Sleep EBM Energy Landscape",
        "temperature_controlled_energy.png"
    )
    
    return model, trainer

def demonstrate_bidirectional_wake_sleep():
    """
    Demonstrate Bidirectional Wake-Sleep EBM
    演示双向Wake-Sleep EBM
    """
    print("\n" + "="*60)
    print("Bidirectional Wake-Sleep EBM Demonstration")
    print("双向Wake-Sleep EBM演示")
    print("="*60)
    
    # Create higher-dimensional dataset / 创建高维数据集
    data = create_synthetic_dataset(1000, dim=100, n_modes=2)
    dataset = TensorDataset(data, torch.zeros(len(data)))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create forward and backward networks / 创建前向和后向网络
    forward_net = nn.Sequential(
        nn.Linear(100, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 1)
    )
    
    backward_net = nn.Sequential(
        nn.Linear(50, 128),  # Latent dimension = 50
        nn.LeakyReLU(0.2),
        nn.Linear(128, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 100),  # Output dimension = 100
        nn.Tanh()
    )
    
    model = BiDirectionalWakeSleepEBM(forward_net, backward_net)
    trainer = WakeSleepEBMTrainer(model)
    
    print("Training Bidirectional Wake-Sleep EBM...")
    print("训练双向Wake-Sleep EBM...")
    
    for epoch in range(5):
        losses = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1}: {losses}")
        print(f"第{epoch+1}轮: {losses}")
    
    # Generate some samples / 生成一些样本
    print("\nGenerating samples from backward network...")
    print("从后向网络生成样本...")
    
    with torch.no_grad():
        noise = torch.randn(10, 50)
        generated_samples = model.backward_net(noise)
        print(f"Generated samples shape: {generated_samples.shape}")
        print(f"生成样本形状: {generated_samples.shape}")
        print(f"Sample statistics - Mean: {generated_samples.mean():.3f}, Std: {generated_samples.std():.3f}")
        print(f"样本统计 - 均值: {generated_samples.mean():.3f}, 标准差: {generated_samples.std():.3f}")
    
    return model, trainer

def demonstrate_adaptive_wake_sleep():
    """
    Demonstrate Adaptive Wake-Sleep EBM
    演示自适应Wake-Sleep EBM
    """
    print("\n" + "="*60)
    print("Adaptive Wake-Sleep EBM Demonstration")
    print("自适应Wake-Sleep EBM演示")
    print("="*60)
    
    # Create dataset / 创建数据集
    data = create_2d_visualization_dataset(1000)
    dataset = TensorDataset(data, torch.zeros(len(data)))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create adaptive model / 创建自适应模型
    energy_net = create_energy_network_2d()
    model = AdaptiveWakeSleepEBM(
        energy_net,
        adaptation_window=50,
        wake_threshold=0.05,
        sleep_threshold=0.05,
        data_dim=(2,)  # Specify 2D data dimension
    )
    
    trainer = WakeSleepEBMTrainer(model)
    
    print("Training Adaptive Wake-Sleep EBM...")
    print("训练自适应Wake-Sleep EBM...")
    
    # Track which phases are executed / 跟踪执行的阶段
    phase_counts = {'wake': 0, 'sleep': 0}
    
    for epoch in range(10):
        losses = trainer.train_epoch(dataloader, epoch, mode='both')
        
        # Count phases in recent history / 统计最近历史中的阶段
        recent_phases = model.phase_history[-100:]  # Last 100 steps
        epoch_wake = recent_phases.count('wake')
        epoch_sleep = recent_phases.count('sleep')
        phase_counts['wake'] += epoch_wake
        phase_counts['sleep'] += epoch_sleep
        
        print(f"Epoch {epoch+1}: {losses}")
        print(f"  Recent phases - Wake: {epoch_wake}, Sleep: {epoch_sleep}")
        print(f"第{epoch+1}轮: {losses}")
        print(f"  最近阶段 - 觉醒: {epoch_wake}, 睡眠: {epoch_sleep}")
    
    print(f"\nTotal phase executions / 总阶段执行次数:")
    print(f"Wake: {phase_counts['wake']}, Sleep: {phase_counts['sleep']}")
    print(f"觉醒: {phase_counts['wake']}, 睡眠: {phase_counts['sleep']}")
    
    # Plot energy landscape / 绘制能量景观
    plot_energy_landscape_2d(
        model, data,
        "Adaptive Wake-Sleep EBM Energy Landscape",
        "adaptive_wake_sleep_energy.png"
    )
    
    return model, trainer

def compare_all_variants():
    """
    Compare all wake-sleep variants on the same dataset
    在相同数据集上比较所有wake-sleep变体
    """
    print("\n" + "="*60)
    print("Comparing All Wake-Sleep Variants")
    print("比较所有Wake-Sleep变体")
    print("="*60)
    
    # Create common dataset / 创建公共数据集
    data = create_2d_visualization_dataset(800)
    dataset = TensorDataset(data, torch.zeros(len(data)))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Define variants to compare / 定义要比较的变体
    variants = {
        'Classical': lambda: ClassicalWakeSleepEBM(
            create_energy_network_2d(), sampler_steps=8, data_dim=(2,)
        ),
        'Alternating': lambda: AlternatingWakeSleepEBM(
            create_energy_network_2d(), sampler_steps=8, data_dim=(2,)
        ),
        'Temperature': lambda: TemperatureControlledWakeSleepEBM(
            create_energy_network_2d(), initial_temp=1.5, sampler_steps=8, data_dim=(2,)
        ),
        'Adaptive': lambda: AdaptiveWakeSleepEBM(
            create_energy_network_2d(), adaptation_window=30, sampler_steps=8, data_dim=(2,)
        )
    }
    
    results = {}
    
    # Train each variant / 训练每个变体
    for name, model_factory in variants.items():
        print(f"\nTraining {name} Wake-Sleep EBM...")
        print(f"训练{name} Wake-Sleep EBM...")
        
        model = model_factory()
        trainer = WakeSleepEBMTrainer(model)
        
        start_time = time.time()
        
        # Train for fewer epochs for comparison / 为比较训练更少的epoch
        for epoch in range(3):
            losses = trainer.train_epoch(dataloader, epoch)
            
        training_time = time.time() - start_time
        
        # Get final statistics / 获取最终统计
        stats = trainer.get_training_stats()
        results[name] = {
            'final_wake_loss': stats['final_wake_loss'],
            'final_sleep_loss': stats['final_sleep_loss'],
            'training_time': training_time,
            'wake_trend': stats['wake_loss_trend'],
            'sleep_trend': stats['sleep_loss_trend']
        }
        
        print(f"  Final losses: Wake={stats['final_wake_loss']:.4f}, Sleep={stats['final_sleep_loss']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  最终损失: 觉醒={stats['final_wake_loss']:.4f}, 睡眠={stats['final_sleep_loss']:.4f}")
        print(f"  训练时间: {training_time:.2f}秒")
    
    # Print comparison table / 打印比较表
    print("\n" + "="*80)
    print("Comparison Results / 比较结果")
    print("="*80)
    print(f"{'Variant':<12} {'Wake Loss':<12} {'Sleep Loss':<12} {'Time(s)':<10} {'Wake Trend':<12} {'Sleep Trend':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<12} {result['final_wake_loss']:<12.4f} {result['final_sleep_loss']:<12.4f} "
              f"{result['training_time']:<10.2f} {result['wake_trend']:<12.4f} {result['sleep_trend']:<12.4f}")
    
    # Create comparison plot / 创建比较图
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Final losses / 图1：最终损失
    plt.subplot(1, 3, 1)
    variants_names = list(results.keys())
    wake_losses = [results[name]['final_wake_loss'] for name in variants_names]
    sleep_losses = [results[name]['final_sleep_loss'] for name in variants_names]
    
    x = np.arange(len(variants_names))
    width = 0.35
    
    plt.bar(x - width/2, wake_losses, width, label='Wake Loss', alpha=0.7)
    plt.bar(x + width/2, sleep_losses, width, label='Sleep Loss', alpha=0.7)
    plt.xlabel('Variants / 变体')
    plt.ylabel('Loss / 损失')
    plt.title('Final Losses Comparison / 最终损失比较')
    plt.xticks(x, variants_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training time / 图2：训练时间
    plt.subplot(1, 3, 2)
    times = [results[name]['training_time'] for name in variants_names]
    plt.bar(variants_names, times, alpha=0.7, color='orange')
    plt.xlabel('Variants / 变体')
    plt.ylabel('Time (seconds) / 时间(秒)')
    plt.title('Training Time Comparison / 训练时间比较')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss trends / 图3：损失趋势
    plt.subplot(1, 3, 3)
    wake_trends = [results[name]['wake_trend'] for name in variants_names]
    sleep_trends = [results[name]['sleep_trend'] for name in variants_names]
    
    plt.bar(x - width/2, wake_trends, width, label='Wake Trend', alpha=0.7)
    plt.bar(x + width/2, sleep_trends, width, label='Sleep Trend', alpha=0.7)
    plt.xlabel('Variants / 变体')
    plt.ylabel('Loss Trend / 损失趋势')
    plt.title('Loss Trends Comparison / 损失趋势比较')
    plt.xticks(x, variants_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("wake_sleep_variants_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def main():
    """
    Main demonstration function
    主演示函数
    """
    print("Wake-Sleep EBM Variants Comprehensive Demonstration")
    print("Wake-Sleep EBM变体综合演示")
    print("=" * 60)
    print("\nBased on the classical wake-sleep algorithm from Helmholtz Machine (1994)")
    print("基于1994年Helmholtz Machine的经典wake-sleep算法")
    print("\nThis demonstration shows various adaptations for Energy-Based Models:")
    print("这个演示展示了能量基模型的各种适应性变体：")
    print("1. Classical Wake-Sleep EBM / 经典Wake-Sleep EBM")
    print("2. Temperature-Controlled Wake-Sleep EBM / 温度控制Wake-Sleep EBM") 
    print("3. Bidirectional Wake-Sleep EBM / 双向Wake-Sleep EBM")
    print("4. Adaptive Wake-Sleep EBM / 自适应Wake-Sleep EBM")
    
    # Create output directory / 创建输出目录
    os.makedirs("wake_sleep_demo_outputs", exist_ok=True)
    os.chdir("wake_sleep_demo_outputs")
    
    try:
        # Run individual demonstrations / 运行单独演示
        print("\nRunning individual demonstrations...")
        print("运行单独演示...")
        
        classical_results = demonstrate_classical_wake_sleep()
        temp_results = demonstrate_temperature_controlled()
        bidirectional_results = demonstrate_bidirectional_wake_sleep()
        adaptive_results = demonstrate_adaptive_wake_sleep()
        
        # Run comparison / 运行比较
        print("\nRunning comprehensive comparison...")
        print("运行综合比较...")
        comparison_results = compare_all_variants()
        
        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("所有演示成功完成！")
        print("Check the 'wake_sleep_demo_outputs' directory for generated plots.")
        print("检查'wake_sleep_demo_outputs'目录中的生成图表。")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Go back to parent directory / 返回父目录
        os.chdir("..")

if __name__ == "__main__":
    # Set random seeds for reproducibility / 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 