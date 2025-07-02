import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
# 导入自定义模块
from toy_data import get_toy_data
from models import EnergyNet
from samplers import LangevinSampler
from losses import ContrastiveDivergenceLoss, DenoisingScoreMatchingLoss, NoiseContrastiveEstimationLoss
from visualize import plot_energy_landscape

def main(args):
    # Set device (CPU or GPU)
    # 设置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    # 如果输出目录不存在，则创建它
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data Loading ---
    # --- 数据加载 ---
    data = get_toy_data(args.dataset, args.num_samples)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # --- Model Initialization ---
    # --- 模型初始化 ---
    energy_network = EnergyNet().to(device)

    # --- Loss and Sampler Initialization ---
    # --- 损失和采样器初始化 ---
    if args.loss_type == 'cd':
        sampler = LangevinSampler(energy_network, args.langevin_step, args.langevin_noise)
        loss_fn = ContrastiveDivergenceLoss(energy_network, sampler, k=args.cd_k)
    elif args.loss_type == 'dsm':
        loss_fn = DenoisingScoreMatchingLoss(energy_network, sigma=args.dsm_sigma)
    elif args.loss_type == 'nce':
        # For NCE, we need a noise distribution. A simple Gaussian is used here.
        # 对于 NCE，我们需要一个噪声分布。这里使用一个简单的高斯分布。
        noise_dist = torch.distributions.Normal(loc=0.0, scale=1.5)
        loss_fn = NoiseContrastiveEstimationLoss(energy_network, noise_dist, noise_ratio=args.nce_noise_ratio)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # --- Optimizer ---
    # --- 优化器 ---
    optimizer = optim.Adam(energy_network.parameters(), lr=args.lr)

    # --- Training Loop ---
    # --- 训练循环 ---
    print(f"Starting training with {args.loss_type.upper()} loss on {args.dataset} dataset...")
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        total_loss = 0
        for batch_data in dataloader:
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # --- Visualization (periodically) ---
        # --- 可视化（定期） ---
        if (epoch + 1) % args.plot_interval == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Get negative samples for visualization if using CD
            # 如果使用 CD，获取负样本用于可视化
            neg_samples = None
            if args.loss_type == 'cd' and hasattr(loss_fn, 'last_negative_samples'):
                neg_samples = loss_fn.last_negative_samples.cpu().numpy()

            plot_energy_landscape(
                energy_network,
                data.cpu().numpy(),
                neg_samples=neg_samples,
                ax=ax,
                title=f'Epoch {epoch + 1}',
                range_limit=8
            )
            
            # Save the plot
            # 保存图像
            plot_path = os.path.join(args.output_dir, f'epoch_{epoch + 1}.png')
            plt.savefig(plot_path)
            plt.close(fig)

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a minimal EBM on 2D toy data.')
    
    # General arguments / 通用参数
    parser.add_argument('--dataset', type=str, default='gmm', choices=['gmm', 'two_moons', 'checkerboard'], help='Toy dataset to use.')
    parser.add_argument('--loss_type', type=str, default='cd', choices=['cd', 'dsm', 'nce'], help='Loss function to use for training.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of data points.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--output_dir', type=str, default='./outputs/default', help='Directory to save plots.')
    parser.add_argument('--plot_interval', type=int, default=10, help='Interval (in epochs) for saving plots.')

    # CD arguments / CD 相关参数
    parser.add_argument('--cd_k', type=int, default=10, help='Number of Langevin steps for CD.')
    parser.add_argument('--langevin_step', type=float, default=0.1, help='Step size for Langevin dynamics.')
    parser.add_argument('--langevin_noise', type=float, default=0.01, help='Noise standard deviation for Langevin dynamics.')

    # DSM arguments / DSM 相关参数
    parser.add_argument('--dsm_sigma', type=float, default=0.1, help='Noise standard deviation for DSM.')

    # NCE arguments / NCE 相关参数
    parser.add_argument('--nce_noise_ratio', type=int, default=1, help='Number of noise samples per data sample for NCE.')

    args = parser.parse_args()
    main(args)
