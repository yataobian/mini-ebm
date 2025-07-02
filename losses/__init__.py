"""
Loss functions for Energy-Based Models (EBMs).
能量基模型 (EBM) 的损失函数。

This package contains various loss functions for training EBMs, including:
该包包含用于训练EBM的各种损失函数，包括：

- Contrastive Divergence (CD) variants / 对比散度 (CD) 变体
- Score Matching (SM) variants / 分数匹配 (SM) 变体  
- Noise Contrastive Estimation (NCE) variants / 噪声对比估计 (NCE) 变体
"""

# Import NCE variants from nce_variants.py
# 从 nce_variants.py 导入 NCE 变体
from .nce_variants import (
    NoiseContrastiveEstimationLoss
)

# Import Score Matching variants from sm_variants.py
# 从 sm_variants.py 导入分数匹配变体
from .sm_variants import (
    DenoisingScoreMatchingLoss
)

# Import CD variants from cd_variants.py
# 从 cd_variants.py 导入CD变体
from .cd_variants import (
    ContrastiveDivergenceLoss,
    PersistentContrastiveDivergenceLoss,
    FastPersistentContrastiveDivergenceLoss
    # TemperedContrastiveDivergenceLoss,
    # ParallelTemperingContrastiveDivergenceLoss,
    # AdaptiveContrastiveDivergenceLoss
)

__all__ = [
    # Score Matching variants / 分数匹配变体
    'DenoisingScoreMatchingLoss',
    
    # Noise Contrastive Estimation variants / 噪声对比估计变体
    'NoiseContrastiveEstimationLoss',
    
    # Contrastive Divergence variants / 对比散度变体
    'ContrastiveDivergenceLoss',
    'PersistentContrastiveDivergenceLoss',
    'FastPersistentContrastiveDivergenceLoss'
    # 'TemperedContrastiveDivergenceLoss',
    # 'ParallelTemperingContrastiveDivergenceLoss',
    # 'AdaptiveContrastiveDivergenceLoss'
] 