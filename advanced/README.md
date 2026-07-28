# Advanced: Wake-Sleep EBM Variants

This directory contains advanced material on **Wake-Sleep learning** (Helmholtz Machines) for Energy-Based Models, implemented independently from the main 2D toy-data teaching pipeline.

## ⚠️ Important Notes

1. **Not part of the guided learning path.** The main curriculum (notebooks 00-05) focuses on **Contrastive Divergence, Denoising Score Matching, and Noise Contrastive Estimation** (CD/PCD/FastPCD, DSM, NCE/AdaptiveNCE). Wake-Sleep is an orthogonal approach with its own mathematical framing.

2. **Dimensional incompatibility.** The `HierarchicalWakeSleepEBM` class in `wake_sleep_variants.py` hardcodes dimensions `(784, 256)`, suitable for MNIST-scale data. **It is not compatible** with the 2D toy data pipeline used elsewhere in this repository. If you want to adapt it to 2D, you'll need to modify the hardcoded dimensions manually.

3. **Standalone implementation.** Unlike the main pipeline, the code in this directory is self-contained: `wake_sleep_variants.py` defines all six Wake-Sleep variants, and `wake_sleep_demo.py` runs independently with its own data generation and visualization. These are excellent for understanding the Wake-Sleep algorithm in detail, but operate separately from `train.py` and `visualize_interactive.py`.

## 📚 Contents

- **`wake_sleep_variants.py`** (808 lines) — Six Wake-Sleep EBM variant classes:
  - `ClassicalWakeSleepEBM` — original Helmholtz machine formulation
  - `AlternatingWakeSleepEBM` — strict wake/sleep alternation
  - `BiDirectionalWakeSleepEBM` — separate forward/backward generative models
  - `TemperatureControlledWakeSleepEBM` — annealing temperature schedule
  - `HierarchicalWakeSleepEBM` — multi-level energy hierarchy (MNIST-specific)
  - `AdaptiveWakeSleepEBM` — adaptive wake/sleep phase duration

- **`wake_sleep_demo.py`** (547 lines) — Runnable demonstration comparing all six variants, generating plots and performance metrics.

- **`WAKE_SLEEP_ANALYSIS.md`** — Theoretical background on the 1994 Helmholtz Machine algorithm, mathematical derivations for each variant, computational-complexity analysis, stability notes, and references.

- **`WAKE_SLEEP_SUMMARY.md`** — Project summary documenting the implementation, design choices, and results.

## 🚀 Getting Started (if you want to explore this)

```bash
# From the mini-ebm root directory
cd advanced
python wake_sleep_demo.py
```

This will generate performance plots and energy landscape visualizations in `wake_sleep_demo_outputs/`.

## 📖 Further Reading

- See `WAKE_SLEEP_ANALYSIS.md` for the full theoretical treatment and implementation details.
- Compare with the main-curriculum materials (`notebooks/02_contrastive_divergence.ipynb` etc.) to understand how Wake-Sleep differs from CD, DSM, and NCE.

---

**Status:** This material is **optional / for advanced exploration**. It is not maintained in sync with changes to the main 2D pipeline. If you find it interesting and want to adapt it to other datasets, you're welcome to modify it independently!
