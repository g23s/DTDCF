# DT-DCF: Dynamic Temporal-Domain Causal Fusion Network

This repository provides a **partial implementation** of DT-DCF for multimodal sentiment analysis.

------

## Highlights

A unified framework that integrates dynamic temporal modeling and causal-aware cross-modal fusion to improve robustness against spurious correlations in multimodal sentiment analysis.

------

## Experimental Environment

The experiments are conducted under the following settings:

- Framework: PyTorch
- GPU: NVIDIA A40 (single GPU)
- Python: 3.8
- Optimizer: AdamW
- Training Strategy:
  - Module-wise learning rates
  - Learning rate warmup
  - Weight decay

### Modalities

- **Text**: BERT-base encoder (768-dim representations)
- **Audio**: COVAREP features + MFCC (≈113-dim)
- **Vision**: OpenFace features (≈35-dim)

------

## Datasets

The experiments involve the following datasets:

- CMU-MOSI
- CMU-MOSEI
- CH-SIMS

**Note:**

- Due to dataset license and preprocessing dependencies, **data loading scripts are not included**.
- Users are expected to prepare datasets following standard protocols.

------

## Usage

This repository contains core components of the DT-DCF model.

```bash
python main.py
```

This code is intended for **research reference only** and may require additional preprocessing or configuration to fully reproduce results.

------

## Project Structure

```text
.
├── main.py
├── solver.py
├── model.py
├── MSA.py
└── utils/
```

------

## Important Notes

- This repository provides **partial implementation** for research and review purposes.
- Some components (e.g., data preprocessing, full training pipeline, and certain implementation details) are **not publicly released**.
- Pretrained model weights are not included.

------

