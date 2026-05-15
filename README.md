# DT-DCF: Dynamic Temporal Modeling and Distribution-Calibrated Fusion for Multimodal Sentiment Analysis

This repository provides an anonymized implementation of the paper **“DT-DCF: Dynamic Temporal Modeling and Distribution-Calibrated Fusion for Multimodal Sentiment Analysis”**.

DT-DCF is a confidence-modulated fusion framework for multimodal sentiment analysis. It estimates modality confidence from temporal granularity selection distributions and Gaussian representation uncertainty, and uses the estimated confidence to dynamically modulate cross-modal fusion and conditional representation alignment.

The model first performs adaptive temporal modeling on audio and visual sequences to capture sample-specific emotional dynamics. Then, textual, acoustic, and visual representations are modeled with distributional uncertainty to estimate modality confidence, which is used to modulate cross-modal fusion. Finally, the fused representation is aligned with unimodal representations through a confidence-guided conditional alignment process.

Experiments are conducted on the CMU-MOSI, CMU-MOSEI, and CH-SIMS datasets. 

---

## Requirements

The recommended environment is as follows:

```bash
python >= 3.8
pytorch >= 1.12
cuda >= 11.3
```

The main dependencies include:

```text
numpy
scipy
scikit-learn
pandas
tqdm
torch
transformers
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset preparation

This project supports the following multimodal sentiment analysis datasets:

- CMU-MOSI
- CMU-MOSEI
- CH-SIMS

Please download and preprocess the corresponding datasets, and place the processed data files under the `data/` directory.

Due to the original data release licenses and usage agreements, this repository does not provide the raw dataset files. Please obtain the datasets according to the official requirements of each benchmark.
