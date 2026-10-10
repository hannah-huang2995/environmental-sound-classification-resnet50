# Environmental Sound Classification with ResNet50

**Deep learning | Audio classification | PyTorch | 5-fold cross-validation**

A supervised machine learning pipeline for classifying short environmental audio recordings into **10 sound categories**. The project converts raw audio into time-frequency features, trains a **ResNet50 from scratch**, and uses stratified cross-validation and probability ensembling to improve robustness.

**Final result: 0.87 on the Kaggle public leaderboard.**

## Project overview

Environmental sounds, such as sirens, dog barks, and engine noise, can be difficult to distinguish because recordings vary in duration and acoustic conditions. This project predicts a class ID (0–9) from each audio clip using a convolutional neural network (CNN) applied to spectrogram representations.

The project was developed for the CSCI-SHU 360 Machine Learning final competition (Fall 2025).

## Approach

```text
Raw WAV audio
    → resample to 22,050 Hz; trim/pad to 4 seconds
    → extract log-Mel spectrogram + first/second-order deltas
    → standardize each channel; resize to 3 × 224 × 224
    → ResNet50 classifier (trained from scratch)
    → 10-class prediction
```

### 1. Audio preprocessing

- Resample recordings to **22,050 Hz** and fix their duration at **4 seconds** (88,200 samples), using truncation or zero padding.
- Compute **128-bin log-Mel spectrograms** with `n_fft=2048` and `hop_length=512`.
- Build three input channels: **log-Mel**, **first-order delta (Δ)**, and **second-order delta (Δ²)**. The delta channels capture changes over time.
- Apply per-channel z-score standardization and resize to **224 × 224** for CNN input.

### 2. Model architecture

The classifier uses `torchvision.models.resnet50` with **no pretrained weights**. A custom classification head replaces the original final layer:

- `Dropout(p=0.2)`
- `Linear(in_features, 10)`

ResNet50 was chosen to learn richer time-frequency patterns than the earlier CNN baseline. Its residual connections support training a deeper network, while regularization is used to manage overfitting.

### 3. Training and regularization

| Component | Configuration / purpose |
| --- | --- |
| Mixup | `α = 0.3`, to encourage smoother decision boundaries |
| SpecAugment | Frequency/time masking to increase robustness |
| Additional augmentation | Random gain, Gaussian noise, and small time shifts |
| Loss | Class-weighted cross-entropy with label smoothing (`0.05`) |
| Optimizer | AdamW (`lr=7e-4`, `weight_decay=1e-4`) |
| Learning-rate schedule | OneCycleLR |
| Training stability | Gradient clipping (`max_norm=1.0`) and CUDA mixed precision when available |

### 4. Validation and ensembling

To reduce dependence on a single train/validation split, the final approach uses **5-fold stratified cross-validation**, preserving similar class distributions across folds.

For each fold:

1. Train on four folds and validate on the remaining fold.
2. Select the best checkpoint using **`0.8 × Accuracy + 0.2 × Macro-F1`**.
3. Generate test-set probabilities from the selected model.

The final prediction averages softmax probabilities across the five fold models and selects the highest-probability class.

## Experiments and results

The model was developed iteratively, moving from a custom CNN baseline to deeper residual networks and stronger validation.

| Model / experiment | Reported result |
| --- | --- |
| CNN baseline | Mid-70% validation accuracy; approximately **0.77** Kaggle score |
| ResNet34 with cross-validation and ensembling | Approximately **0.80** score |
| **Final ResNet50 pipeline** | **0.87 Kaggle public leaderboard score** |

These scores are reported in the accompanying competition report; the final **0.87 is a public leaderboard score**, not a separately reported cross-validation accuracy or private leaderboard result.

## Repository contents

```text
.
├── README.md                   # Project summary and methodology
├── resnet50.ipynb              # Model development / training notebook
├── ML_Competition_Report.pdf   # Detailed methods and experimental discussion
└── submission.csv              # Competition predictions (optional)
```

The dataset and trained `.pth` checkpoints are not included in this lightweight project showcase. The original data should be obtained through its authorized competition or dataset source where available. Running the notebook may require updating local input/output paths and installing its imported Python packages.

## Tools and libraries

**Python**, **PyTorch**, **torchvision**, **librosa**, and **scikit-learn** are used in the documented pipeline, along with supporting audio processing and model-training utilities.

## Key takeaways

- Designed an end-to-end audio classification workflow, from waveform preprocessing to ensemble inference.
- Compared successive model architectures and adjusted the training strategy based on validation behavior.
- Used stratified cross-validation, Macro-F1, class weighting, and regularization to emphasize generalization beyond one data split.
- Achieved a **0.87 Kaggle public leaderboard score** with the final ResNet50 ensemble.

## Further details

See [`ML_Competition_Report.pdf`](ML_Competition_Report.pdf) for the full technical report, hyperparameters, experiment descriptions, and Kaggle result documentation.
