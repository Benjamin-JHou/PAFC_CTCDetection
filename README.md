
# PAFC_CTCDetection: Photoacoustic Flow Cytometry Melanoma Signal Analysis

This repository provides a complete workflow and reproducible codebase for **photoacoustic flow cytometry (PAFC)** signal analysis to distinguish **melanoma circulating tumor cells (CTCs)** from healthy controls. It covers signal denoising, feature extraction (time-domain, frequency-domain, noise, morphology), fingerprint construction, gene association, classification, and self-supervised learning models (Transformer & Mamba).  

---

## 📋 Features

✅ Data pre-processing, adaptive segmentation and signal denoising with Kalman & wavelet filters.  
✅ Time-domain, frequency-domain, noise, and morphological feature extraction.  
✅ Hierarchical photoacoustic fingerprint construction.  
✅ Gene-immunogenicity association and statistical analysis.  
✅ Multi-algorithm classification pipeline with cross-validation & feature selection.  
✅ Transformer & Mamba-based self-supervised representation learning.  
✅ Multi-task CNN-Transformer model with attention heatmaps & Grad-CAM.  

---

## 🖥️ Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- scikit-learn
- scipy
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Examples

### 1️⃣ Feature extraction & classification

```python
from feature_extraction import (
    extract_time_domain, extract_frequency_domain,
    extract_noise_features, extract_morphological
)
from classification import prepare_features, feature_selection, evaluate_model
from sklearn.ensemble import RandomForestClassifier

# X_raw: samples × features (concatenate time, freq, noise, morph)
# y: labels (0=Healthy, 1=Melanoma)
X_raw, y = ..., ...

# Preprocess & select features
X_scaled, y = prepare_features(X_raw, y)
idx = feature_selection(X_scaled, y)
X_selected = X_scaled[:, idx]

# Train & evaluate classifier
clf = RandomForestClassifier()
results = evaluate_model(X_selected, y, clf)
print(results)
```

---

### 2️⃣ Fingerprint construction & visualization

```python
from fingerprint import (
    compute_f_statistics, select_features_by_fstat,
    build_fingerprint, plot_fingerprint
)

# Input: X (n_samples × n_features), y (labels: 0=Healthy, 1=Melanoma)
X, y = ..., ...

# Compute F-statistics
f_stats = compute_f_statistics(X, y)

# Select hierarchical feature tiers
core_idx, subtype_idx, progression_idx = select_features_by_fstat(f_stats)

# Build fingerprint matrix
fingerprint = build_fingerprint(X, core_idx, subtype_idx, progression_idx)

# Plot fingerprint of first sample
plot_fingerprint(fingerprint, sample_idx=0)
```

---

### 3️⃣ Self-supervised Transformer & Mamba

```python
from models import PhotoacousticTransformer, PAFCMamba, masked_reconstruction_loss
import torch, torch.nn as nn, torch.optim as optim

# Example data
batch_size, seq_len, input_dim = 16, 100, 64
x_batch = torch.randn(batch_size, seq_len, input_dim)
mask = torch.rand_like(x_batch[:,:,0]) < 0.15

# Stage 1: Pretraining
model = PhotoacousticTransformer(input_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(10):
    model.train()
    y_pred = model(x_batch, task='reconstruct')
    loss = masked_reconstruction_loss(y_pred, x_batch, mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Pretrain Epoch {epoch} Loss: {loss.item():.4f}")

# Stage 2: Fine-tuning
labels = torch.randint(0,2,(batch_size,), dtype=torch.float)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
bce = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    logits = model(x_batch, task='classify')
    loss = bce(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Finetune Epoch {epoch} Loss: {loss.item():.4f}")
```

---

## 📝 Training Notes

| Stage         | Optimizer | Learning Rate | Epochs  | Dropout | Mask |
|---------------|-----------|----------------|---------|---------|------|
| Pretrain      | Adam      | 1e-3           | 10–50   | 0.1     | 15%  |
| Fine-tune     | Adam      | 1e-4           | 10–50   | 0.1/0.2 | -    |

- Transformer uses dropout=0.1.
- Mamba uses dropout=0.2.
- Both support masked reconstruction & classification.
- Weight decay: 1e-4.
- BCE loss for classification.

---

## 📚 Data Sources

- **PAFC signals:** Experimentally acquired at 532/1064 nm from melanoma patients & healthy controls.
- **Gene dataset:** [Curatopes Melanoma database](https://www.curatopes.com)

---

## 🗃️ Project Structure

```
├── src/ 
│ ├── classification.py
│ ├── cnn_transformer.py
│ ├── data_processing.py
│ ├── feature_extraction.py
│ ├── fingerprint.py
│ ├── gene_analysis.py
│ ├── self_supervised_transformer_mamba.py
│ ├── signal_denoising.py
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Acknowledgments

- Photoacoustic flow cytometry team for providing the data and domain expertise.
- Curatopes for the gene-epitope datasets.

---

## 🔗 References

1. [Scikit-learn documentation](https://scikit-learn.org/)
2. [PyTorch documentation](https://pytorch.org/)
3. Curatopes Melanoma database
4. Transformer and Mamba papers cited in the respective modules.

---

Feel free to open issues or submit pull requests if you have questions or improvements!
