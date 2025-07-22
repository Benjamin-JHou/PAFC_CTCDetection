import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def compute_f_statistics(X, y):
    """
    Compute F-statistics for each feature to measure discriminative power
    """
    f_scores = []
    labels = np.unique(y)
    for i in range(X.shape[1]):
        groups = [X[y==label, i] for label in labels]
        f_val, _ = f_oneway(*groups)
        f_scores.append(f_val)
    return np.array(f_scores)

def select_features_by_fstat(f_stats, num_core=20, num_subtype=30, num_progression=50):
    """
    Select features into three tiers: core, subtype, progression
    """
    idx_sorted = np.argsort(-f_stats)
    core_idx = idx_sorted[:num_core]
    subtype_idx = idx_sorted[num_core:num_core+num_subtype]
    progression_idx = idx_sorted[num_core+num_subtype:num_core+num_subtype+num_progression]
    return core_idx, subtype_idx, progression_idx

def build_fingerprint(X, core_idx, subtype_idx, progression_idx, weights=(1.0,0.7,0.4)):
    """
    Construct standardized fingerprint with hierarchical weights
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    fingerprint = np.zeros(X.shape)

    # Assign weighted intensities
    fingerprint[:, core_idx] = X_scaled[:, core_idx] * weights[0]
    fingerprint[:, subtype_idx] = X_scaled[:, subtype_idx] * weights[1]
    fingerprint[:, progression_idx] = X_scaled[:, progression_idx] * weights[2]

    return fingerprint

def plot_fingerprint(fingerprint, feature_labels=None, sample_idx=0, figsize=(12,4)):
    """
    Visualize fingerprint of one sample as bar plot with color gradation
    """
    fp = fingerprint[sample_idx]
    plt.figure(figsize=figsize)
    cmap = plt.cm.get_cmap("coolwarm")

    colors = cmap(fp/np.max(fp + 1e-8))
    plt.bar(np.arange(len(fp)), fp, color=colors)

    if feature_labels is not None:
        plt.xticks(np.arange(len(fp)), feature_labels, rotation=90, fontsize=8)
    else:
        plt.xticks(np.arange(len(fp)), np.arange(len(fp)), fontsize=8)

    plt.title(f"Photoacoustic Fingerprint - Sample {sample_idx}")
    plt.xlabel("Features")
    plt.ylabel("Weighted Intensity")
    plt.tight_layout()
    plt.show()
