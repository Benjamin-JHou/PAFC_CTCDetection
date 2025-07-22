import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from glob import glob
import random

def load_signals(csv_folder, group_size=8, signal_length=512):
    all_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv') and 'n' in f]
    file_groups = {}
    for file in all_files:
        key = '_'.join(file.split('_', 3)[:3])
        file_groups.setdefault(key, []).append(file)

    signals_by_group = {}
    for i, key in enumerate(sorted(file_groups.keys()), 1):
        group_files = random.sample(file_groups[key], group_size)
        signals = []
        for file in group_files:
            path = os.path.join(csv_folder, file)
            data = pd.read_csv(path, header=None).values
            if data.shape[0] > data.shape[1]:
                data = data.T
            signals.append(data[:, :signal_length])
        signals_by_group[key] = np.vstack(signals)
    return signals_by_group

def preprocess_signals(signals):
    mean_signal = np.mean(signals, axis=0)
    centered_signals = signals - mean_signal
    return centered_signals, mean_signal

def pca_common_trend(centered_signals, sid=0):
    pca = PCA()
    score = pca.fit_transform(centered_signals)
    coeff = pca.components_
    common_trend = coeff[sid] * np.std(score[:, sid])
    return common_trend, pca
