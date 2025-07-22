import numpy as np
from scipy import stats, signal
from scipy.signal import welch, find_peaks, detrend
from sklearn.preprocessing import StandardScaler

def extract_time_domain(signal):
    """
    Extract time-domain features
    """
    mean = np.mean(signal)
    median = np.median(signal)
    q25, q75 = np.percentile(signal, [25,75])
    iqr = q75 - q25
    std = np.std(signal)
    ptp = np.ptp(signal)
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)
    rms = np.sqrt(np.mean(signal**2))
    energy = np.sum(signal**2)
    crest = np.max(np.abs(signal)) / rms
    zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
    mac = np.mean(np.abs(np.diff(signal)))
    return np.array([mean, median, q25, q75, iqr, std, ptp, skewness, kurtosis,
                     rms, energy, crest, zero_crossings, mac])

def extract_frequency_domain(signal, fs=50e6):
    """
    Extract frequency-domain features
    """
    f, Pxx = welch(signal, fs=fs, window='hann', nperseg=256, noverlap=128)
    Pxx_norm = Pxx / np.sum(Pxx)
    dom_freq = f[np.argmax(Pxx)]
    dom_power = np.max(Pxx)
    mean_power = np.mean(Pxx)
    median_power = np.median(Pxx)
    low_band = np.sum(Pxx[(f<=10e6)])
    high_band = np.sum(Pxx[(f>10e6) & (f<=50e6)])
    ratio_lh = low_band / (high_band + 1e-8)
    entropy = -np.sum(Pxx_norm*np.log(Pxx_norm+1e-12))
    centroid = np.sum(f*Pxx)/np.sum(Pxx)
    bw = np.sqrt(np.sum(((f-centroid)**2)*Pxx)/np.sum(Pxx))
    rolloff = f[np.where(np.cumsum(Pxx_norm) >= 0.85)[0][0]]
    flatness = stats.gmean(Pxx) / (np.mean(Pxx)+1e-8)
    skewness = stats.skew(Pxx)
    kurtosis = stats.kurtosis(Pxx)

    return np.array([dom_freq, dom_power, mean_power, median_power,
                     ratio_lh, entropy, centroid, bw, rolloff,
                     flatness, skewness, kurtosis])

def extract_noise_features(signal):
    """
    Extract noise-related features
    """
    window_size = max(15, len(signal)//2) | 1  # make odd
    trend = signal - signal
    for i in range(len(signal)):
        start = max(0,i-window_size//2)
        end = min(len(signal),i+window_size//2+1)
        trend[i] = np.median(signal[start:end])
    noise = signal - trend
    noise_var = np.var(noise)
    snr = np.var(signal) / (noise_var + 1e-8)
    rms = np.sqrt(np.mean(noise**2))
    autocorr = np.correlate(noise, noise, mode='full')[len(noise)-1:]
    autocorr /= autocorr[0]
    decay = np.where(autocorr<0.5)[0][0] if np.any(autocorr<0.5) else len(noise)
    f, Pxx = welch(noise)
    total_power = np.sum(Pxx)
    entropy = -np.sum(Pxx/np.sum(Pxx)*np.log(Pxx/np.sum(Pxx)+1e-12))
    peak_factor = np.max(np.abs(noise))/rms
    impulse_factor = np.max(np.abs(noise))/np.mean(np.abs(noise))
    shape_factor = rms/np.mean(np.abs(noise))
    crest_factor = np.max(noise)/rms

    return np.array([noise_var, snr, rms, decay, total_power, entropy,
                     peak_factor, impulse_factor, shape_factor, crest_factor])

def extract_morphological(signal):
    """
    Extract morphological features
    """
    std = np.std(signal)
    peaks,_ = find_peaks(signal,height=0.5*std,distance=5,prominence=0.5)
    valleys,_ = find_peaks(-signal,height=0.5*std,distance=5,prominence=0.5)
    n_peaks = len(peaks)
    n_valleys = len(valleys)
    if len(peaks)>1:
        peak_dist = np.mean(np.diff(peaks))
    else:
        peak_dist = 0
    if len(valleys)>1:
        valley_dist = np.mean(np.diff(valleys))
