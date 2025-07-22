import numpy as np
from skimage.filters import threshold_otsu
from pykalman import KalmanFilter

def otsu_thresholding(variances):
    norm_var = (variances - np.percentile(variances, 5)) / (np.percentile(variances, 95) - np.percentile(variances, 5))
    level = threshold_otsu(norm_var)
    var_threshold = level * (np.percentile(variances, 95) - np.percentile(variances, 5)) + np.percentile(variances, 5)
    high_var_mask = variances > var_threshold
    return high_var_mask, var_threshold

def adaptive_segmentation(common_trend):
    signal_part1, signal_part2 = common_trend[:80], common_trend[80:300]
    var1 = np.array([np.var(signal_part1[max(0,i-2):i+3]) for i in range(len(signal_part1))])
    var2 = np.array([np.var(signal_part2[max(0,i-16):i+17]) for i in range(len(signal_part2))])

    mask1, _ = otsu_thresholding(var1)
    mask2, _ = otsu_thresholding(var2)

    starts1 = np.where(np.diff(np.concatenate(([0], mask1, [0]))) == 1)[0]
    ends1 = np.where(np.diff(np.concatenate(([0], mask1, [0]))) == -1)[0] - 1
    starts2 = np.where(np.diff(np.concatenate(([0], mask2, [0]))) == 1)[0] + 80
    ends2 = np.where(np.diff(np.concatenate(([0], mask2, [0]))) == -1)[0] - 1 + 80

    osc1 = [starts1[-1], ends1[-1]]
    osc2 = [starts2[0], ends2[0]]
    k = round((osc1[1] + osc2[0]) / 2)

    seg1, seg2, seg3 = common_trend[:k], common_trend[k:300], common_trend[300:]
    return seg1, seg2, seg3, k, osc1, osc2

def kalman_denoise(signal, segpoint1, segpoint2):
    pre = signal[:segpoint1]
    target = signal[segpoint1:segpoint2]
    post = signal[segpoint2:]

    obs_noise_var = np.var(post)
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[target[0], 0],
        transition_matrices=[[1,1],[0,1]],
        observation_matrices=[[1,0]],
        observation_covariance=obs_noise_var,
        transition_covariance=np.diag([0.1,0.01]),
        initial_state_covariance=np.eye(2)
    )

    kf = kf.em(target.reshape(-1,1), n_iter=50, em_vars=[
        'transition_matrices',
        'transition_covariance',
        'initial_state_mean',
        'initial_state_covariance'
    ])

    smoothed, _ = kf.smooth(target.reshape(-1,1))
    smoothed_target = smoothed[:,0]

    denoised_signal = np.concatenate([pre, smoothed_target, post])
    return denoised_signal
