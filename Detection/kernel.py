from scipy import signal, fft
from matplotlib import pyplot as plt
from acoustics import Signal
from sklearn.kernel_approximation import Nystroem
from constants import *
import numpy as np
import pandas as pd
import librosa, time, copy

def main():
    # Kernel type list: 'additive_chi2', 'chi2', 'linear', 'poly' or 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'
    kernel = Nystroem(kernel=kernel_type, n_components=dim_mult, gamma=gamma, degree=degree, coef0=coeff)
    
    time_data_ns, sr_ns = librosa.load(file_names[4], sr=None, mono=False)
    time_data_ns[0] = butter_band_filter(time_data_ns[0], low_high, sr_ns, order)
    time_data_ns[1] = butter_band_filter(time_data_ns[1], low_high, sr_ns, order)
    time_data_ns[2] = butter_band_filter(time_data_ns[2], low_high, sr_ns, order)
    time_data_ns[3] = butter_band_filter(time_data_ns[3], low_high, sr_ns, order)
    time_data_ns = np.transpose(time_data_ns)
    time_data_ns = kernel.fit_transform(time_data_ns)
    time_data_ns = time_data_ns.flatten()
    
    time_data_n, sr_n = librosa.load(file_names[13], sr=None, mono=False)
    time_data_n[0] = butter_band_filter(time_data_n[0], low_high, sr_n, order)
    time_data_n[1] = butter_band_filter(time_data_n[1], low_high, sr_n, order)
    time_data_n[2] = butter_band_filter(time_data_n[2], low_high, sr_n, order)
    time_data_n[3] = butter_band_filter(time_data_n[3], low_high, sr_n, order)
    time_data_n = np.transpose(time_data_n)
    time_data_n = kernel.fit_transform(time_data_n)
    time_data_n = time_data_n.flatten()

    plt.hist(time_data_ns, bins=100, density=True, label='PDF Noisy Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()

def butter_band(low_high, fs, order, filt_type='bandpass'):
    nyq = 0.5 * fs
    low = low_high[0] / nyq
    high = low_high[1] / nyq
    b, a = signal.butter(order, [low, high], btype=filt_type)
    return b, a

def butter_band_filter(data, low_high, fs, order, filt_type='bandpass'):
    b, a = butter_band(low_high, fs, order, filt_type)
    y = signal.filtfilt(b, a, data)
    # y[~np.isfinite(y)] = 0.0
    y = y.astype(np.float32)
    return y

if __name__ == '__main__':
    main()