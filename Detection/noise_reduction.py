from scipy import signal
from matplotlib import pyplot as plt
from acoustics import Signal
from constants import *
import noisereduce as nr
import numpy as np
import librosa

def main():    
    # time_data_ns, sr_ns = librosa.load(file_names[7], sr=None, mono=False)
    # time_data_n, sr_n = librosa.load(file_names[9], sr=None, mono=False)
    # time_data_n, sr_n = librosa.load(file_names[10], sr=None)
    
    time_data_ns = mult*Signal.from_wav(file_names[7], normalize=False)
    sr_ns = time_data_ns.fs
    time_data_n = mult*Signal.from_wav(file_names[10], normalize=False)
    sr_n = time_data_n.fs
    time_data_n = np.array(librosa.to_mono(time_data_n))
    
    time_data_ns[0] = butter_band_filter(time_data_ns[0], low_high, sr_ns, order)
    time_data_ns[1] = butter_band_filter(time_data_ns[1], low_high, sr_ns, order)
    time_data_ns[2] = butter_band_filter(time_data_ns[2], low_high, sr_ns, order)
    time_data_ns[3] = butter_band_filter(time_data_ns[3], low_high, sr_ns, order)
    time_data_n = butter_band_filter(time_data_n, low_high, sr_n, order)
    
    time_data_ns_red = np.empty(shape=(4, 441000))
    # time_data_ns_red[0] = nr.reduce_noise(y=time_data_ns[0], sr=sr_ns, y_noise=time_data_n, stationary=False)
    # time_data_ns_red[1] = nr.reduce_noise(y=time_data_ns[1], sr=sr_ns, y_noise=time_data_n, stationary=False)
    # time_data_ns_red[2] = nr.reduce_noise(y=time_data_ns[2], sr=sr_ns, y_noise=time_data_n, stationary=False)
    # time_data_ns_red[3] = nr.reduce_noise(y=time_data_ns[3], sr=sr_ns, y_noise=time_data_n, stationary=False)
    time_data_ns_red[0] = nr.reduce_noise(y=time_data_ns[0], sr=sr_ns)
    time_data_ns_red[1] = nr.reduce_noise(y=time_data_ns[1], sr=sr_ns)
    time_data_ns_red[2] = nr.reduce_noise(y=time_data_ns[2], sr=sr_ns)
    time_data_ns_red[3] = nr.reduce_noise(y=time_data_ns[3], sr=sr_ns)
    
    time_data_ns = np.array(librosa.to_mono(time_data_ns))
    time_data_ns_red = np.array(librosa.to_mono(time_data_ns_red))
    
    plt.hist(time_data_ns, bins=100, density=True, label='PDF Noisy Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.hist(time_data_ns_red, bins=100, density=True, label='PDF Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    sub_freq, sub_psd = signal.welch(x=time_data_ns_red, fs=sr_ns, nperseg=32768, average='mean')
    n_freq, n_psd = signal.welch(x=time_data_n, fs=sr_n, nperseg=32768, average='mean')
    ns_freq, ns_psd = signal.welch(x=time_data_ns, fs=sr_ns, nperseg=32768, average='mean')
    sub_freq = sub_freq[sub_freq <= 2000]
    sub_psd = sub_psd[:len(sub_freq)]
    n_freq = n_freq[n_freq <= 2000]
    n_psd = n_psd[:len(n_freq)]
    ns_freq = ns_freq[ns_freq <= 2000]
    ns_psd = ns_psd[:len(ns_freq)]
    
    plt.semilogy(sub_freq, sub_psd, label='H(S + N)', lw=1, alpha=0.5)
    plt.semilogy(n_freq, n_psd, label='N', lw=1, alpha=0.5)
    plt.semilogy(ns_freq, ns_psd, label='S + N', lw=1, alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
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