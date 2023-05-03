from scipy import signal, fft
from matplotlib import pyplot as plt
from obspy.signal.util import next_pow_2
from constants import *
import numpy as np
import pandas as pd
import librosa, copy

def main():
    time_data_s, sr_s = librosa.load(file_names[11], sr=None)
    time_data_s = butter_band_filter(time_data_s, low_high, sr_s, order)
    time_data_ns, sr_ns = librosa.load(file_names[9], sr=None)
    time_data_ns = butter_band_filter(time_data_ns, low_high, sr_ns, order)
    time_data_n, sr_n = librosa.load(file_names[12], sr=None)
    time_data_n = butter_band_filter(time_data_n, low_high, sr_n, order)
    
    filtered_data = matched_filter(time_data_s, time_data_ns)
    
    plt.hist(time_data_ns, bins=100, density=True, label='PDF Noisy Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.hist(filtered_data, bins=100, density=True, label='PDF Filtered', lw=1, alpha=0.5)
    plt.hist(time_data_ns, bins=100, density=True, label='PDF Noisy Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    sub_freq, sub_psd = signal.welch(x=filtered_data, fs=sr_ns, nperseg=32768, average='mean')
    n_freq, n_psd = signal.welch(x=time_data_n, fs=sr_n, nperseg=32768, average='mean')
    ns_freq, ns_psd = signal.welch(x=time_data_ns, fs=sr_ns, nperseg=32768, average='mean')
    sub_freq = sub_freq[sub_freq >= 50]
    sub_freq = sub_freq[sub_freq <= 2000]
    sub_psd = sub_psd[:len(sub_freq)]
    n_freq = n_freq[n_freq >= 50]
    n_freq = n_freq[n_freq <= 2000]
    n_psd = n_psd[:len(n_freq)]
    ns_freq = ns_freq[ns_freq >= 50]
    ns_freq = ns_freq[ns_freq <= 2000]
    ns_psd = ns_psd[:len(ns_freq)]
    
    sub_psd = 10*np.log10(sub_psd/(10**(-12)))
    n_psd = 10*np.log10(n_psd/(10**(-12)))
    ns_psd = 10*np.log10(ns_psd/(10**(-12)))
    
    plt.plot(sub_freq, sub_psd, label='H(S + N)', lw=1, alpha=0.5)
    plt.plot(n_freq, n_psd, label='N', lw=1, alpha=0.5)
    plt.plot(ns_freq, ns_psd, label='S + N', lw=1, alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.legend()
    plt.show()

def matched_filter(sig_time_data, ns_time_data, N=None):
    ns_time_data = spec_white(ns_time_data, N)
    matched_filt = np.conj(sig_time_data[::-1])
    return signal.convolve(ns_time_data, matched_filt)

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

def spec_white(data, N=None):
    n = len(data)
    data_2 = copy.deepcopy(data)[:n]
    nfft = next_pow_2(n)
    spec = fft.fft(data_2, nfft)
    spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
    if not N == None:
        shift = N // 2
        spec = spec[shift:-shift]
        spec_ampl = pd.Series(spec_ampl).rolling(N, center=True).mean().to_numpy()[shift:-shift]
    spec = np.true_divide(spec, spec_ampl)
    return np.real(fft.ifft(spec, nfft))

if __name__ == '__main__':
    main()