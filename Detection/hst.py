from scipy import signal
from matplotlib import pyplot as plt
from constants import *
import numpy as np
import librosa

def main():
    time_data, sr = librosa.load(file_names[9], sr=None, mono=False)
    time_data[0] = butter_band_filter(time_data[0], low_high, sr, order)
    time_data[1] = butter_band_filter(time_data[1], low_high, sr, order)
    time_data[2] = butter_band_filter(time_data[2], low_high, sr, order)
    time_data[3] = butter_band_filter(time_data[3], low_high, sr, order)
    
    time_data[0] = butter_band_filter(time_data[0], special_filt, sr, special_order, 'bandstop')
    time_data[1] = butter_band_filter(time_data[1], special_filt, sr, special_order, 'bandstop')
    time_data[2] = butter_band_filter(time_data[2], special_filt, sr, special_order, 'bandstop')
    time_data[3] = butter_band_filter(time_data[3], special_filt, sr, special_order, 'bandstop')
    
    # time_data, sr = librosa.load(file_names[13], sr=None, mono=False)
    # time_data = butter_bandpass_filter(time_data, low_high, sr, order)
    
    if len(time_data) > 4:
        time_data = [time_data]
    for s, dat in zip(range(len(time_data)), time_data):
        tm = len(dat)/sr
        pow_of_2 = np.int64(np.floor(np.log2(len(dat))))
        this_s = signal.resample(dat, np.power(2, pow_of_2))
        new_sr = len(this_s)/tm
        for r in range(1, num_harmonics + 1):
            if r == 1:
                this_s_spec = fft_vectorized(this_s, r)
            else:
                this_s_spec = np.vstack((this_s_spec, fft_vectorized(this_s, r)))
        this_s_spec = np.power(np.abs(this_s_spec), mean_type)
        this_s_spec = np.power(np.sum(this_s_spec, axis=0)/num_harmonics, 1/mean_type)
        freqs = librosa.fft_frequencies(sr=new_sr, n_fft=len(this_s)*2)[1:]
        
        if s == 0:
            this_sig_spec = this_s_spec
        else:
            this_sig_spec = np.vstack((this_sig_spec, this_s_spec))
    this_sig_spec = np.power(np.abs(this_sig_spec), mean_type)
    this_sig_spec = np.power(np.sum(this_sig_spec, axis=0)/len(time_data), 1/mean_type)
    
    # this_sig_spec = (this_sig_spec - np.min(this_sig_spec))/(np.max(this_sig_spec) - np.min(this_sig_spec)) + 1
    # this_sig_spec = 20*np.log10(this_sig_spec/1)
    freqs = freqs[freqs < 1501]
    
    plt.plot(2*freqs, this_sig_spec[:len(freqs)], label='', lw=1, alpha=0.75)
    plt.title('Harmonic Spectral Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
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

def fft_vectorized(sig, r_harmonic):
    sig = np.asarray(sig, dtype=float)
    big_N = sig.shape[0]
    if np.log2(big_N) % 1 > 0:
        raise ValueError("must be a power of 2")
    min_N = min(big_N, 2)
    n = np.arange(min_N)
    k = n[:, None]
    
    exp_term = np.exp(-2j * np.pi * n * k * r_harmonic / min_N)
    sig = sig.reshape(min_N, -1)
    sum_term = np.dot(exp_term, sig)
    while sum_term.shape[0] < big_N:
        even = sum_term[:, :int(sum_term.shape[1] / 2)]
        odd = sum_term[:, int(sum_term.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(sum_term.shape[0]) / sum_term.shape[0])[:, None]
        sum_term = np.vstack([even + terms * odd, even - terms * odd])
    return sum_term.ravel()

if __name__ == '__main__':
    main()