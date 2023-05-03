from scipy import signal, fft
from matplotlib import pyplot as plt
from obspy.signal.util import next_pow_2
from geopy.distance import distance as dst
from constants import *
import numpy as np
import pandas as pd
import librosa, copy

def main():
    spkr_loc = [35.12323630, -89.81369912, 51.132]
    spkr_loc_std = [3.508, 4.2, 9.285] # in meters

def apply_hst(this_data, fs):
    if len(this_data) > dim_mult:
        this_data = [this_data]
    for s, dat in zip(range(len(this_data)), this_data):
        tm = len(dat)/fs
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
    if not len(this_data) == 1:
        this_sig_spec = np.power(np.abs(this_sig_spec), mean_type)
        this_sig_spec = np.power(np.sum(this_sig_spec, axis=0)/len(this_data), 1/mean_type)
    freqs = 2*freqs[freqs < 1501]
    this_sig_spec = this_sig_spec[:len(freqs)]
    return freqs, this_sig_spec

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