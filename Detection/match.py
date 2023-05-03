from scipy import signal, fft
from matplotlib import pyplot as plt
from obspy.signal.util import next_pow_2
from constants import *
import numpy as np
import pandas as pd
import librosa, copy

def main():
    time_data_s, sr_s = librosa.load(file_names[5], sr=big_sr, mono=False)
    time_data_n, sr_n = librosa.load(file_names[12], sr=big_sr, mono=False)
    time_data_ns, sr_ns = librosa.load(file_names[3], sr=big_sr, mono=False)
    
    # for i in range(4):
    #     time_data_s[i] = (time_data_s[i] - np.mean(time_data_s[i]))/np.std(time_data_s[i])
    #     time_data_n[i] = (time_data_n[i] - np.mean(time_data_n[i]))/np.std(time_data_n[i])
    #     time_data_ns[i] = (time_data_ns[i] - np.mean(time_data_ns[i]))/np.std(time_data_ns[i])
    
    time_data_s_mono = librosa.to_mono(time_data_s)
    time_data_n_mono = librosa.to_mono(time_data_n)
    time_data_ns_mono = librosa.to_mono(time_data_ns)
    
    # temp = matched_filter_time(time_data_s, time_data_ns[0])
    # filtered_data = np.empty(shape=(4, len(temp)))
    # filtered_data[0] = temp
    # filtered_data[1] = matched_filter_time(time_data_s, time_data_ns[1])
    # filtered_data[2] = matched_filter_time(time_data_s, time_data_ns[2])
    # filtered_data[3] = matched_filter_time(time_data_s, time_data_ns[3])
    
    # temp = matched_filter_time(time_data_s, time_data_n[0])
    # filtered_noise = np.empty(shape=(4, len(temp)))
    # filtered_noise[0] = temp
    # filtered_noise[1] = matched_filter_time(time_data_s, time_data_n[1])
    # filtered_noise[2] = matched_filter_time(time_data_s, time_data_n[2])
    # filtered_noise[3] = matched_filter_time(time_data_s, time_data_n[3])
    
    # filtered_sig = matched_filter_time(time_data_s, time_data_s)
    
    temp = matched_filter_freq(time_data_s_mono, time_data_ns[0], time_data_n_mono)
    filtered_data = np.empty(shape=(4, len(temp)))
    filtered_data[0] = temp
    filtered_data[1] = matched_filter_freq(time_data_s_mono, time_data_ns[1], time_data_n_mono)
    filtered_data[2] = matched_filter_freq(time_data_s_mono, time_data_ns[2], time_data_n_mono)
    filtered_data[3] = matched_filter_freq(time_data_s_mono, time_data_ns[3], time_data_n_mono)
    
    # temp = matched_filter_freq(time_data_s_mono, time_data_n[0], time_data_n_mono)
    # filtered_noise = np.empty(shape=(4, len(temp)))
    # filtered_noise[0] = temp
    # filtered_noise[1] = matched_filter_freq(time_data_s_mono, time_data_n[1], time_data_n_mono)
    # filtered_noise[2] = matched_filter_freq(time_data_s_mono, time_data_n[2], time_data_n_mono)
    # filtered_noise[3] = matched_filter_freq(time_data_s_mono, time_data_n[3], time_data_n_mono)
    
    filtered_noise = matched_filter_freq(time_data_s_mono, time_data_n_mono, time_data_n_mono)
    
    filtered_sig = matched_filter_freq(time_data_s_mono, time_data_s_mono, time_data_n_mono)
    
    # filtered_sig = time_data_s
    
    # filtered_data = np.empty(shape=(4, len(time_data_ns[0])))
    # filtered_data[0] = time_data_ns[0]
    # filtered_data[1] = time_data_ns[1]
    # filtered_data[2] = time_data_ns[2]
    # filtered_data[3] = time_data_ns[3]
    
    # filtered_noise = np.empty(shape=(4, len(time_data_n[0])))
    # filtered_noise[0] = time_data_n[0]
    # filtered_noise[1] = time_data_n[1]
    # filtered_noise[2] = time_data_n[2]
    # filtered_noise[3] = time_data_n[3]
    
    filtered_data_mono = librosa.to_mono(filtered_data)
    filtered_noise_mono = librosa.to_mono(filtered_noise)
    filtered_sig_mono = librosa.to_mono(filtered_sig)
    
    fd_freq, fd_psd = signal.welch(x=filtered_data_mono, fs=sr_ns, nperseg=32768, average='mean')
    n_freq, n_psd = signal.welch(x=filtered_noise_mono, fs=sr_n, nperseg=32768, average='mean')
    s_freq, s_psd = signal.welch(x=filtered_sig_mono, fs=sr_s, nperseg=32768, average='mean')
    
    low_count = len(fd_freq[fd_freq < mat_filt_welch[0]])
    high_count = len(fd_freq[fd_freq > mat_filt_welch[1]])
    fd_freq = fd_freq[fd_freq >= mat_filt_welch[0]]
    fd_freq = fd_freq[fd_freq <= mat_filt_welch[1]]
    fd_psd = fd_psd[low_count:-high_count]
    
    low_count = len(n_freq[n_freq < mat_filt_welch[0]])
    high_count = len(n_freq[n_freq > mat_filt_welch[1]])
    n_freq = n_freq[n_freq >= mat_filt_welch[0]]
    n_freq = n_freq[n_freq <= mat_filt_welch[1]]
    n_psd = n_psd[low_count:-high_count]
    
    low_count = len(s_freq[s_freq < mat_filt_welch[0]])
    high_count = len(s_freq[s_freq > mat_filt_welch[1]])
    s_freq = s_freq[s_freq >= mat_filt_welch[0]]
    s_freq = s_freq[s_freq <= mat_filt_welch[1]]
    s_psd = s_psd[low_count:-high_count]
    
    fd_psd = 10*np.log10(fd_psd/(10**(-12)))
    n_psd = 10*np.log10(n_psd/(10**(-12)))
    s_psd = 10*np.log10(s_psd/(10**(-12)))
    # fd_psd = 10*np.log10(fd_psd*np.power(10, 12))
    # n_psd = 10*np.log10(n_psd*np.power(10, 12))
    # s_psd = 10*np.log10(s_psd*np.power(10, 12))
    
    plt.plot(fd_freq, fd_psd, label='H(S + N)', lw=1, alpha=0.5)
    plt.plot(n_freq, n_psd, label='H(N)', lw=1, alpha=0.5)
    plt.plot(s_freq, s_psd, label='H(S)', lw=1, alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    fd_freqs, fd_hst = apply_hst(filtered_data, sr_ns)
    n_freqs, n_hst = apply_hst(filtered_noise, sr_n)
    s_freqs, s_hst = apply_hst(filtered_sig_mono, sr_s)
    
    # fd_hst = np.true_divide(fd_hst - np.mean(fd_hst), np.std(fd_hst))
    # n_hst = np.true_divide(n_hst - np.mean(n_hst), np.std(n_hst))
    # s_hst = np.true_divide(s_hst - np.mean(s_hst), np.std(s_hst))
    
    # fd_hst = (fd_hst - np.min(fd_hst))/(np.max(fd_hst) - np.min(fd_hst))
    # n_hst = (n_hst - np.min(n_hst))/(np.max(n_hst) - np.min(n_hst))
    # s_hst = (s_hst - np.min(s_hst))/(np.max(s_hst) - np.min(s_hst))
    
    # fd_hst = 20*np.log10(fd_hst/1)
    # n_hst = 20*np.log10(n_hst/1)
    # s_hst = 20*np.log10(s_hst/1)
    
    fd_freqs_new = np.linspace(np.min(fd_freqs), np.max(fd_freqs), ds, endpoint=True)
    fd_hst_new = np.interp(fd_freqs_new, fd_freqs, fd_hst)
    s_freqs_new = np.linspace(np.min(s_freqs), np.max(s_freqs), ds, endpoint=True)
    s_hst_new = np.interp(s_freqs_new, s_freqs, s_hst)
    n_freqs_new = np.linspace(np.min(n_freqs), np.max(n_freqs), ds, endpoint=True)
    n_hst_new = np.interp(n_freqs_new, n_freqs, n_hst)
    
    plt.plot(fd_freqs_new, fd_hst_new, label='H(S + N)', lw=1, alpha=0.5)
    plt.plot(s_freqs_new, s_hst_new, label='H(S)', lw=1, alpha=0.5)
    plt.plot(n_freqs_new, n_hst_new, label='H(N)', lw=1, alpha=0.5)
    plt.title('Harmonic Spectral Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    s_e = s_hst_new - fd_hst_new
    n_e = n_hst_new - fd_hst_new
    s_ae = np.abs(s_e)
    n_ae = np.abs(n_e)
    s_mae = np.mean(s_ae)
    n_mae = np.mean(n_ae)
    s_se = np.square(s_e)
    n_se = np.square(n_e)
    s_mse = np.mean(s_se)
    n_mse = np.mean(n_se)
    s_rmse = np.sqrt(s_mse)
    n_rmse = np.sqrt(n_mse)
    
    print()
    print(s_mse)
    print(n_mse)
    print(s_rmse)
    print(n_rmse)
    print(s_mae)
    print(n_mae)
    print()
    
    det = True if s_rmse < n_rmse else False
    
    print('Prediction: Threat Detected') if det else print('Prediction: No Threat Detected')
    print()

def matched_filter_time(sig_time_data, ns_time_data, N=None):
    ns_time_data = spec_white(ns_time_data, N)
    # matched_filt = np.conj(sig_time_data[::-1])
    matched_filt = sig_time_data[::-1]
    return signal.convolve(ns_time_data, matched_filt)

def matched_filter_freq(sig_time_data, ns_time_data, n_time_data, N=None):
    _, n_ps = signal.welch(n_time_data, fs=big_sr, nfft=2048, average='mean')
    n_ps = np.flip(n_ps)
    # ns_time_data = spec_white(ns_time_data, N)
    ns_fft = librosa.stft(ns_time_data)
    s_fft = np.conjugate(librosa.stft(sig_time_data))
    s_fft = np.transpose(np.divide(np.transpose(s_fft), n_ps))
    fd_fft = np.multiply(ns_fft, s_fft)
    return librosa.istft(fd_fft)

def spec_white(data, N=None):
    n = len(data)
    data_2 = copy.deepcopy(data)[:n]
    print()
    print(len(data_2))
    nfft = next_pow_2(n)
    spec = fft.fft(data_2, n)
    print(len(spec))
    spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
    print(len(spec_ampl))
    if not N == None:
        shift = N // 2
        spec = spec[shift:-shift]
        spec_ampl = pd.Series(spec_ampl).rolling(N, center=True).mean().to_numpy()[shift:-shift]
    spec = np.true_divide(spec, spec_ampl)
    print(len(spec))
    print()
    return np.real(fft.ifft(spec, n))

def apply_hst(this_data, fs):
    if len(this_data) > 4:
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
        freqs = librosa.fft_frequencies(sr=new_sr, n_fft=len(this_s))[1:]
        if s == 0:
            this_sig_spec = this_s_spec
        else:
            this_sig_spec = np.vstack((this_sig_spec, this_s_spec))
    if not len(this_data) == 1:
        this_sig_spec = np.power(np.abs(this_sig_spec), mean_type)
        this_sig_spec = np.power(np.sum(this_sig_spec, axis=0)/len(this_data), 1/mean_type)
    freqs = freqs[freqs <= 1500]
    this_sig_spec = this_sig_spec[:len(freqs)]
    return freqs, this_sig_spec

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