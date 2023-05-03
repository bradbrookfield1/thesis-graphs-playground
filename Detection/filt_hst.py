from scipy import signal, fft
from matplotlib import pyplot as plt
from obspy.signal.util import next_pow_2
from constants import *
import numpy as np
import pandas as pd
import librosa, copy, pywt

def main():
    time_data_s, sr_s = librosa.load(file_names[17], sr=big_sr, mono=False)
    time_data_n, sr_n = librosa.load(file_names[12], sr=big_sr, mono=False)
    time_data_ns, sr_ns = librosa.load(file_names[9], sr=big_sr, mono=False)
    
    for i in range(4):
        time_data_s[i] = (time_data_s[i] - np.mean(time_data_s[i]))/np.std(time_data_s[i])
        time_data_n[i] = (time_data_n[i] - np.mean(time_data_n[i]))/np.std(time_data_n[i])
        time_data_ns[i] = (time_data_ns[i] - np.mean(time_data_ns[i]))/np.std(time_data_ns[i])
    
    # for i in range(4):
    #     time_data_s = butter_band_filter(time_data_s, low_high, sr_s, order)
    #     time_data_n[i] = butter_band_filter(time_data_n[i], low_high, sr_n, order)
    #     time_data_ns[i] = butter_band_filter(time_data_ns[i], low_high, sr_ns, order)
    
    time_data_s_mono = librosa.to_mono(time_data_s)
    time_data_n_mono = librosa.to_mono(time_data_n)
    time_data_ns_mono = librosa.to_mono(time_data_ns)
    
    temp = wiener_filt(time_data_s_mono, time_data_ns[0], time_data_n_mono)
    filtered_data = np.empty(shape=(4, len(temp)))
    for i in range(4):
        filtered_data[i] = wiener_filt(time_data_s_mono, time_data_ns[i], time_data_n_mono)
    
    # filtered_data = time_data_ns
    
    filtered_sig = time_data_s
    
    # filtered_data = np.empty(shape=(4, len(time_data_ns[0])))
    # filtered_data[0] = time_data_ns[0]
    # filtered_data[1] = time_data_ns[1]
    # filtered_data[2] = time_data_ns[2]
    # filtered_data[3] = time_data_ns[3]
    
    filtered_noise = np.empty(shape=(4, len(time_data_n[0])))
    filtered_noise[0] = time_data_n[0]
    filtered_noise[1] = time_data_n[1]
    filtered_noise[2] = time_data_n[2]
    filtered_noise[3] = time_data_n[3]
    
    filtered_data_mono = librosa.to_mono(filtered_data)
    filtered_noise_mono = librosa.to_mono(filtered_noise)
    filtered_sig_mono = librosa.to_mono(filtered_sig)
    
    
    
    # filtered_noise_mono_2 = matched_filter_freq(filtered_sig_mono, filtered_noise_mono, filtered_noise_mono)
    # filtered_sig_mono_2 = matched_filter_freq(filtered_sig_mono, filtered_sig_mono, filtered_noise_mono)
    
    # temp = matched_filter_freq(filtered_sig_mono, filtered_data[0], filtered_noise_mono)
    # filtered_data_2 = np.empty(shape=(filtered_data.shape[0], len(temp)))
    # for k in range(4):
    #     filtered_data_2[k] = matched_filter_freq(filtered_sig_mono, filtered_data[k], filtered_noise_mono)
    
    # filtered_data_mono_2 = librosa.to_mono(filtered_data_2)
    
    filtered_data_mono_2 = filtered_data_mono
    filtered_noise_mono_2 = filtered_noise_mono
    filtered_sig_mono_2 = filtered_sig_mono
    filtered_data_2 = filtered_data
    
    sub_freq, sub_psd = signal.welch(x=filtered_data_mono_2, fs=sr_ns, nperseg=32768, average='mean')
    n_freq, n_psd = signal.welch(x=filtered_noise_mono_2, fs=sr_n, nperseg=32768, average='mean')
    s_freq, s_psd = signal.welch(x=filtered_sig_mono_2, fs=sr_s, nperseg=32768, average='mean')
    
    low_count = len(sub_freq[sub_freq < mat_filt_welch[0]])
    high_count = len(sub_freq[sub_freq > mat_filt_welch[1]])
    sub_freq = sub_freq[sub_freq >= mat_filt_welch[0]]
    sub_freq = sub_freq[sub_freq <= mat_filt_welch[1]]
    sub_psd = sub_psd[low_count:-high_count]
    
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
    
    sub_psd = 10*np.log10(sub_psd/(10**(-12)))
    n_psd = 10*np.log10(n_psd/(10**(-12)))
    s_psd = 10*np.log10(s_psd/(10**(-12)))
    
    plt.plot(sub_freq, sub_psd, label='H(S + N)', lw=1, alpha=0.5)
    plt.plot(n_freq, n_psd, label='H(N)', lw=1, alpha=0.5)
    plt.plot(s_freq, s_psd, label='H(S)', lw=1, alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # filtered_data[0] = butter_band_filter(filtered_data[0], special_filt, sr_ns, special_order, 'bandstop')
    # filtered_data[1] = butter_band_filter(filtered_data[1], special_filt, sr_ns, special_order, 'bandstop')
    # filtered_data[2] = butter_band_filter(filtered_data[2], special_filt, sr_ns, special_order, 'bandstop')
    # filtered_data[3] = butter_band_filter(filtered_data[3], special_filt, sr_ns, special_order, 'bandstop')
    
    fd_freqs, fd_hst = apply_hst(filtered_data_2, sr_ns)
    n_freqs, n_hst = apply_hst(filtered_noise_mono_2, sr_n)
    s_freqs, s_hst = apply_hst(filtered_sig_mono_2, sr_s)
    
    fd_hst = np.true_divide(fd_hst - np.mean(fd_hst), np.std(fd_hst))
    n_hst = np.true_divide(n_hst - np.mean(n_hst), np.std(n_hst))
    s_hst = np.true_divide(s_hst - np.mean(s_hst), np.std(s_hst))
    
    # fd_hst = (fd_hst - np.min(fd_hst))/(np.max(fd_hst) - np.min(fd_hst))
    # n_hst = (n_hst - np.min(n_hst))/(np.max(n_hst) - np.min(n_hst))
    # s_hst = (s_hst - np.min(s_hst))/(np.max(s_hst) - np.min(s_hst))
    
    # fd_hst = 20*np.log10(fd_hst/1)
    # n_hst = 20*np.log10(n_hst/1)
    # s_hst = 20*np.log10(s_hst/1)
    
    # downsamp = np.int32(np.floor(len(fd_hst)/hst_downsamp))
    downsamp = ds
    
    # mx = np.round((np.max(fd_freqs) + np.max(s_freqs) + np.max(n_freqs))/3)
    # mn = np.round((np.min(fd_freqs) + np.min(s_freqs) + np.min(n_freqs))/3)
    fd_freqs_new = np.linspace(np.min(fd_freqs), np.max(fd_freqs), downsamp, endpoint=True)
    fd_hst_new = np.interp(fd_freqs_new, fd_freqs, fd_hst)
    s_freqs_new = np.linspace(np.min(s_freqs), np.max(s_freqs), downsamp, endpoint=True)
    s_hst_new = np.interp(s_freqs_new, s_freqs, s_hst)
    n_freqs_new = np.linspace(np.min(n_freqs), np.max(n_freqs), downsamp, endpoint=True)
    n_hst_new = np.interp(n_freqs_new, n_freqs, n_hst)
    
    # fd_freqs_new = fd_freqs
    # fd_hst_new = fd_hst
    # s_freqs_new = s_freqs
    # s_hst_new = s_hst
    # n_freqs_new = n_freqs
    # n_hst_new = n_hst
    
    # fd_hst_new = 20*np.log10(fd_hst_new/fd_hst_new)
    # n_hst_new = 20*np.log10(n_hst_new/fd_hst_new)
    # s_hst_new = 20*np.log10(s_hst_new/fd_hst_new)
    
    plt.plot(fd_freqs_new, fd_hst_new, label='H(S + N)', lw=1, alpha=0.5)
    plt.plot(s_freqs_new, s_hst_new, label='H(S)', lw=1, alpha=0.5)
    plt.plot(n_freqs_new, n_hst_new, label='H(N)', lw=1, alpha=0.5)
    plt.title('Harmonic Spectral Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # std_mult = 1
    # s_e = s_hst_new - fd_hst_new
    # n_e = n_hst_new - fd_hst_new
    s_e = s_hst_new[fd_hst_new > std_mult*np.std(fd_hst_new)] - fd_hst_new[fd_hst_new > std_mult*np.std(fd_hst_new)]
    n_e = n_hst_new[fd_hst_new > std_mult*np.std(fd_hst_new)] - fd_hst_new[fd_hst_new > std_mult*np.std(fd_hst_new)]
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
    
    # plt.plot(fd_freqs_new, s_ae, label='Signal AE', lw=1, alpha=0.5)
    # plt.plot(fd_freqs_new, n_ae, label='Noise AE', lw=1, alpha=0.5)
    # plt.plot(fd_freqs_new, s_se, label='Signal SE', lw=1, alpha=0.5)
    # plt.plot(fd_freqs_new, n_se, label='Noise SE', lw=1, alpha=0.5)
    # plt.title('Errors')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Error Deviation')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
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
    
    # plt.hist(s_mse, bins=2, label='S MSE', lw=1, alpha=0.5)
    # plt.hist(n_mse, bins=2, label='N MSE', lw=1, alpha=0.5)
    # plt.hist(s_rmse, bins=2, label='S RMSE', lw=1, alpha=0.5)
    # plt.hist(n_rmse, bins=2, label='N RMSE', lw=1, alpha=0.5)
    # plt.xlabel('Prediction')
    # plt.ylabel('Counts')
    
    # orig_s_corr = signal.correlate(time_data_ns_mono, time_data_s_mono, mode='same')
    # orig_n_corr = signal.correlate(time_data_ns_mono, time_data_n_mono, mode='same')
    # match_s_corr = signal.correlate(filtered_data_mono, filtered_sig_mono, mode='same')
    # match_n_corr = signal.correlate(filtered_data_mono, filtered_noise_mono, mode='same')
    # s_hst_corr = signal.correlate(fd_hst_new, s_hst_new, mode='same')
    # n_hst_corr = signal.correlate(fd_hst_new, n_hst_new, mode='same')
    
    # plt.plot(orig_s_corr, label='Signal', lw=1, alpha=0.5)
    # plt.plot(orig_n_corr, label='Noise', lw=1, alpha=0.5)
    # plt.title('Original Cross-Correlation')
    # plt.xlabel('Samples')
    # plt.ylabel('Cross-Correlation')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # plt.plot(match_s_corr, label='Signal', lw=1, alpha=0.5)
    # plt.plot(match_n_corr, label='Noise', lw=1, alpha=0.5)
    # plt.title('Matched Cross-Correlation')
    # plt.xlabel('Samples')
    # plt.ylabel('Cross-Correlation')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # plt.plot(fd_freqs_new, s_hst_corr, label='Signal', lw=1, alpha=0.5)
    # plt.plot(fd_freqs_new, n_hst_corr, label='Noise', lw=1, alpha=0.5)
    # plt.title('HST Cross-Correlation')
    # plt.xlabel('Samples')
    # plt.ylabel('Cross-Correlation')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
def apply_hst(this_data, fs, sig=False):
    if len(this_data) > 4:
        this_data = [this_data]
    for s, dat in zip(range(len(this_data)), this_data):
        tm = len(dat)/fs
        pow_of_2 = np.int64(np.floor(np.log2(len(dat))))
        this_s = signal.resample(dat, np.power(2, pow_of_2))
        new_sr = len(this_s)/tm
        for d in range(1, num_harmonics + 1):
            for n in range(1, num_harmonics + 1):
                if n == 1 and d == 1:
                    this_s_spec = fft_vectorized(this_s, (n/d))
                else:
                    this_s_spec = np.vstack((this_s_spec, fft_vectorized(this_s, (n/d))))
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

# def apply_hst_win(this_data, fs, sig=False):
#     if len(this_data) > 4:
#         incr = np.int32(len(this_data)/wndw)
#         big_sig_list = [[this_data[0:incr]], [this_data[incr:2*incr]], [this_data[2*incr:3*incr]], [this_data[3*incr:4*incr]]]
#                         # [this_data[np.int32(4*len(this_data)/5):np.int32(5*len(this_data)/5)]]]
#     else:
#         big_sig_list = []
#         for s in range(len(this_data)):
#             incr = np.int32(len(this_data[s])/wndw)
#             sig_list = [this_data[s][:incr], this_data[s][incr:2*incr], this_data[s][2*incr:3*incr],
#                         this_data[s][3*incr:4*incr]]
#                         # this_data[s][np.int32(4*len(this_data[s])/5):np.int32(5*len(this_data[s])/5)]]
#             big_sig_list.append(sig_list)
#     big_sig_list = np.transpose(np.array(big_sig_list), (1, 0, 2))
    
#     for w, big_dat in zip(range(len(big_sig_list)), big_sig_list):
#         for s, dat in zip(range(len(big_dat)), big_dat):
#             tm = len(dat)/fs
#             pow_of_2 = np.int64(np.floor(np.log2(len(dat))))
#             this_s = signal.resample(dat, np.power(2, pow_of_2))
#             new_sr = len(this_s)/tm
#             for r in range(1, num_harmonics + 1):
#                 if r == 1:
#                     this_s_spec = fft_vectorized(this_s, r)
#                 else:
#                     this_s_spec = np.vstack((this_s_spec, fft_vectorized(this_s, r)))
#             this_s_spec = np.power(np.abs(this_s_spec), mean_type)
#             this_s_spec = np.power(np.sum(this_s_spec, axis=0)/num_harmonics, 1/mean_type)
#             freqs = librosa.fft_frequencies(sr=new_sr, n_fft=len(this_s))[1:]
#             if s == 0:
#                 this_sig_spec = this_s_spec
#             else:
#                 this_sig_spec = np.vstack((this_sig_spec, this_s_spec))
#         if not len(big_dat) == 1:
#             this_sig_spec = np.power(np.abs(this_sig_spec), mean_type)
#             this_sig_spec = np.power(np.sum(this_sig_spec, axis=0)/len(big_dat), 1/mean_type)
#         if w == 0:
#             this_big_sig_spec = this_sig_spec
#         else:
#             this_big_sig_spec = np.vstack((this_big_sig_spec, this_sig_spec))
#     if not len(big_sig_list) == 1:
#         this_big_sig_spec = np.power(np.abs(this_big_sig_spec), mean_type)
#         this_big_sig_spec = np.power(np.sum(this_big_sig_spec, axis=0)/len(big_sig_list), 1/mean_type)
    
#     freqs = freqs[freqs <= 1500]
#     # if sig:
#     #     freqs = freqs*2
#     this_big_sig_spec = this_big_sig_spec[:len(freqs)]
#     return freqs, this_big_sig_spec

# def wavelet_xfm(sig, lvl=3, wavlt='db8', md='smooth'):
#     # ca3, cd3, cd2, cd1
#     coeffs = pywt.wavedec(sig, wavlt, level=lvl, mode=md)
#     ns_length = [len(coeffs[i]) for i in range(len(coeffs))]
#     mad = [np.median(np.abs(coeffs[i]))/0.6745 for i in range(len(coeffs))]
#     thresh = np.empty(shape=len(coeffs))
#     for i in range(len(coeffs)):
#         thresh[i] = mad[i]*np.sqrt(2*np.log10(ns_length[i]))
#     for i in range(len(coeffs)):
#         for j in range(len(coeffs[i])):
#             coeffs[i][j] = 0 if coeffs[i][j] < thresh[i] else coeffs[i][j]*(np.abs(coeffs[i][j]) - thresh[i])
#     return pywt.waverec(coeffs, wavlt, mode=md)

# def wiener_filt(sig, noisy_sig, noise):
#     s_per = np.mean(np.square(np.abs(librosa.stft(sig))), axis=1)
#     n_per = np.mean(np.square(np.abs(librosa.stft(noise))), axis=1)
#     opt_filt = np.true_divide(s_per, (s_per + n_per))
#     ns_spec = librosa.stft(noisy_sig)
#     ns_spec = np.transpose(ns_spec)
#     for j in range(len(ns_spec)):
#         ns_spec[j] = np.multiply(ns_spec[j], opt_filt)
#     ns_spec = ns_spec.transpose()
#     return librosa.istft(ns_spec)

# def matched_filter(sig_time_data, ns_time_data, N=None):
#     ns_time_data = spec_white(ns_time_data, N)
#     matched_filt = np.conj(sig_time_data[::-1])
#     return signal.convolve(ns_time_data, matched_filt)

# def butter_band(low_high, fs, order, filt_type='bandpass'):
#     nyq = 0.5 * fs
#     low = low_high[0] / nyq
#     high = low_high[1] / nyq
#     b, a = signal.butter(order, [low, high], btype=filt_type)
#     return b, a

# def butter_band_filter(data, low_high, fs, order, filt_type='bandpass'):
#     b, a = butter_band(low_high, fs, order, filt_type)
#     y = signal.filtfilt(b, a, data)
#     # y[~np.isfinite(y)] = 0.0
#     y = y.astype(np.float32)
#     return y

# def spec_white(data, N=None):
#     n = len(data)
#     data_2 = copy.deepcopy(data)[:n]
#     nfft = next_pow_2(n)
#     spec = fft.fft(data_2, nfft)
#     spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
#     if not N == None:
#         shift = N // 2
#         spec = spec[shift:-shift]
#         spec_ampl = pd.Series(spec_ampl).rolling(N, center=True).mean().to_numpy()[shift:-shift]
#     spec = np.true_divide(spec, spec_ampl)
#     return np.real(fft.ifft(spec, nfft))

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

def wiener_filt(sig, noisy_sig, noise):
    s_per = np.mean(np.square(np.abs(librosa.stft(sig))), axis=1)
    n_per = np.mean(np.square(np.abs(librosa.stft(noise))), axis=1)
    opt_filt = np.true_divide(s_per, (s_per + n_per))
    ns_spec = librosa.stft(noisy_sig)
    ns_spec = np.transpose(ns_spec)
    for j in range(len(ns_spec)):
        ns_spec[j] = np.multiply(ns_spec[j], opt_filt)
    ns_spec = ns_spec.transpose()
    return librosa.istft(ns_spec)

def matched_filter_time(sig_time_data, ns_time_data, N=None):
    ns_time_data = spec_white(ns_time_data, N)
    # matched_filt = np.conj(sig_time_data[::-1])
    matched_filt = sig_time_data[::-1]
    return signal.convolve(ns_time_data, matched_filt)

def matched_filter_freq(sig_time_data, ns_time_data, n_time_data, colored=False, N=None):
    if colored:
        _, n_ps = signal.welch(n_time_data, fs=big_sr, nfft=2048, average='mean')
        n_ps = np.flip(n_ps)
    else:
        ns_time_data = spec_white(ns_time_data, N)
    ns_fft = librosa.stft(ns_time_data)
    s_fft = np.conjugate(librosa.stft(sig_time_data))
    if colored:
        s_fft = np.transpose(np.divide(np.transpose(s_fft), n_ps))
    fd_fft = np.multiply(ns_fft, s_fft)
    return librosa.istft(fd_fft)

def spec_white(data, N=None):
    n = len(data)
    data_2 = copy.deepcopy(data)[:n]
    nfft = next_pow_2(n)
    spec = fft.fft(data_2, n)
    spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
    if not N == None:
        shift = N // 2
        spec = spec[shift:-shift]
        spec_ampl = pd.Series(spec_ampl).rolling(N, center=True).mean().to_numpy()[shift:-shift]
    spec = np.true_divide(spec, spec_ampl)
    return np.real(fft.ifft(spec, n))

if __name__ == '__main__':
    main()