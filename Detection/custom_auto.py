from scipy import signal, fft
from matplotlib import pyplot as plt
from obspy.signal.util import next_pow_2
from constants import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa, copy, os

def main():
    # os.chdir('Detection')
    col_df = pd.DataFrame(columns=['File', 'S/N Test', 'Filter', 'Filt S', 'Norm', 'S Mean', 'S Std', 'S Freq', 'S Max', 'N Mean', 'N Std', 'N Freq', 'N Max', 'NS Mean', 'NS Std', 'NS Freq', 'NS Max', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE', 'Hypothesis'])
    # state_df = pd.DataFrame([['S', 'Wiener White', 'NS', 'Both']], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    state_df = pd.DataFrame([['S', 'Wiener White', 'NS', 'Both'] for _ in range(len(fl_files))], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    big_df = pd.concat([state_df, col_df])
    
    err_type = ['MAE', 'RMSE', 'MSE']
    std_mult = [1, 2, 3, 4, 5]
    # fil_nam = [[17, 12, 9, 22], [17, 22, 20, 23], [2, 12, 0, 27], [5, 12, 3, 13], [8, 13, 6, 12]]
    # fil_nam = [[17, 22, 20, 23]]
    # nam = ['D-17-12-9-22-E', 'DF', '50', '250', '1000']
    
    for s in tqdm(std_mult):
        for err in err_type:
            for f, i in zip(tqdm(fl_files), range(len(fl_files))):
                big_df.at[i, 'File'] = f
                big_df = run_me(big_df, i, s, err, f)
            big_df.to_csv('All-Flight-bruh-' + str(err) + str(s) + '.csv')

def run_me(df, i, std_mult, err='RMSE', fil_nam=None):
    # time_data_s, sr_s = librosa.load(file_names[fil_nam[0]], sr=big_sr, mono=False)
    # time_data_n, sr_n = librosa.load(file_names[fil_nam[1]], sr=big_sr, mono=False)
    # time_data_ns, sr_ns = librosa.load(file_names[fil_nam[2]], sr=big_sr, mono=False)
    # time_data_ns_n, sr_ns_n = librosa.load(file_names[fil_nam[3]], sr=big_sr, mono=False)
    
    time_data_ns_temp, sr_ns = librosa.load(fil_nam, sr=big_sr, mono=False)
    time_data_s_temp, sr_s = librosa.load(file_names[17], sr=big_sr, mono=False)
    time_data_n_temp, sr_n = librosa.load(file_names[22], sr=big_sr, mono=False)
    
    if time_data_ns_temp.shape[1] == time_data_s_temp.shape[1] and time_data_ns_temp.shape[1] == time_data_n_temp.shape[1] and time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
    elif time_data_ns_temp.shape[1] == time_data_s_temp.shape[1] and time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
        time_data_n = time_data_n_temp
    elif time_data_ns_temp.shape[1] == time_data_n_temp.shape[1] and time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
        time_data_s = time_data_s_temp
    elif time_data_s_temp.shape[1] == time_data_n_temp.shape[1] and time_data_s_temp.shape[1] % 2 == 1:
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
        time_data_ns = time_data_ns_temp
    elif time_data_ns_temp.shape[1] % 2 == 1:
        time_data_ns = np.empty(shape=(time_data_ns_temp.shape[0], time_data_ns_temp.shape[1] - 1))
        for j in range(4):
            time_data_ns[j] = time_data_ns_temp[j][:time_data_ns.shape[1]]
        time_data_s = time_data_s_temp
        time_data_n = time_data_n_temp
    elif time_data_s_temp.shape[1] % 2 == 1:
        time_data_s = np.empty(shape=(time_data_s_temp.shape[0], time_data_s_temp.shape[1] - 1))
        for j in range(4):
            time_data_s[j] = time_data_s_temp[j][:time_data_s.shape[1]]
        time_data_ns = time_data_ns_temp
        time_data_n = time_data_n_temp
    elif time_data_n_temp.shape[1] % 2 == 1:
        time_data_n = np.empty(shape=(time_data_n_temp.shape[0], time_data_n_temp.shape[1] - 1))
        for j in range(4):
            time_data_n[j] = time_data_n_temp[j][:time_data_n.shape[1]]
        time_data_ns = time_data_ns_temp
        time_data_s = time_data_s_temp
    else:
        time_data_ns = time_data_ns_temp
        time_data_s = time_data_s_temp
        time_data_n = time_data_n_temp
    
    time_data_s_temp = time_data_s
    time_data_n_temp = time_data_n
    
    time_data_s = np.empty(shape=time_data_ns.shape)
    time_data_n = np.empty(shape=time_data_ns.shape)
    for j in range(4):
        time_data_s[j] = time_data_s_temp[j][0:time_data_ns.shape[1]]
        time_data_n[j] = time_data_n_temp[j][0:time_data_ns.shape[1]]
    
    time_data_s_norm = np.empty(shape=time_data_s.shape)
    time_data_n_norm = np.empty(shape=time_data_n.shape)
    time_data_ns_norm = np.empty(shape=time_data_ns.shape)
    # time_data_ns_n_norm = np.empty(shape=time_data_ns_n.shape)
    for j in range(4):
        time_data_s_norm[j] = (time_data_s[j] - np.mean(time_data_s[j]))/np.std(time_data_s[j])
        time_data_n_norm[j] = (time_data_n[j] - np.mean(time_data_n[j]))/np.std(time_data_n[j])
        time_data_ns_norm[j] = (time_data_ns[j] - np.mean(time_data_ns[j]))/np.std(time_data_ns[j])
        # time_data_ns_n_norm[j] = (time_data_ns_n[j] - np.mean(time_data_ns_n[j]))/np.std(time_data_ns_n[j])
    
    time_data_s_mono = librosa.to_mono(time_data_s)
    time_data_n_mono = librosa.to_mono(time_data_n)
    time_data_ns_mono = librosa.to_mono(time_data_ns)
    # time_data_ns_n_mono = librosa.to_mono(time_data_ns_n)
    
    time_data_s_norm_mono = librosa.to_mono(time_data_s_norm)
    time_data_n_norm_mono = librosa.to_mono(time_data_n_norm)
    time_data_ns_norm_mono = librosa.to_mono(time_data_ns_norm)
    # time_data_ns_n_norm_mono = librosa.to_mono(time_data_ns_n_norm)
    
    # for i in tqdm(range(df.shape[0])):
    if df.at[i, 'Norm'] == 'None' or df.at[i, 'Norm'] == 'Late':
        filtered_noise = time_data_n_mono
        filtered_sig = time_data_s_mono
        if df.at[i, 'S/N Test'] == 'S':
            temp = wiener_filt(filtered_sig, time_data_ns[0], filtered_noise)
            filtered_data = np.empty(shape=(time_data_ns.shape[0], len(temp)))
            for k in range(4):
                filtered_data[k] = wiener_filt(filtered_sig, time_data_ns[k], filtered_noise)
        # else:
        #     temp = wiener_filt(filtered_sig, time_data_ns_n[0], filtered_noise)
        #     filtered_data = np.empty(shape=(time_data_ns_n.shape[0], len(temp)))
        #     for k in range(4):
        #         filtered_data[k] = wiener_filt(filtered_sig, time_data_ns_n[k], filtered_noise)
    else:
        filtered_noise = time_data_n_norm_mono
        filtered_sig = time_data_s_norm_mono
        if df.at[i, 'S/N Test'] == 'S':
            temp = wiener_filt(filtered_sig, time_data_ns_norm[0], filtered_noise)
            filtered_data = np.empty(shape=(time_data_ns_norm.shape[0], len(temp)))
            for k in range(4):
                filtered_data[k] = wiener_filt(filtered_sig, time_data_ns_norm[k], filtered_noise)
        # else:
        #     temp = wiener_filt(filtered_sig, time_data_ns_n_norm[0], filtered_noise)
        #     filtered_data = np.empty(shape=(time_data_ns_n_norm.shape[0], len(temp)))
        #     for k in range(4):
        #         filtered_data[k] = wiener_filt(filtered_sig, time_data_ns_n_norm[k], filtered_noise)

    filtered_noise_2 = matched_filter_freq(filtered_sig, filtered_noise, filtered_noise)
    filtered_sig_2 = matched_filter_freq(filtered_sig, filtered_sig, filtered_noise)
    
    temp = matched_filter_freq(filtered_sig, filtered_data[0], filtered_noise)
    filtered_data_2 = np.empty(shape=(filtered_data.shape[0], len(temp)))
    for k in range(4):
        filtered_data_2[k] = matched_filter_freq(filtered_sig, filtered_data[k], filtered_noise)
    
    filtered_data_mono = librosa.to_mono(filtered_data_2)
    filtered_noise_mono = librosa.to_mono(filtered_noise_2)
    filtered_sig_mono = librosa.to_mono(filtered_sig_2)
    
    # sub_freq, sub_psd = signal.welch(x=filtered_data_mono, fs=sr_ns, nperseg=32768, average='mean')
    # n_freq, n_psd = signal.welch(x=filtered_noise_mono, fs=sr_n, nperseg=32768, average='mean')
    # s_freq, s_psd = signal.welch(x=filtered_sig_mono, fs=sr_s, nperseg=32768, average='mean')
    
    # low_count = len(sub_freq[sub_freq < mat_filt_welch[0]])
    # high_count = len(sub_freq[sub_freq > mat_filt_welch[1]])
    # sub_freq = sub_freq[sub_freq >= mat_filt_welch[0]]
    # sub_freq = sub_freq[sub_freq <= mat_filt_welch[1]]
    # sub_psd = sub_psd[low_count:-high_count]
    
    # low_count = len(n_freq[n_freq < mat_filt_welch[0]])
    # high_count = len(n_freq[n_freq > mat_filt_welch[1]])
    # n_freq = n_freq[n_freq >= mat_filt_welch[0]]
    # n_freq = n_freq[n_freq <= mat_filt_welch[1]]
    # n_psd = n_psd[low_count:-high_count]
    
    # low_count = len(s_freq[s_freq < mat_filt_welch[0]])
    # high_count = len(s_freq[s_freq > mat_filt_welch[1]])
    # s_freq = s_freq[s_freq >= mat_filt_welch[0]]
    # s_freq = s_freq[s_freq <= mat_filt_welch[1]]
    # s_psd = s_psd[low_count:-high_count]
    
    # sub_psd = 10*np.log10(sub_psd/(10**(-12)))
    # n_psd = 10*np.log10(n_psd/(10**(-12)))
    # s_psd = 10*np.log10(s_psd/(10**(-12)))
    
    # plt.plot(sub_freq, sub_psd, label='H(S + N)', lw=1, alpha=0.5)
    # plt.plot(n_freq, n_psd, label='H(N)', lw=1, alpha=0.5)
    # plt.plot(s_freq, s_psd, label='H(S)', lw=1, alpha=0.5)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    fd_freqs, fd_hst = apply_hst(filtered_data_2, sr_ns)
    n_freqs, n_hst = apply_hst(filtered_noise_mono, sr_n)
    s_freqs, s_hst = apply_hst(filtered_sig_mono, sr_s)
    
    if df.at[i, 'Norm'] == 'Late' or df.at[i, 'Norm'] == 'Both':
        fd_hst = np.true_divide(fd_hst - np.mean(fd_hst), np.std(fd_hst))
        n_hst = np.true_divide(n_hst - np.mean(n_hst), np.std(n_hst))
        s_hst = np.true_divide(s_hst - np.mean(s_hst), np.std(s_hst))
    
    fd_freqs_new = np.linspace(np.min(fd_freqs), np.max(fd_freqs), ds, endpoint=True)
    fd_hst_new = np.interp(fd_freqs_new, fd_freqs, fd_hst)
    s_freqs_new = np.linspace(np.min(s_freqs), np.max(s_freqs), ds, endpoint=True)
    s_hst_new = np.interp(s_freqs_new, s_freqs, s_hst)
    n_freqs_new = np.linspace(np.min(n_freqs), np.max(n_freqs), ds, endpoint=True)
    n_hst_new = np.interp(n_freqs_new, n_freqs, n_hst)
    
    # plt.plot(fd_freqs_new, fd_hst_new, label='H(S + N)', lw=1, alpha=0.5)
    # plt.plot(s_freqs_new, s_hst_new, label='H(S)', lw=1, alpha=0.5)
    # plt.plot(n_freqs_new, n_hst_new, label='H(N)', lw=1, alpha=0.5)
    # plt.title('Harmonic Spectral Transform')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    df.at[i, 'NS Mean'] = np.mean(fd_hst_new)
    df.at[i, 'NS Std'] = np.std(fd_hst_new)
    df.at[i, 'NS Max'] = np.max(fd_hst_new)
    df.at[i, 'NS Freq'] = fd_freqs_new[np.where(fd_hst_new == df.at[i, 'NS Max'])[0][0]]
    
    df.at[i, 'S Mean'] = np.mean(s_hst_new)
    df.at[i, 'S Std'] = np.std(s_hst_new)
    df.at[i, 'S Max'] = np.max(s_hst_new)
    df.at[i, 'S Freq'] = s_freqs_new[np.where(s_hst_new == df.at[i, 'S Max'])[0][0]]
    
    df.at[i, 'N Mean'] = np.mean(n_hst_new)
    df.at[i, 'N Std'] = np.std(n_hst_new)
    df.at[i, 'N Max'] = np.max(n_hst_new)
    df.at[i, 'N Freq'] = n_freqs_new[np.where(n_hst_new == df.at[i, 'N Max'])[0][0]]
    
    # Error Before
    # s_e = s_hst_new - fd_hst_new
    # n_e = n_hst_new - fd_hst_new
    # Error 1, 2, 3, 4 - std_mult = 1, 2, 3, 4
    s_e = s_hst_new[fd_hst_new > std_mult*df.at[i, 'NS Std']] - fd_hst_new[fd_hst_new > std_mult*df.at[i, 'NS Std']]
    n_e = n_hst_new[fd_hst_new > std_mult*df.at[i, 'NS Std']] - fd_hst_new[fd_hst_new > std_mult*df.at[i, 'NS Std']]
    s_ae = np.abs(s_e)
    n_ae = np.abs(n_e)
    df.at[i, 'S MAE'] = np.mean(s_ae)
    df.at[i, 'N MAE'] = np.mean(n_ae)
    s_se = np.square(s_e)
    n_se = np.square(n_e)
    df.at[i, 'S MSE'] = np.mean(s_se)
    df.at[i, 'N MSE'] = np.mean(n_se)
    df.at[i, 'S RMSE'] = np.sqrt(df.at[i, 'S MSE'])
    df.at[i, 'N RMSE'] = np.sqrt(df.at[i, 'N MSE'])
    df.at[i, 'Hypothesis'] = 'Threat' if df.at[i, 'S ' + err] < df.at[i, 'N ' + err] else 'No Threat'
    return df

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

def apply_hst(this_data, fs):
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