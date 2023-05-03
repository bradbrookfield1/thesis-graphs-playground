from matplotlib import pyplot as plt
from constants import *
from tqdm import tqdm
import numpy as np
import librosa, os

def main():
    # os.chdir('Detection')
    
    time_data_s, sr_s = librosa.load(file_names[5], sr=big_sr)
    time_data_s_mono = librosa.to_mono(time_data_s)
    
    time_data_n, sr_n = librosa.load(file_names[12], sr=big_sr, mono=False)
    time_data_n_mono = librosa.to_mono(time_data_n)
    
    time_data_ns, sr_ns = librosa.load(file_names[3], sr=big_sr, mono=False)
    time_data_ns_mono = librosa.to_mono(time_data_ns)
    
    num_harm = 5
    
    filtered_sig = time_data_s
    
    filtered_data = np.empty(shape=(4, len(time_data_ns[0])))
    filtered_data[0] = time_data_ns[0]
    filtered_data[1] = time_data_ns[1]
    filtered_data[2] = time_data_ns[2]
    filtered_data[3] = time_data_ns[3]
    
    filtered_noise = np.empty(shape=(4, len(time_data_n[0])))
    filtered_noise[0] = time_data_n[0]
    filtered_noise[1] = time_data_n[1]
    filtered_noise[2] = time_data_n[2]
    filtered_noise[3] = time_data_n[3]
    
    filtered_data_mono = librosa.to_mono(filtered_data)
    filtered_noise_mono = librosa.to_mono(filtered_noise)
    filtered_sig_mono = librosa.to_mono(filtered_sig)
    
    spec_ns = librosa.stft(filtered_data)   # 4
    spec_n = librosa.stft(filtered_noise)   # 4
    spec_s = librosa.stft(filtered_sig)     # 1
    
    fft_freqs = librosa.fft_frequencies(sr=big_sr)
    fd_hst = apply_lib_hst('Noisy Signal', spec_ns, num_harm, fft_freqs)
    n_hst = apply_lib_hst('Noise', spec_n, num_harm, fft_freqs)
    s_hst = apply_lib_hst('Signal', spec_s, num_harm, fft_freqs)
    
    plt.plot(fft_freqs[1:], fd_hst, label='H(S + N)', lw=1, alpha=0.5)
    plt.plot(fft_freqs[1:], s_hst, label='H(S)', lw=1, alpha=0.5)
    plt.plot(fft_freqs[1:], n_hst, label='H(N)', lw=1, alpha=0.5)
    plt.title('Harmonic Spectral Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def apply_lib_hst(name, lib_multi_spec, num_harm, fft_freqs, mean_harm=1, mean_sig=1, mean_win=1):
    fft_freqs = fft_freqs[fft_freqs < (big_sr/(4*num_harm))]
    big_shape = lib_multi_spec.shape
    lib_multi_spec = np.abs(lib_multi_spec)
    lib_spec_hst, med_shape = hst_harmonics(lib_multi_spec, big_shape, num_harm, fft_freqs, mean_harm)
    if len(big_shape) == 3:
        lib_spec_hst, med_shape = hst_signals(lib_spec_hst, med_shape, fft_freqs, mean_sig)
    librosa.display.specshow(
        lib_spec_hst,
        sr=big_sr/num_harm,
        cmap='turbo',
        x_axis='time',
        y_axis='linear'
    )
    plt.colorbar()
    plt.title(name + ' HST Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    lib_single_hst, _ = hst_windows(lib_spec_hst, med_shape, fft_freqs, mean_win)
    return lib_single_hst[1:]
    
def hst_harmonics(lib_multi_spec, big_shape, num_harm, fft_freqs, mean_harm):
    if len(big_shape) == 3:
        # lib_spec_hst = np.empty(shape=(big_shape[0], np.int64(big_shape[1]/(2*num_harm)), big_shape[2]), dtype=np.float64)
        lib_spec_hst = np.empty(shape=(big_shape[0], big_shape[1], big_shape[2]), dtype=np.float64)
        lib_multi_spec = np.flip(lib_multi_spec, 1)
        for i in tqdm(range(len(lib_multi_spec))):
            for t in range(len(lib_multi_spec[i][1])):
                for f in range(len(fft_freqs)):
                    if f == 0:
                        lib_spec_hst[i][f][t] = lib_multi_spec[i][f][t]
                    else:
                        mag_harm = np.empty(shape=num_harm, dtype=np.float64)
                        for r in range(1, num_harm + 1):
                            f_idx = np.where(fft_freqs == fft_freqs[f]*r)[0]
                            if f_idx.size > 0:
                                mag_harm[r - 1] = lib_multi_spec[i][f_idx[0]][t]
                        lib_spec_hst[i][f][t] = np.power(np.sum(np.power(np.abs(mag_harm), mean_harm))/num_harm, 1/mean_harm)
        lib_spec_hst = np.flip(lib_spec_hst, 1)
    else:
        # lib_spec_hst = np.empty(shape=(big_shape[0]/(2*num_harm), big_shape[1]), dtype=np.float64)
        lib_spec_hst = np.empty(shape=(big_shape[0], big_shape[1]), dtype=np.float64)
        lib_multi_spec = np.flip(lib_multi_spec, 0)
        for t in tqdm(range(len(lib_multi_spec[1]))):
            for f in range(len(fft_freqs)):
                if f == 0:
                    lib_spec_hst[f][t] = lib_multi_spec[f][t]
                else:
                    mag_harm = np.empty(shape=num_harm, dtype=np.float64)
                    for r in range(1, num_harm + 1):
                        f_idx = np.where(fft_freqs == fft_freqs[f]*r)[0]
                        if f_idx.size > 0:
                            mag_harm[r - 1] = lib_multi_spec[f_idx[0]][t]
                    lib_spec_hst[f][t] = np.power(np.sum(np.power(np.abs(mag_harm), mean_harm))/num_harm, 1/mean_harm)
        lib_spec_hst = np.flip(lib_spec_hst, 0)
    return lib_spec_hst, lib_spec_hst.shape
    
def hst_signals(lib_multi_spec_hst, big_shape, fft_freqs, mean_sig):
    lib_single_spec_hst = np.empty(shape=(big_shape[1], big_shape[2]), dtype=np.float64)
    for f in tqdm(range(len(fft_freqs))):
        for t in range(len(lib_multi_spec_hst[0][0])):
            mag_sig = np.empty(shape=big_shape[0], dtype=np.float64)
            for i in range(big_shape[0]):
                mag_sig[i] = lib_multi_spec_hst[i][f][t]
            lib_single_spec_hst[f][t] = np.power(np.sum(np.power(np.abs(mag_sig), mean_sig))/big_shape[0], 1/mean_sig)
    return lib_single_spec_hst, lib_single_spec_hst.shape
    
def hst_windows(lib_single_spec_hst, med_shape, fft_freqs, mean_win):
    lib_single_hst = np.empty(shape=med_shape[0], dtype=np.float64)
    for f in tqdm(range(len(fft_freqs))):
        mag_win = np.empty(shape=med_shape[1], dtype=np.float64)
        for t in range(med_shape[1]):
            mag_win[t] = lib_single_spec_hst[f][t]
        lib_single_hst[f] = np.power(np.sum(np.power(np.abs(mag_win), mean_win))/med_shape[1], 1/mean_win)
    return lib_single_hst, lib_single_hst.shape

if __name__ == '__main__':
    main()