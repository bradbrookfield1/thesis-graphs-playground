from scipy import signal
from matplotlib import pyplot as plt
from acoustics import Signal
from constants import *
import numpy as np
import librosa

def take1():
    # time_data_ns, sr_ns = librosa.load(file_names[7], sr=None, mono=False)
    # time_data_n, sr_n = librosa.load(file_names[9], sr=None, mono=False)
    # time_data_n, sr_n = librosa.load(file_names[10], sr=None)
    
    time_data_ns = mult*Signal.from_wav(file_names[7], normalize=False)
    sr_ns = time_data_ns.fs
    time_data_n = mult*Signal.from_wav(file_names[13], normalize=False)
    sr_n = time_data_n.fs
    time_data_n = np.array(librosa.to_mono(time_data_n))
    
    time_data_ns[0] = butter_band_filter(time_data_ns[0], low_high, sr_ns, order)
    time_data_ns[1] = butter_band_filter(time_data_ns[1], low_high, sr_ns, order)
    time_data_ns[2] = butter_band_filter(time_data_ns[2], low_high, sr_ns, order)
    time_data_ns[3] = butter_band_filter(time_data_ns[3], low_high, sr_ns, order)
    time_data_n = butter_band_filter(time_data_n, low_high, sr_n, order)
    
    fft_data_ns = np.empty(shape=(4, 1025, 862))
    fft_data_ns[0] = librosa.stft(time_data_ns[0])
    fft_data_ns[1] = librosa.stft(time_data_ns[1])
    fft_data_ns[2] = librosa.stft(time_data_ns[2])
    fft_data_ns[3] = librosa.stft(time_data_ns[3])
    
    new_mag_sub = np.empty(shape=(4, 1025, 862))
    new_mag_sub[0] = np.sqrt(np.power(np.abs(fft_data_ns[0]), 2) - np.power(np.abs(time_data_n), 2))
    new_mag_sub[1] = np.sqrt(np.power(np.abs(fft_data_ns[1]), 2) - np.power(np.abs(time_data_n), 2))
    new_mag_sub[2] = np.sqrt(np.power(np.abs(fft_data_ns[2]), 2) - np.power(np.abs(time_data_n), 2))
    new_mag_sub[3] = np.sqrt(np.power(np.abs(fft_data_ns[3]), 2) - np.power(np.abs(time_data_n), 2))
    new_mag_sub[~np.isfinite(new_mag_sub)] = 0.0
    
    time_data_sub = np.empty(shape=(4, 440832))
    time_data_sub[0] = librosa.istft(new_mag_sub[0] + np.angle(fft_data_ns[0]))
    time_data_sub[1] = librosa.istft(new_mag_sub[1] + np.angle(fft_data_ns[1]))
    time_data_sub[2] = librosa.istft(new_mag_sub[2] + np.angle(fft_data_ns[2]))
    time_data_sub[3] = librosa.istft(new_mag_sub[3] + np.angle(fft_data_ns[3]))
    
    # fft_data_ns = np.empty(shape=(4, 1025, 862))
    # fft_data_ns[0] = librosa.stft(time_data_ns[0])
    # fft_data_ns[1] = librosa.stft(time_data_ns[1])
    # fft_data_ns[2] = librosa.stft(time_data_ns[2])
    # fft_data_ns[3] = librosa.stft(time_data_ns[3])
    # fft_data_n = librosa.stft(time_data_n)
    
    # fft_mag_ns = np.empty(shape=(4, 1025, 862))
    # fft_mag_ns[0] = np.abs(fft_data_ns[0])
    # fft_mag_ns[1] = np.abs(fft_data_ns[1])
    # fft_mag_ns[2] = np.abs(fft_data_ns[2])
    # fft_mag_ns[3] = np.abs(fft_data_ns[3])
    
    # fft_pow_ns = np.empty(shape=(4, 1025, 862))
    # fft_pow_ns[0] = np.power(fft_mag_ns[0], 2)
    # fft_pow_ns[1] = np.power(fft_mag_ns[1], 2)
    # fft_pow_ns[2] = np.power(fft_mag_ns[2], 2)
    # fft_pow_ns[3] = np.power(fft_mag_ns[3], 2)
    
    # fft_mag_n = np.abs(fft_data_n)
    # fft_pow_n = np.power(fft_mag_n, 2)
    
    # new_pow_sub = np.empty(shape=(4, 1025, 862))
    # new_pow_sub[0] = np.power(np.abs(fft_data_ns[0]), 2) - fft_pow_n
    # new_pow_sub[1] = np.power(np.abs(fft_data_ns[1]), 2) - fft_pow_n
    # new_pow_sub[2] = np.power(np.abs(fft_data_ns[2]), 2) - fft_pow_n
    # new_pow_sub[3] = np.power(np.abs(fft_data_ns[3]), 2) - fft_pow_n
    
    # new_mag_sub = np.empty(shape=(4, 1025, 862))
    # new_mag_sub[0] = np.sqrt(new_pow_sub[0])
    # new_mag_sub[1] = np.sqrt(new_pow_sub[1])
    # new_mag_sub[2] = np.sqrt(new_pow_sub[2])
    # new_mag_sub[3] = np.sqrt(new_pow_sub[3])
    # new_mag_sub[~np.isfinite(new_mag_sub)] = 0.0
    
    # new_fft_sub = np.empty(shape=(4, 1025, 862))
    # new_fft_sub[0] = new_mag_sub[0] + np.angle(fft_data_ns[0])
    # new_fft_sub[1] = new_mag_sub[1] + np.angle(fft_data_ns[1])
    # new_fft_sub[2] = new_mag_sub[2] + np.angle(fft_data_ns[2])
    # new_fft_sub[3] = new_mag_sub[3] + np.angle(fft_data_ns[3])
    
    # time_data_sub = np.empty(shape=(4, 440832))
    # time_data_sub[0] = librosa.istft(new_fft_sub[0])
    # time_data_sub[1] = librosa.istft(new_fft_sub[1])
    # time_data_sub[2] = librosa.istft(new_fft_sub[2])
    # time_data_sub[3] = librosa.istft(new_fft_sub[3])
    
    time_data_ns = np.array(librosa.to_mono(time_data_ns))
    # time_data_sub[~np.isfinite(time_data_sub)] = 0.0
    time_data_sub = np.array(librosa.to_mono(time_data_sub))
    
    plt.hist(time_data_ns, bins=100, density=True, label='PDF Noisy Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.hist(time_data_sub, bins=100, density=True, label='PDF Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    sub_freq, sub_psd = signal.welch(x=time_data_sub, fs=sr_ns, nperseg=32768, average='mean')
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
    
    # sample1 = np.random.normal(loc=20, scale=5, size=1000)
    # sample1 = librosa.istft(librosa.stft(sample1))
    # sample2 = np.random.normal(loc=40, scale=5, size=1000)
    # sample2 = librosa.istft(librosa.stft(sample2))
    # sample = np.hstack((sample1, sample2))
    # plt.hist(sample, bins=100, density=True, label='Both', lw=1, alpha=0.5)
    # plt.hist(sample1, bins=100, density=True, label='20', lw=1, alpha=0.5)
    # plt.hist(sample2, bins=100, density=True, label='40', lw=1, alpha=0.5)
    # plt.xlabel('Signal Value')
    # plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def take2():
    file_names = ['FM_50_W20_C.wav', 'FM_50_W13_C.wav', 'FM_50_C.wav',
                  'FM_250_W20_C.wav', 'FM_250_W13_C.wav', 'FM_250_C.wav',
                  'FM_1000_W20_C.wav', 'FM_1000_W13_C.wav', 'FM_1000_C.wav',
                  'FM_W20_C.wav', 'FM_W13_C.wav', 'FM_C.wav']
    low_high = [50, 3999]
    order = 4
    
    time_data_ns, sr_ns = librosa.load(file_names[7], sr=None)
    time_data_n, sr_n = librosa.load(file_names[9], sr=None)
    
    time_data_ns = butter_band_filter(time_data_ns, low_high, sr_ns, order)
    time_data_n = butter_band_filter(time_data_n, low_high, sr_n, order)
    
    fft_data_ns = librosa.stft(time_data_ns)
    fft_data_n = librosa.stft(time_data_n)
    fft_mag_ns = np.abs(fft_data_ns)
    fft_mag_n = np.abs(fft_data_n)
    fft_pow_ns = np.power(fft_mag_ns, 2)
    fft_pow_n = np.power(fft_mag_n, 2)
    
    new_pow_sub = np.power(np.abs(fft_data_ns), 2) - fft_pow_n
    new_mag_sub = np.sqrt(new_pow_sub)
    new_mag_sub[~np.isfinite(new_mag_sub)] = 0.0
    new_fft_sub = new_mag_sub + np.angle(fft_data_ns)
    time_data_sub = librosa.istft(new_fft_sub)
    
    plt.hist(time_data_ns, bins=100, density=True, label='PDF Noisy Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.hist(time_data_sub, bins=100, density=True, label='PDF Signal', lw=1, alpha=0.5)
    plt.hist(time_data_n, bins=100, density=True, label='PDF Noise', lw=1, alpha=0.5)
    plt.xlabel('Signal Value')
    plt.ylabel('PDF')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    sub_freq, sub_psd = signal.welch(x=time_data_sub, fs=sr_ns, nperseg=32768, average='mean')
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
    take2()