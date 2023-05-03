from scipy import signal
from matplotlib import pyplot as plt
from constants import *
import numpy as np
import librosa

def main():
    time_data, sr = librosa.load(file_names[4], sr=None, mono=False)
    time_data[0] = butter_band_filter(time_data[0], low_high, sr, order)
    time_data[1] = butter_band_filter(time_data[1], low_high, sr, order)
    time_data[2] = butter_band_filter(time_data[2], low_high, sr, order)
    time_data[3] = butter_band_filter(time_data[3], low_high, sr, order)
    freq, coh = signal.coherence(time_data[0], time_data[1], fs=4000, nperseg=512)
    plt.plot(freq, coh)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.grid(True)
    plt.show()

    freq, coh = signal.coherence(time_data[2], time_data[3], fs=4000, nperseg=512)
    plt.plot(freq, coh)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
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

if __name__ == '__main__':
    main()