from scipy import signal
from matplotlib import pyplot as plt
from constants import *
from acoustics import Signal
import numpy as np
import pandas as pd
import librosa, os

def main():
    # os.chdir('Characterization')
    
    s_list, gt_list = get_sigs(s_LS_files_2, gt_files)
    get_mounts_response(s_list, gt_list[1], nc_mounts)
    # wd_list = retrieve_files(n_20_files)
    # get_mounts_os(wd_list, all_mounts)
    # sig_list = retrieve_signals(s_LS_files)
    # get_mounts_phase(sig_list, all_mounts)
    # wind_list = retrieve_signals(n_20_files)
    # get_mounts_phase(wind_list, all_mounts)

def get_mounts_phase(data_1, mounts):
    plt.figure(2, figsize=(12, 3)).clf()
    for l1, mt in zip(data_1, mounts):
        t = np.arange(0, 1/50, 1/big_sr)
        inst_phase = l1.instantaneous_phase()
        inst_phase = inst_phase[0:int(big_sr/50)]
        plt.plot(t, inst_phase, label=mt, lw=0.5, alpha=0.75)
    plt.title('Microphone Mount Instantaneous Phase (1st tenth of a second)')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_mounts_os(data_1, mounts):
    for l1, mt in zip(data_1, mounts):
        onset_strength = librosa.onset.onset_strength(y=l1, sr=big_sr)
        times = librosa.times_like(onset_strength, sr=big_sr)
        os = pd.Series(onset_strength).rolling(window, center=True).mean().to_numpy()
        plt.plot(times, os, label=mt, lw=0.5, alpha=0.75)
    plt.title('Microphone Mount Onset Strength')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Strength')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def get_mounts_response(data_1, data_2, mounts):
    for l1, mt in zip(data_1, mounts):
        freq, db_rolled = get_response_arrays(l1, data_2)
        low_count = len(freq[freq < 2500])
        freq = freq[freq < 2500]
        db_rolled = db_rolled[0:low_count]
        plt.plot(freq, db_rolled, label=mt, lw=1, alpha=0.75)
    plt.title('Microphone Mount Spectra')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude Response (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_response_arrays(data_1, data_2):
    data_1_freq, data_1_data = signal.welch(x=data_1, fs=big_sr, nperseg=32768, average='mean')
    data_2_freq, data_2_data = signal.welch(x=data_2, fs=big_sr, nperseg=32768, average='mean')
    
    data_1_roll = pd.Series(np.sqrt(data_1_data)).rolling(window, center=True).mean().to_numpy()
    data_2_roll = pd.Series(np.sqrt(data_2_data)).rolling(window, center=True).mean().to_numpy()
    
    norm_resp = data_1_roll/data_2_roll
    db_rolled = 20*np.log10(pd.Series(norm_resp).rolling(window, center=True).mean().to_numpy())
    
    return data_1_freq, db_rolled

def get_sigs(s_files, gt_files, mono=True):
    return retrieve_files(s_files, mono), retrieve_files(gt_files, mono)

def retrieve_signals(file_list, mono=True):
    s_list = []
    if file_list:
        for f in file_list:
            dat, _ = librosa.load(f, sr=big_sr, mono=mono)
            dat = Signal(dat, fs=big_sr)
            s_list.append(dat)
    return s_list

def retrieve_files(file_list, mono=True):
    s_list = []
    if file_list:
        for f in file_list:
            dat, _ = librosa.load(f, sr=big_sr, mono=mono)
            s_list.append(dat)
    return s_list

if __name__ == '__main__':
    main()