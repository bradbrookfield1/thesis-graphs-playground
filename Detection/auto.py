from scipy import signal, fft
from matplotlib import pyplot as plt
from obspy.signal.util import next_pow_2
from constants import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa, copy

def main():
    col_df = pd.DataFrame(columns=['S/N Test', 'Filter', 'Filt S', 'Norm', 'S Mean', 'S Std', 'S Freq', 'S Max', 'N Mean', 'N Std', 'N Freq', 'N Max', 'NS Mean', 'NS Std', 'NS Freq', 'NS Max', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE', 'Hypothesis'])
    # state_df = pd.DataFrame([['S', 'None', 'None', 'None'], ['S', 'None', 'None', 'Early'], ['S', 'None', 'None', 'Late'], ['S', 'None', 'None', 'Both'],
    #                          ['N', 'None', 'None', 'None'], ['N', 'None', 'None', 'Early'], ['N', 'None', 'None', 'Late'], ['N', 'None', 'None', 'Both'],
    #                          ['S', 'Wiener', 'NS', 'None'], ['S', 'Wiener', 'NS', 'Early'], ['S', 'Wiener', 'NS', 'Late'], ['S', 'Wiener', 'NS', 'Both'],
    #                          ['N', 'Wiener', 'NS', 'None'], ['N', 'Wiener', 'NS', 'Early'], ['N', 'Wiener', 'NS', 'Late'], ['N', 'Wiener', 'NS', 'Both'],
    #                          ['S', 'Wiener', 'N', 'None'], ['S', 'Wiener', 'N', 'Early'], ['S', 'Wiener', 'N', 'Late'], ['S', 'Wiener', 'N', 'Both'],
    #                          ['N', 'Wiener', 'N', 'None'], ['N', 'Wiener', 'N', 'Early'], ['N', 'Wiener', 'N', 'Late'], ['N', 'Wiener', 'N', 'Both'],
    #                          ['S', 'Wiener', 'S', 'None'], ['S', 'Wiener', 'S', 'Early'], ['S', 'Wiener', 'S', 'Late'], ['S', 'Wiener', 'S', 'Both'],
    #                          ['N', 'Wiener', 'S', 'None'], ['N', 'Wiener', 'S', 'Early'], ['N', 'Wiener', 'S', 'Late'], ['N', 'Wiener', 'S', 'Both'],
    #                          ['S', 'Colored', 'NS', 'None'], ['S', 'Colored', 'NS', 'Early'], ['S', 'Colored', 'NS', 'Late'], ['S', 'Colored', 'NS', 'Both'],
    #                          ['N', 'Colored', 'NS', 'None'], ['N', 'Colored', 'NS', 'Early'], ['N', 'Colored', 'NS', 'Late'], ['N', 'Colored', 'NS', 'Both'],
    #                          ['S', 'Colored', 'N', 'None'], ['S', 'Colored', 'N', 'Early'], ['S', 'Colored', 'N', 'Late'], ['S', 'Colored', 'N', 'Both'],
    #                          ['N', 'Colored', 'N', 'None'], ['N', 'Colored', 'N', 'Early'], ['N', 'Colored', 'N', 'Late'], ['N', 'Colored', 'N', 'Both'],
    #                          ['S', 'Colored', 'S', 'None'], ['S', 'Colored', 'S', 'Early'], ['S', 'Colored', 'S', 'Late'], ['S', 'Colored', 'S', 'Both'],
    #                          ['N', 'Colored', 'S', 'None'], ['N', 'Colored', 'S', 'Early'], ['N', 'Colored', 'S', 'Late'], ['N', 'Colored', 'S', 'Both'],
    #                          ['S', 'White', 'NS', 'None'], ['S', 'White', 'NS', 'Early'], ['S', 'White', 'NS', 'Late'], ['S', 'White', 'NS', 'Both'],
    #                          ['N', 'White', 'NS', 'None'], ['N', 'White', 'NS', 'Early'], ['N', 'White', 'NS', 'Late'], ['N', 'White', 'NS', 'Both'],
    #                          ['S', 'White', 'N', 'None'], ['S', 'White', 'N', 'Early'], ['S', 'White', 'N', 'Late'], ['S', 'White', 'N', 'Both'],
    #                          ['N', 'White', 'N', 'None'], ['N', 'White', 'N', 'Early'], ['N', 'White', 'N', 'Late'], ['N', 'White', 'N', 'Both'],
    #                          ['S', 'White', 'S', 'None'], ['S', 'White', 'S', 'Early'], ['S', 'White', 'S', 'Late'], ['S', 'White', 'S', 'Both'],
    #                          ['N', 'White', 'S', 'None'], ['N', 'White', 'S', 'Early'], ['N', 'White', 'S', 'Late'], ['N', 'White', 'S', 'Both'],
    #                          ['S', 'Time', 'NS', 'None'], ['S', 'Time', 'NS', 'Early'], ['S', 'Time', 'NS', 'Late'], ['S', 'Time', 'NS', 'Both'],
    #                          ['N', 'Time', 'NS', 'None'], ['N', 'Time', 'NS', 'Early'], ['N', 'Time', 'NS', 'Late'], ['N', 'Time', 'NS', 'Both'],
    #                          ['S', 'Time', 'N', 'None'], ['S', 'Time', 'N', 'Early'], ['S', 'Time', 'N', 'Late'], ['S', 'Time', 'N', 'Both'],
    #                          ['N', 'Time', 'N', 'None'], ['N', 'Time', 'N', 'Early'], ['N', 'Time', 'N', 'Late'], ['N', 'Time', 'N', 'Both'],
    #                          ['S', 'Time', 'S', 'None'], ['S', 'Time', 'S', 'Early'], ['S', 'Time', 'S', 'Late'], ['S', 'Time', 'S', 'Both'],
    #                          ['N', 'Time', 'S', 'None'], ['N', 'Time', 'S', 'Early'], ['N', 'Time', 'S', 'Late'], ['N', 'Time', 'S', 'Both']], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    state_df = pd.DataFrame([['S', 'Wiener', 'NS', 'None'], ['S', 'Wiener', 'NS', 'Early'], ['S', 'Wiener', 'NS', 'Late'], ['S', 'Wiener', 'NS', 'Both'],
                             ['N', 'Wiener', 'NS', 'None'], ['N', 'Wiener', 'NS', 'Early'], ['N', 'Wiener', 'NS', 'Late'], ['N', 'Wiener', 'NS', 'Both'],
                             ['S', 'White', 'S', 'None'], ['S', 'White', 'S', 'Early'], ['S', 'White', 'S', 'Late'], ['S', 'White', 'S', 'Both'],
                             ['N', 'White', 'S', 'None'], ['N', 'White', 'S', 'Early'], ['N', 'White', 'S', 'Late'], ['N', 'White', 'S', 'Both']], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    wiener_df = pd.DataFrame([['S', 'Wiener', 'NS', 'None'], ['S', 'Wiener', 'NS', 'Early'], ['S', 'Wiener', 'NS', 'Late'], ['S', 'Wiener', 'NS', 'Both'],
                             ['N', 'Wiener', 'NS', 'None'], ['N', 'Wiener', 'NS', 'Early'], ['N', 'Wiener', 'NS', 'Late'], ['N', 'Wiener', 'NS', 'Both']], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    white_df = pd.DataFrame([['S', 'White', 'S', 'None'], ['S', 'White', 'S', 'Early'], ['S', 'White', 'S', 'Late'], ['S', 'White', 'S', 'Both'],
                             ['N', 'White', 'S', 'None'], ['N', 'White', 'S', 'Early'], ['N', 'White', 'S', 'Late'], ['N', 'White', 'S', 'Both']], columns=['S/N Test', 'Filter', 'Filt S', 'Norm'])
    big_df = pd.concat([state_df, col_df])
    wi_df = pd.concat([wiener_df, col_df])
    wht_df = pd.concat([white_df, col_df])
    
    # df = run_me(big_df)
    # df.to_csv('dataframe.csv')
    
    err_type = ['MAE', 'RMSE', 'MSE']
    std_mult = [1, 4]
    fil_nam = [[17, 22, 20, 23], [17, 12, 9, 22], [2, 12, 0, 27], [5, 12, 3, 13], [8, 13, 6, 12]]
    nam = ['DF', 'D', '50', '250', '1000']
    for f, n in zip(fil_nam, nam):
        for err in err_type:
            for sm in tqdm(std_mult):
                df = run_me(wi_df, err, sm, f) if sm == 1 else run_me(wht_df, err, sm, f)
                df.to_csv(n + '-' + err + str(sm) + '-2.csv')
                # df.to_csv('D-17-12-9-22-E' + str(sm) + '-' + err + '.csv')

def run_me(df, err='RMSE', std_mult=None, fil_nam=None):
    # time_data_s, sr_s = librosa.load(file_names[17], sr=big_sr, mono=False)
    # time_data_n, sr_n = librosa.load(file_names[12], sr=big_sr, mono=False)
    # time_data_ns, sr_ns = librosa.load(file_names[9], sr=big_sr, mono=False)
    # time_data_ns_n, sr_ns_n = librosa.load(file_names[22], sr=big_sr, mono=False)
    
    time_data_s, sr_s = librosa.load(file_names[fil_nam[0]], sr=big_sr, mono=False)
    time_data_n, sr_n = librosa.load(file_names[fil_nam[1]], sr=big_sr, mono=False)
    time_data_ns, sr_ns = librosa.load(file_names[fil_nam[2]], sr=big_sr, mono=False)
    time_data_ns_n, sr_ns_n = librosa.load(file_names[fil_nam[3]], sr=big_sr, mono=False)
    
    time_data_s_norm = np.empty(shape=time_data_s.shape)
    time_data_n_norm = np.empty(shape=time_data_n.shape)
    time_data_ns_norm = np.empty(shape=time_data_ns.shape)
    time_data_ns_n_norm = np.empty(shape=time_data_ns_n.shape)
    for i in range(4):
        time_data_s_norm[i] = (time_data_s[i] - np.mean(time_data_s[i]))/np.std(time_data_s[i])
        time_data_n_norm[i] = (time_data_n[i] - np.mean(time_data_n[i]))/np.std(time_data_n[i])
        time_data_ns_norm[i] = (time_data_ns[i] - np.mean(time_data_ns[i]))/np.std(time_data_ns[i])
        time_data_ns_n_norm[i] = (time_data_ns_n[i] - np.mean(time_data_ns_n[i]))/np.std(time_data_ns_n[i])
    
    time_data_s_mono = librosa.to_mono(time_data_s)
    time_data_n_mono = librosa.to_mono(time_data_n)
    time_data_ns_mono = librosa.to_mono(time_data_ns)
    time_data_ns_n_mono = librosa.to_mono(time_data_ns_n)
    
    time_data_s_norm_mono = librosa.to_mono(time_data_s_norm)
    time_data_n_norm_mono = librosa.to_mono(time_data_n_norm)
    time_data_ns_norm_mono = librosa.to_mono(time_data_ns_norm)
    time_data_ns_n_norm_mono = librosa.to_mono(time_data_ns_n_norm)
    
    for i in tqdm(range(df.shape[0])):
        match df.at[i, 'Filter']:
            case 'None':
                if df.at[i, 'Norm'] == 'None' or df.at[i, 'Norm'] == 'Late':
                    filtered_sig = time_data_s_mono
                    filtered_noise = time_data_n_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        filtered_data = np.empty(shape=time_data_ns.shape)
                        for k in range(4):
                            filtered_data[k] = time_data_ns[k]
                    else:
                        filtered_data = np.empty(shape=time_data_ns_n.shape)
                        for k in range(4):
                            filtered_data[k] = time_data_ns_n[k]
                else:
                    filtered_sig = time_data_s_norm_mono
                    filtered_noise = time_data_n_norm_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        filtered_data = np.empty(shape=time_data_ns_norm.shape)
                        for k in range(4):
                            filtered_data[k] = time_data_ns_norm[k]
                    else:
                        filtered_data = np.empty(shape=time_data_ns_n_norm.shape)
                        for k in range(4):
                            filtered_data[k] = time_data_ns_n_norm[k]
            case 'Wiener':
                if df.at[i, 'Norm'] == 'None' or df.at[i, 'Norm'] == 'Late':
                    filtered_noise = wiener_filt(time_data_s_mono, time_data_n_mono, time_data_n_mono) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_mono
                    filtered_sig = wiener_filt(time_data_s_mono, time_data_s_mono, time_data_n_mono) if df.at[i, 'Filt S'] == 'S' else time_data_s_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = wiener_filt(time_data_s_mono, time_data_ns[0], time_data_n_mono)
                        filtered_data = np.empty(shape=(time_data_ns.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = wiener_filt(time_data_s_mono, time_data_ns[k], time_data_n_mono)
                    else:
                        temp = wiener_filt(time_data_s_mono, time_data_ns_n[0], time_data_n_mono)
                        filtered_data = np.empty(shape=(time_data_ns_n.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = wiener_filt(time_data_s_mono, time_data_ns_n[k], time_data_n_mono)
                else:
                    filtered_noise = wiener_filt(time_data_s_norm_mono, time_data_n_norm_mono, time_data_n_norm_mono) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_norm_mono
                    filtered_sig = wiener_filt(time_data_s_norm_mono, time_data_s_norm_mono, time_data_n_norm_mono) if df.at[i, 'Filt S'] == 'S' else time_data_s_norm_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = wiener_filt(time_data_s_norm_mono, time_data_ns_norm[0], time_data_n_norm_mono)
                        filtered_data = np.empty(shape=(time_data_ns_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = wiener_filt(time_data_s_norm_mono, time_data_ns_norm[k], time_data_n_norm_mono)
                    else:
                        temp = wiener_filt(time_data_s_norm_mono, time_data_ns_n_norm[0], time_data_n_norm_mono)
                        filtered_data = np.empty(shape=(time_data_ns_n_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = wiener_filt(time_data_s_norm_mono, time_data_ns_n_norm[k], time_data_n_norm_mono)
            case 'Colored':
                if df.at[i, 'Norm'] == 'None' or df.at[i, 'Norm'] == 'Late':
                    filtered_noise = matched_filter_freq(time_data_s_mono, time_data_n_mono, time_data_n_mono, True) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_mono
                    filtered_sig = matched_filter_freq(time_data_s_mono, time_data_s_mono, time_data_n_mono, True) if df.at[i, 'Filt S'] == 'S' else time_data_s_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = matched_filter_freq(time_data_s_mono, time_data_ns[0], time_data_n_mono, True)
                        filtered_data = np.empty(shape=(time_data_ns.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_mono, time_data_ns[k], time_data_n_mono, True)
                    else:
                        temp = matched_filter_freq(time_data_s_mono, time_data_ns[0], time_data_n_mono, True)
                        filtered_data = np.empty(shape=(time_data_ns_n.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_mono, time_data_ns_n[k], time_data_n_mono, True)
                else:
                    filtered_noise = matched_filter_freq(time_data_s_norm_mono, time_data_n_norm_mono, time_data_n_norm_mono, True) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_norm_mono
                    filtered_sig = matched_filter_freq(time_data_s_norm_mono, time_data_s_norm_mono, time_data_n_norm_mono, True) if df.at[i, 'Filt S'] == 'S' else time_data_s_norm_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = matched_filter_freq(time_data_s_norm_mono, time_data_ns_norm[0], time_data_n_norm_mono, True)
                        filtered_data = np.empty(shape=(time_data_ns_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_norm_mono, time_data_ns_norm[k], time_data_n_norm_mono, True)
                    else:
                        temp = matched_filter_freq(time_data_s_norm_mono, time_data_ns_norm[0], time_data_n_norm_mono, True)
                        filtered_data = np.empty(shape=(time_data_ns_n_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_norm_mono, time_data_ns_n_norm[k], time_data_n_norm_mono, True)
            case 'White':
                if df.at[i, 'Norm'] == 'None' or df.at[i, 'Norm'] == 'Late':
                    filtered_noise = matched_filter_freq(time_data_s_mono, time_data_n_mono, time_data_n_mono) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_mono
                    filtered_sig = matched_filter_freq(time_data_s_mono, time_data_s_mono, time_data_n_mono) if df.at[i, 'Filt S'] == 'S' else time_data_s_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = matched_filter_freq(time_data_s_mono, time_data_ns[0], time_data_n_mono)
                        filtered_data = np.empty(shape=(time_data_ns.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_mono, time_data_ns[k], time_data_n_mono)
                    else:
                        temp = matched_filter_freq(time_data_s_mono, time_data_ns_n[0], time_data_n_mono)
                        filtered_data = np.empty(shape=(time_data_ns_n.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_mono, time_data_ns_n[k], time_data_n_mono)
                else:
                    filtered_noise = matched_filter_freq(time_data_s_norm_mono, time_data_n_norm_mono, time_data_n_norm_mono) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_norm_mono
                    filtered_sig = matched_filter_freq(time_data_s_norm_mono, time_data_s_norm_mono, time_data_n_norm_mono) if df.at[i, 'Filt S'] == 'S' else time_data_s_norm_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = matched_filter_freq(time_data_s_norm_mono, time_data_ns_norm[0], time_data_n_norm_mono)
                        filtered_data = np.empty(shape=(time_data_ns_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_norm_mono, time_data_ns_norm[k], time_data_n_norm_mono)
                    else:
                        temp = matched_filter_freq(time_data_s_norm_mono, time_data_ns_n_norm[0], time_data_n_norm_mono)
                        filtered_data = np.empty(shape=(time_data_ns_n_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_freq(time_data_s_norm_mono, time_data_ns_n_norm[k], time_data_n_norm_mono)
            case 'Time':
                if df.at[i, 'Norm'] == 'None' or df.at[i, 'Norm'] == 'Late':
                    filtered_noise = matched_filter_time(time_data_s_mono, time_data_n_mono) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_mono
                    filtered_sig = matched_filter_time(time_data_s_mono, time_data_s_mono) if df.at[i, 'Filt S'] == 'S' else time_data_s_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = matched_filter_time(time_data_s_mono, time_data_ns[0])
                        filtered_data = np.empty(shape=(time_data_ns.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_time(time_data_s_mono, time_data_ns[k])
                    else:
                        temp = matched_filter_time(time_data_s_mono, time_data_ns_n[0])
                        filtered_data = np.empty(shape=(time_data_ns_n.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_time(time_data_s_mono, time_data_ns_n[k])
                else:
                    filtered_noise = matched_filter_time(time_data_s_norm_mono, time_data_n_norm_mono) if df.at[i, 'Filt S'] == 'N' or df.at[i, 'Filt S'] == 'S' else time_data_n_norm_mono
                    filtered_sig = matched_filter_time(time_data_s_norm_mono, time_data_s_norm_mono) if df.at[i, 'Filt S'] == 'S' else time_data_s_norm_mono
                    if df.at[i, 'S/N Test'] == 'S':
                        temp = matched_filter_time(time_data_s_norm_mono, time_data_ns_norm[0])
                        filtered_data = np.empty(shape=(time_data_ns_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_time(time_data_s_norm_mono, time_data_ns_norm[k])
                    else:
                        temp = matched_filter_time(time_data_s_norm_mono, time_data_ns_n_norm[0])
                        filtered_data = np.empty(shape=(time_data_ns_n_norm.shape[0], len(temp)))
                        for k in range(4):
                            filtered_data[k] = matched_filter_time(time_data_s_norm_mono, time_data_ns_n_norm[k])
        filtered_data_mono = librosa.to_mono(filtered_data)
        filtered_noise_mono = librosa.to_mono(filtered_noise)
        filtered_sig_mono = librosa.to_mono(filtered_sig)
        
        fd_freqs, fd_hst = apply_hst(filtered_data, sr_ns)
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
        if std_mult == None:
            std_mult = 5
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