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
    df5 = pd.read_csv('All-Flight5.csv')
    df3 = pd.read_csv('All-Flight3.csv')
    df5 = df5[['File', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE', 'Hypothesis']].drop([i for i in range(45, len(df5))])
    df3 = df3[['File', 'S MAE', 'N MAE', 'S RMSE', 'N RMSE', 'S MSE', 'N MSE', 'Hypothesis']].drop([i for i in range(45)])
    for i, r in df5.iterrows():
        if '10_-2.wav' in df5.at[i, 'File'] or '10_2.wav' in df5.at[i, 'File']:
            df5.at[i, 'Dist'] = np.sqrt(np.square(310) + np.square(40))
        elif '10_-1.wav' in df5.at[i, 'File'] or '10_1.wav' in df5.at[i, 'File']:
            df5.at[i, 'Dist'] = np.sqrt(np.square(210) + np.square(40))
        else:
            df5.at[i, 'Dist'] = 40
    for i, r in df3.iterrows():
        if '5_-4.wav' in df3.at[i, 'File'] or '5_4.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(310) + np.square(40))
        elif '5_-3.wav' in df3.at[i, 'File'] or '5_3.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(290) + np.square(40))
        elif '5_-2.wav' in df3.at[i, 'File'] or '5_2.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(210) + np.square(40))
        elif '5_-1.wav' in df3.at[i, 'File'] or '5_1.wav' in df3.at[i, 'File'] or '1_-5.wav' in df3.at[i, 'File'] or '1_5.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(100) + np.square(40))
        elif '1_-9.wav' in df3.at[i, 'File'] or '1_9.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(190) + np.square(40))
        elif '1_-8.wav' in df3.at[i, 'File'] or '1_8.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(160) + np.square(40))
        elif '1_-7.wav' in df3.at[i, 'File'] or '1_7.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(140) + np.square(40))
        elif '1_-6.wav' in df3.at[i, 'File'] or '1_6.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(120) + np.square(40))
        elif '1_-4.wav' in df3.at[i, 'File'] or '1_4.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(80) + np.square(40))
        elif '1_-3.wav' in df3.at[i, 'File'] or '1_3.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(60) + np.square(40))
        elif '1_-2.wav' in df3.at[i, 'File'] or '1_2.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(40) + np.square(40))
        elif '1_-1.wav' in df3.at[i, 'File'] or '1_1.wav' in df3.at[i, 'File']:
            df3.at[i, 'Dist'] = np.sqrt(np.square(20) + np.square(40))
        else:
            df3.at[i, 'Dist'] = 40
    new_df = pd.concat([df5, df3]).sort_values(by=['Dist'])
    deg = 2
    s_mae = np.poly1d(np.polyfit(new_df['Dist'], new_df['S MAE'], deg))
    n_mae = np.poly1d(np.polyfit(new_df['Dist'], new_df['N MAE'], deg))
    s_rmse = np.poly1d(np.polyfit(new_df['Dist'], new_df['S RMSE'], deg))
    n_rmse = np.poly1d(np.polyfit(new_df['Dist'], new_df['N RMSE'], deg))
    s_mse = np.poly1d(np.polyfit(new_df['Dist'], new_df['S MSE'], deg))
    n_mse = np.poly1d(np.polyfit(new_df['Dist'], new_df['N MSE'], deg))
    
    mae_sgn = True
    rmse_sgn = True
    mse_sgn = True
    mae_crosses = []
    rmse_crosses = []
    mse_crosses = []
    for d in new_df['Dist']:
        if mae_sgn:
            if s_mae(d) <= n_mae(d):
                mae_crosses.append(d)
                mae_sgn = False
        else:
            if s_mae(d) >= n_mae(d):
                mae_crosses.append(d)
                mae_sgn = True
        if rmse_sgn:
            if s_rmse(d) <= n_rmse(d):
                rmse_crosses.append(d)
                rmse_sgn = False
        else:
            if s_rmse(d) >= n_rmse(d):
                rmse_crosses.append(d)
                rmse_sgn = True
        if mse_sgn:
            if s_mse(d) <= n_mse(d):
                mse_crosses.append(d)
                mse_sgn = False
        else:
            if s_mse(d) >= n_mse(d):
                mse_crosses.append(d)
                mse_sgn = True
    
    # plt.scatter(new_df['Dist'], new_df['S MAE'], label='PS MAE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['N MAE'], label='PN MAE', marker='o', alpha=0.5)
    plt.scatter(new_df['Dist'], new_df['S RMSE'], label='PS RMSE', marker='o', alpha=0.5)
    plt.scatter(new_df['Dist'], new_df['N RMSE'], label='PN RMSE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['S MSE'], label='PS MSE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['N MSE'], label='PN MSE', marker='o', alpha=0.5)
    # plt.plot(new_df['Dist'], s_mae(new_df['Dist']), label='S MAE', alpha=0.75)
    # plt.plot(new_df['Dist'], n_mae(new_df['Dist']), label='N MAE', alpha=0.75)
    plt.plot(new_df['Dist'], s_rmse(new_df['Dist']), label='S RMSE', alpha=0.75)
    plt.plot(new_df['Dist'], n_rmse(new_df['Dist']), label='N RMSE', alpha=0.75)
    # plt.plot(new_df['Dist'], s_mse(new_df['Dist']), label='S MSE', alpha=0.75)
    # plt.plot(new_df['Dist'], n_mse(new_df['Dist']), label='N MSE', alpha=0.75)
    plt.title('RMSE Trend with Distance')
    plt.xlabel('Distance (m)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # plt.scatter(new_df['Dist'], new_df['S MAE'], label='PS MAE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['N MAE'], label='PN MAE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['S RMSE'], label='PS RMSE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['N RMSE'], label='PN RMSE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['S MSE'], label='PS MSE', marker='o', alpha=0.5)
    # plt.scatter(new_df['Dist'], new_df['N MSE'], label='PN MSE', marker='o', alpha=0.5)
    plt.plot(new_df['Dist'], s_mae(new_df['Dist']), label='S MAE', alpha=0.75)
    plt.plot(new_df['Dist'], n_mae(new_df['Dist']), label='N MAE', alpha=0.75)
    plt.plot(new_df['Dist'], s_rmse(new_df['Dist']), label='S RMSE', alpha=0.75)
    plt.plot(new_df['Dist'], n_rmse(new_df['Dist']), label='N RMSE', alpha=0.75)
    plt.plot(new_df['Dist'], s_mse(new_df['Dist']), label='S MSE', alpha=0.75)
    plt.plot(new_df['Dist'], n_mse(new_df['Dist']), label='N MSE', alpha=0.75)
    plt.scatter(mae_crosses, s_mae(mae_crosses), marker='x', alpha=0.75)
    plt.scatter(rmse_crosses, s_rmse(rmse_crosses), marker='x', alpha=0.75)
    plt.scatter(mse_crosses, s_mse(mse_crosses), marker='x', alpha=0.75)
    plt.title('All Error Trends Compared')
    plt.xlabel('Distance (m)')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()