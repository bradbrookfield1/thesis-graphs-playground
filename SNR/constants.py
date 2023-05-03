import pywt
import numpy as np

# Propagation
# speed_noise_spl_dict = {'13': 80, '15': 84, '18': 87.5, '20': 90}   # m/s: dB SPL
# speed_noise_spl_dict = {'13': 89, '15': 92, '18': 97, '20': 99}     # m/s: dB SPL
# spl_speed = speed_noise_spl_dict['13']

# vel_array = [
#     0. Uav air speed,
#     1. Wind speed to adjust speed of sound.
#     2. Wind speed to add to noise.
# ]
# Note: positive is toward the observer, negative is away from the observer.

# Wind Tunnel
wt_vel_array = [13, 1, 1]
albatross_alt = 40

# Angel Flight
angel_air_speed = 25                # m/s
angel_vel_array = [25, 6, 6]        # m/s
angel_p_bar = 101600                # Pascals
angel_p_ref = 101500                # Pascals
angel_rel_hum = 0.45                # Decimal
angel_temp = 14.25                  # Celcius
angel_temp = angel_temp + 273.15    # To Kelvin
angel_alt = 40                      # Meters

# Mic hole
mic_rad_inch = 0.03125
mic_rad_mm = mic_rad_inch*25.4
cs_area = np.pi*np.square(mic_rad_mm/1000)

spl_src = 100
dst_from_src = 1
dir_fact = 4

# freqs = np.linspace(1, 1000000, 100000)                             # Hz
freqs = np.logspace(1, 5, 100000)                                   # Hz

# distance = 2.5                                                      # Meters
distance = 100                                                      # Meters
tunnel_dist = 2.5                                                   # Meters

dist_array_lin = np.linspace(1, 100, 10)                     # Meters
dist_array_log = np.logspace(0, 2, 10)
dist_array_big = np.linspace(1, 320, 100)

# Absorption coefficient only?
rel_hum_array = [i/10 for i in range(11)]                   # Decimal (0, 0.1, 0.2, ..., 1)
# rel_hum_array = np.logspace(0, 2, 5)/100                   # Decimal (0, 0.1, 0.2, ..., 1)
temp_array = [((i*5 - 20) + 273.15) for i in range(9)]      # Kelvin (-20, -15, -10, ..., 20 deg C)
# temp_array = [((i*14 - 20) + 273.15) for i in range(5)]      # Kelvin (-20, -15, -10, ..., 20 deg C)
p_bar_array = np.linspace(101325, 151325, 11)
# freqs = freqs/(p_bar/101325)                                # Normalized by barometric pressure

# Signal Amplitude Multiplier (Spectral Subtraction)
mult = 100000

# SNR
window = 50

# Matched Filter
mat_filt_welch = [4, 3999]
big_sr = 48000

# Harmonic Spectral Transform
num_harmonics = 5
mean_type = 1
special_filt = [455, 465]
# special_filt = [915, 925]
# special_filt = [225, 235]
special_order = 3
hst_downsamp = 10
ds = 1000
wndw = 4

# Wavelet
# print(pywt.wavelist(kind='discrete'))
# haar, sym2, db8
wavlt = 'db8'
md = 'smooth'
lvl = 4

# Butterworth Band Pass Filter
low_high = [4, 3999]
order = 4

# Kernel Functions
# Kernel type list: 'additive_chi2', 'chi2', 'linear', 'poly' or 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'
kernel_type = 'rbf'
dim_mult = 10
gamma = 1.0
degree = 3.0
coeff = 1.0

nc_mounts = ['No Mount', 'Tight Deep (No Cotton)', 'Tight Shallow (No Cotton)', 'Wide Deep (No Cotton)', 'Wide Shallow (No Cotton)', 'Flat (No Cotton)', 'Flat (Cotton)']
all_mounts = ['Flat (No Cotton)', 'Flat (Cotton)', 'Tight Deep (Cotton)', 'Tight Shallow (Cotton)', 'Wide Deep (Cotton)', 'Wide Shallow (Cotton)']
ramp_mounts = ['Flat (Cotton)', 'Flat (No Cotton)', 'High Ramp (No Cotton)', 'Low Ramp (No Cotton)']

flight_names = ['Pass_5_D.wav', 'Noise_5_6.wav', 'D_4_Track_10.wav']

# Extra file lists
file_names = ['FM_50_W20_C.wav', 'FM_50_W13_C.wav', 'FM_50_C.wav',          # 0
              'FM_250_W20_C.wav', 'FM_250_W13_C.wav', 'FM_250_C.wav',       # 3
              'FM_1000_W20_C.wav', 'FM_1000_W13_C.wav', 'FM_1000_C.wav',    # 6
              'FM_D_W20_C.wav', 'FM_D_W13_C.wav', 'FM_D_C.wav',             # 9
              'FM_W20_C.wav', 'FM_W13_C.wav', 'FM_C.wav',                   # 12
              'D_1_Track_30.wav', 'D_1_Track_10.wav', 'D_4_Track_10.wav',   # 15
              'Pass_1_D.wav', 'Pass_2_D.wav', 'Pass_3_D.wav',               # 18
              'Noise_1_2.wav', 'Noise_2_3.wav', 'Noise_3_4.wav',            # 21
              'Pass_4_D.wav', 'Pass_5_D.wav', 'Pass_6_D.wav',               # 24
              'Noise_4_5.wav', 'Noise_5_6.wav', 'Noise_6_7.wav',            # 27
              'Pass_7_D.wav', 'Pass_8_D.wav', 'Pass_9_D.wav',               # 30
              'Noise_7_8.wav', 'Noise_8_9.wav', 'Noise_9_10.wav']           # 33

s_FM_NC_files = ['FM_ES.wav', 'FM_LS.wav', 'FM_D.wav', 'FM_50.wav', 'FM_250.wav', 'FM_1000.wav']
s_FM_files = ['FM_ES_C.wav', 'FM_LS_C.wav', 'FM_D_C.wav', 'FM_50_C.wav', 'FM_250_C.wav', 'FM_1000_C.wav']
s_TD_files = ['TD_ES_C.wav', 'TD_LS_C.wav', 'TD_D_C.wav', 'TD_50_C.wav', 'TD_250_C.wav', 'TD_1000_C.wav']
s_TS_files = ['TS_ES_C.wav', 'TS_LS_C.wav', 'TS_D_C.wav', 'TS_50_C.wav', 'TS_250_C.wav', 'TS_1000_C.wav']
s_WD_files = ['WD_ES_C.wav', 'WD_LS_C.wav', 'WD_D_C.wav', 'WD_50_C.wav', 'WD_250_C.wav', 'WD_1000_C.wav']
s_WS_files = ['WS_ES_C.wav', 'WS_LS_C.wav', 'WS_D_C.wav', 'WS_50_C.wav', 'WS_250_C.wav', 'WS_1000_C.wav']

s_ES_files = ['FM_ES.wav', 'FM_ES_C.wav', 'TD_ES_C.wav', 'TS_ES_C.wav', 'WD_ES_C.wav', 'WS_ES_C.wav']
s_LS_files = ['FM_LS.wav', 'FM_LS_C.wav', 'TD_LS_C.wav', 'TS_LS_C.wav', 'WD_LS_C.wav', 'WS_LS_C.wav']
s_D_files = ['FM_D.wav', 'FM_D_C.wav', 'TD_D_C.wav', 'TS_D_C.wav', 'WD_D_C.wav', 'WS_D_C.wav']
s_50_files = ['FM_50.wav', 'FM_50_C.wav', 'TD_50_C.wav', 'TS_50_C.wav', 'WD_50_C.wav', 'WS_50_C.wav']
s_250_files = ['FM_250.wav', 'FM_250_C.wav', 'TD_250_C.wav', 'TS_250_C.wav', 'WD_250_C.wav', 'WS_250_C.wav']
s_1000_files = ['FM_1000.wav', 'FM_1000_C.wav', 'TD_1000_C.wav', 'TS_1000_C.wav', 'WD_1000_C.wav', 'WS_1000_C.wav']

s_ES_files_2 = ['NM_ES.wav', 'TD_ES.wav', 'TS_ES.wav', 'WD_ES.wav', 'WS_ES.wav', 'FM_ES.wav', 'FM_ES_C.wav']
ns_ES_13_files_2 = ['NM_ES_W13.wav', 'TD_ES_W13.wav', 'TS_ES_W13.wav', 'WD_ES_W13.wav', 'WS_ES_W13.wav', 'FM_ES_W13.wav', 'FM_ES_W13_C.wav']
n_13_files_2 = ['NM_W13.wav', 'TD_W13.wav', 'TS_W13.wav', 'WD_W13.wav', 'WS_W13.wav', 'FM_W13.wav', 'FM_W13_C.wav']
ns_ES_20_files_2 = ['NM_ES_W20.wav', 'TD_ES_W20.wav', 'TS_ES_W20.wav', 'WD_ES_W20.wav', 'WS_ES_W20.wav', 'FM_ES_W20.wav', 'FM_ES_W20_C.wav']
n_20_files_2 = ['NM_W20.wav', 'TD_W20.wav', 'TS_W20.wav', 'WD_W20.wav', 'WS_W20.wav', 'FM_W20.wav', 'FM_W20_C.wav']

ns_FM_ES_files_2 = ['FM_ES_W20.wav', 'FM_ES_W18.wav', 'FM_ES_W15.wav', 'FM_ES_W13.wav']
n_FM_files_2 = ['FM_W20.wav', 'FM_W18.wav', 'FM_W15.wav', 'FM_W13.wav']

s_R_ES_files_2 = ['FM_ES_C.wav', 'FM_ES.wav', 'HR_ES.wav', 'LR_ES.wav']
ns_R_ES_13_files_2 = ['FM_ES_W13_C.wav', 'FM_ES_W13.wav', 'HR_ES_W13.wav', 'LR_ES_W13.wav']
n_R_13_files_2 = ['FM_W13_C.wav', 'FM_W13.wav', 'HR_W13.wav', 'LR_W13.wav']

ns_HR_ES_files = ['HR_ES_W20.wav', 'HR_ES_W18.wav', 'HR_ES_W15.wav', 'HR_ES_W13.wav']
n_HR_files = ['HR_W20.wav', 'HR_W18.wav', 'HR_W15.wav', 'HR_W13.wav']



n_FM_NC_files = ['FM_W20.wav', 'FM_W18.wav', 'FM_W15.wav', 'FM_W13.wav']
n_FM_files = ['FM_W20_C.wav', 'FM_W18_C.wav', 'FM_W15_C.wav', 'FM_W13_C.wav']
n_TD_files = ['TD_W20_C.wav', 'TD_W18_C.wav', 'TD_W15_C.wav', 'TD_W13_C.wav']
n_TS_files = ['TS_W20_C.wav', 'TS_W18_C.wav', 'TS_W15_C.wav', 'TS_W13_C.wav']
n_WD_files = ['WD_W20_C.wav', 'WD_W18_C.wav', 'WD_W15_C.wav', 'WD_W13_C.wav']
n_WS_files = ['WS_W20_C.wav', 'WS_W18_C.wav', 'WS_W15_C.wav', 'WS_W13_C.wav']

n_20_files = ['FM_W20.wav', 'FM_W20_C.wav', 'TD_W20_C.wav', 'TS_W20_C.wav', 'WD_W20_C.wav', 'WS_W20_C.wav']
n_18_files = ['FM_W18.wav', 'FM_W18_C.wav', 'TD_W18_C.wav', 'TS_W18_C.wav', 'WD_W18_C.wav', 'WS_W18_C.wav']
n_15_files = ['FM_W15.wav', 'FM_W15_C.wav', 'TD_W15_C.wav', 'TS_W15_C.wav', 'WD_W15_C.wav', 'WS_W15_C.wav']
n_13_files = ['FM_W13.wav', 'FM_W13_C.wav', 'TD_W13_C.wav', 'TS_W13_C.wav', 'WD_W13_C.wav', 'WS_W13_C.wav']



# ts_files = ['ES.wav', 'LS.wav', 'D.wav', '50.wav', '250.wav', '1000.wav']
ts_files = []







# ns_files = ['FM_ES_W20_C.wav', 'FM_LS_W20_C.wav', 'FM_D_W20_C.wav', 'FM_50_W20_C.wav', 'FM_250_W20_C.wav', 'FM_1000_W20_C.wav']
# ns_files = ['FM_ES_W18_C.wav', 'FM_LS_W18_C.wav', 'FM_D_W18_C.wav', 'FM_50_W18_C.wav', 'FM_250_W18_C.wav', 'FM_1000_W18_C.wav']
# ns_files = ['FM_ES_W15_C.wav', 'FM_LS_W15_C.wav', 'FM_D_W15_C.wav', 'FM_50_W15_C.wav', 'FM_250_W15_C.wav', 'FM_1000_W15_C.wav']
# ns_files = ['FM_ES_W13_C.wav', 'FM_LS_W13_C.wav', 'FM_D_W13_C.wav', 'FM_50_W13_C.wav', 'FM_250_W13_C.wav', 'FM_1000_W13_C.wav']

# ns_files = ['FM_ES_W20.wav', 'FM_LS_W20.wav', 'FM_D_W20.wav', 'FM_50_W20.wav', 'FM_250_W20.wav', 'FM_1000_W20.wav']
# ns_files = ['FM_ES_W18.wav', 'FM_LS_W18.wav', 'FM_D_W18.wav', 'FM_50_W18.wav', 'FM_250_W18.wav', 'FM_1000_W18.wav']
# ns_files = ['FM_ES_W15.wav', 'FM_LS_W15.wav', 'FM_D_W15.wav', 'FM_50_W15.wav', 'FM_250_W15.wav', 'FM_1000_W15.wav']
# ns_files = ['FM_ES_W13.wav', 'FM_LS_W13.wav', 'FM_D_W13.wav', 'FM_50_W13.wav', 'FM_250_W13.wav', 'FM_1000_W13.wav']

# ns_files = ['TD_ES_W20_C.wav', 'TD_LS_W20_C.wav', 'TD_D_W20_C.wav', 'TD_50_W20_C.wav', 'TD_250_W20_C.wav', 'TD_1000_W20_C.wav']
# ns_files = ['TD_ES_W18_C.wav', 'TD_LS_W18_C.wav', 'TD_D_W18_C.wav', 'TD_50_W18_C.wav', 'TD_250_W18_C.wav', 'TD_1000_W18_C.wav']
# ns_files = ['TD_ES_W15_C.wav', 'TD_LS_W15_C.wav', 'TD_D_W15_C.wav', 'TD_50_W15_C.wav', 'TD_250_W15_C.wav', 'TD_1000_W15_C.wav']
# ns_files = ['TD_ES_W13_C.wav', 'TD_LS_W13_C.wav', 'TD_D_W13_C.wav', 'TD_50_W13_C.wav', 'TD_250_W13_C.wav', 'TD_1000_W13_C.wav']

# ns_files = ['TS_ES_W20_C.wav', 'TS_LS_W20_C.wav', 'TS_D_W20_C.wav', 'TS_50_W20_C.wav', 'TS_250_W20_C.wav', 'TS_1000_W20_C.wav']
# ns_files = ['TS_ES_W18_C.wav', 'TS_LS_W18_C.wav', 'TS_D_W18_C.wav', 'TS_50_W18_C.wav', 'TS_250_W18_C.wav', 'TS_1000_W18_C.wav']
# ns_files = ['TS_ES_W15_C.wav', 'TS_LS_W15_C.wav', 'TS_D_W15_C.wav', 'TS_50_W15_C.wav', 'TS_250_W15_C.wav', 'TS_1000_W15_C.wav']
# ns_files = ['TS_ES_W13_C.wav', 'TS_LS_W13_C.wav', 'TS_D_W13_C.wav', 'TS_50_W13_C.wav', 'TS_250_W13_C.wav', 'TS_1000_W13_C.wav']

# ns_files = ['WD_ES_W20_C.wav', 'WD_LS_W20_C.wav', 'WD_D_W20_C.wav', 'WD_50_W20_C.wav', 'WD_250_W20_C.wav', 'WD_1000_W20_C.wav']
# ns_files = ['WD_ES_W18_C.wav', 'WD_LS_W18_C.wav', 'WD_D_W18_C.wav', 'WD_50_W18_C.wav', 'WD_250_W18_C.wav', 'WD_1000_W18_C.wav']
# ns_files = ['WD_ES_W15_C.wav', 'WD_LS_W15_C.wav', 'WD_D_W15_C.wav', 'WD_50_W15_C.wav', 'WD_250_W15_C.wav', 'WD_1000_W15_C.wav']
# ns_files = ['WD_ES_W13_C.wav', 'WD_LS_W13_C.wav', 'WD_D_W13_C.wav', 'WD_50_W13_C.wav', 'WD_250_W13_C.wav', 'WD_1000_W13_C.wav']

# ns_files = ['WS_ES_W20_C.wav', 'WS_LS_W20_C.wav', 'WS_D_W20_C.wav', 'WS_50_W20_C.wav', 'WS_250_W20_C.wav', 'WS_1000_W20_C.wav']
# ns_files = ['WS_ES_W18_C.wav', 'WS_LS_W18_C.wav', 'WS_D_W18_C.wav', 'WS_50_W18_C.wav', 'WS_250_W18_C.wav', 'WS_1000_W18_C.wav']
# ns_files = ['WS_ES_W15_C.wav', 'WS_LS_W15_C.wav', 'WS_D_W15_C.wav', 'WS_50_W15_C.wav', 'WS_250_W15_C.wav', 'WS_1000_W15_C.wav']
# ns_files = ['WS_ES_W13_C.wav', 'WS_LS_W13_C.wav', 'WS_D_W13_C.wav', 'WS_50_W13_C.wav', 'WS_250_W13_C.wav', 'WS_1000_W13_C.wav']







ns_FM_NC_ES_files = ['FM_ES_W20.wav', 'FM_ES_W18.wav', 'FM_ES_W15.wav', 'FM_ES_W13.wav']
ns_FM_ES_files = ['FM_ES_W20_C.wav', 'FM_ES_W18_C.wav', 'FM_ES_W15_C.wav', 'FM_ES_W13_C.wav']
ns_TD_ES_files = ['TD_ES_W20_C.wav', 'TD_ES_W18_C.wav', 'TD_ES_W15_C.wav', 'TD_ES_W13_C.wav']
ns_TS_ES_files = ['TS_ES_W20_C.wav', 'TS_ES_W18_C.wav', 'TS_ES_W15_C.wav', 'TS_ES_W13_C.wav']
ns_WD_ES_files = ['WD_ES_W20_C.wav', 'WD_ES_W18_C.wav', 'WD_ES_W15_C.wav', 'WD_ES_W13_C.wav']
ns_WS_ES_files = ['WS_ES_W20_C.wav', 'WS_ES_W18_C.wav', 'WS_ES_W15_C.wav', 'WS_ES_W13_C.wav']

ns_FM_NC_LS_files = ['FM_LS_W20.wav', 'FM_LS_W18.wav', 'FM_LS_W15.wav', 'FM_LS_W13.wav']
ns_FM_LS_files = ['FM_LS_W20_C.wav', 'FM_LS_W18_C.wav', 'FM_LS_W15_C.wav', 'FM_LS_W13_C.wav']
ns_TD_LS_files = ['TD_LS_W20_C.wav', 'TD_LS_W18_C.wav', 'TD_LS_W15_C.wav', 'TD_LS_W13_C.wav']
ns_TS_LS_files = ['TS_LS_W20_C.wav', 'TS_LS_W18_C.wav', 'TS_LS_W15_C.wav', 'TS_LS_W13_C.wav']
ns_WD_LS_files = ['WD_LS_W20_C.wav', 'WD_LS_W18_C.wav', 'WD_LS_W15_C.wav', 'WD_LS_W13_C.wav']
ns_WS_LS_files = ['WS_LS_W20_C.wav', 'WS_LS_W18_C.wav', 'WS_LS_W15_C.wav', 'WS_LS_W13_C.wav']

# ns_files = ['FM_D_W20.wav', 'FM_D_W18.wav', 'FM_D_W15.wav', 'FM_D_W13.wav']
# ns_files = ['FM_D_W20_C.wav', 'FM_D_W18_C.wav', 'FM_D_W15_C.wav', 'FM_D_W13_C.wav']
# ns_files = ['TD_D_W20_C.wav', 'TD_D_W18_C.wav', 'TD_D_W15_C.wav', 'TD_D_W13_C.wav']
# ns_files = ['TS_D_W20_C.wav', 'TS_D_W18_C.wav', 'TS_D_W15_C.wav', 'TS_D_W13_C.wav']
# ns_files = ['WD_D_W20_C.wav', 'WD_D_W18_C.wav', 'WD_D_W15_C.wav', 'WD_D_W13_C.wav']
# ns_files = ['WS_D_W20_C.wav', 'WS_D_W18_C.wav', 'WS_D_W15_C.wav', 'WS_D_W13_C.wav']

# ns_files = ['FM_50_W20.wav', 'FM_50_W18.wav', 'FM_50_W15.wav', 'FM_50_W13.wav']
# ns_files = ['FM_50_W20_C.wav', 'FM_50_W18_C.wav', 'FM_50_W15_C.wav', 'FM_50_W13_C.wav']
# ns_files = ['TD_50_W20_C.wav', 'TD_50_W18_C.wav', 'TD_50_W15_C.wav', 'TD_50_W13_C.wav']
# ns_files = ['TS_50_W20_C.wav', 'TS_50_W18_C.wav', 'TS_50_W15_C.wav', 'TS_50_W13_C.wav']
# ns_files = ['WD_50_W20_C.wav', 'WD_50_W18_C.wav', 'WD_50_W15_C.wav', 'WD_50_W13_C.wav']
# ns_files = ['WS_50_W20_C.wav', 'WS_50_W18_C.wav', 'WS_50_W15_C.wav', 'WS_50_W13_C.wav']

# ns_files = ['FM_250_W20.wav', 'FM_250_W18.wav', 'FM_250_W15.wav', 'FM_250_W13.wav']
# ns_files = ['FM_250_W20_C.wav', 'FM_250_W18_C.wav', 'FM_250_W15_C.wav', 'FM_250_W13_C.wav']
# ns_files = ['TD_250_W20_C.wav', 'TD_250_W18_C.wav', 'TD_250_W15_C.wav', 'TD_250_W13_C.wav']
# ns_files = ['TS_250_W20_C.wav', 'TS_250_W18_C.wav', 'TS_250_W15_C.wav', 'TS_250_W13_C.wav']
# ns_files = ['WD_250_W20_C.wav', 'WD_250_W18_C.wav', 'WD_250_W15_C.wav', 'WD_250_W13_C.wav']
# ns_files = ['WS_250_W20_C.wav', 'WS_250_W18_C.wav', 'WS_250_W15_C.wav', 'WS_250_W13_C.wav']

# ns_files = ['FM_1000_W20.wav', 'FM_1000_W18.wav', 'FM_1000_W15.wav', 'FM_1000_W13.wav']
# ns_files = ['FM_1000_W20_C.wav', 'FM_1000_W18_C.wav', 'FM_1000_W15_C.wav', 'FM_1000_W13_C.wav']
# ns_files = ['TD_1000_W20_C.wav', 'TD_1000_W18_C.wav', 'TD_1000_W15_C.wav', 'TD_1000_W13_C.wav']
# ns_files = ['TS_1000_W20_C.wav', 'TS_1000_W18_C.wav', 'TS_1000_W15_C.wav', 'TS_1000_W13_C.wav']
# ns_files = ['WD_1000_W20_C.wav', 'WD_1000_W18_C.wav', 'WD_1000_W15_C.wav', 'WD_1000_W13_C.wav']
# ns_files = ['WS_1000_W20_C.wav', 'WS_1000_W18_C.wav', 'WS_1000_W15_C.wav', 'WS_1000_W13_C.wav']







ns_ES_20_files = ['FM_ES_W20.wav', 'FM_ES_W20_C.wav', 'TD_ES_W20_C.wav', 'TS_ES_W20_C.wav', 'WD_ES_W20_C.wav', 'WS_ES_W20_C.wav']
ns_ES_18_files = ['FM_ES_W18.wav', 'FM_ES_W18_C.wav', 'TD_ES_W18_C.wav', 'TS_ES_W18_C.wav', 'WD_ES_W18_C.wav', 'WS_ES_W18_C.wav']
ns_ES_15_files = ['FM_ES_W15.wav', 'FM_ES_W15_C.wav', 'TD_ES_W15_C.wav', 'TS_ES_W15_C.wav', 'WD_ES_W15_C.wav', 'WS_ES_W15_C.wav']
ns_ES_13_files = ['FM_ES_W13.wav', 'FM_ES_W13_C.wav', 'TD_ES_W13_C.wav', 'TS_ES_W13_C.wav', 'WD_ES_W13_C.wav', 'WS_ES_W13_C.wav']

ns_LS_20_files = ['FM_LS_W20.wav', 'FM_LS_W20_C.wav', 'TD_LS_W20_C.wav', 'TS_LS_W20_C.wav', 'WD_LS_W20_C.wav', 'WS_LS_W20_C.wav']
ns_LS_18_files = ['FM_LS_W18.wav', 'FM_LS_W18_C.wav', 'TD_LS_W18_C.wav', 'TS_LS_W18_C.wav', 'WD_LS_W18_C.wav', 'WS_LS_W18_C.wav']
ns_LS_15_files = ['FM_LS_W15.wav', 'FM_LS_W15_C.wav', 'TD_LS_W15_C.wav', 'TS_LS_W15_C.wav', 'WD_LS_W15_C.wav', 'WS_LS_W15_C.wav']
ns_LS_13_files = ['FM_LS_W13.wav', 'FM_LS_W13_C.wav', 'TD_LS_W13_C.wav', 'TS_LS_W13_C.wav', 'WD_LS_W13_C.wav', 'WS_LS_W13_C.wav']

# ns_files = ['FM_D_W20.wav', 'FM_D_W20_C.wav', 'TD_D_W20_C.wav', 'TS_D_W20_C.wav', 'WD_D_W20_C.wav', 'WS_D_W20_C.wav']
# ns_files = ['FM_D_W18.wav', 'FM_D_W18_C.wav', 'TD_D_W18_C.wav', 'TS_D_W18_C.wav', 'WD_D_W18_C.wav', 'WS_D_W18_C.wav']
# ns_files = ['FM_D_W15.wav', 'FM_D_W15_C.wav', 'TD_D_W15_C.wav', 'TS_D_W15_C.wav', 'WD_D_W15_C.wav', 'WS_D_W15_C.wav']
# ns_files = ['FM_D_W13.wav', 'FM_D_W13_C.wav', 'TD_D_W13_C.wav', 'TS_D_W13_C.wav', 'WD_D_W13_C.wav', 'WS_D_W13_C.wav']

# ns_files = ['FM_50_W20.wav', 'FM_50_W20_C.wav', 'TD_50_W20_C.wav', 'TS_50_W20_C.wav', 'WD_50_W20_C.wav', 'WS_50_W20_C.wav']
# ns_files = ['FM_50_W18.wav', 'FM_50_W18_C.wav', 'TD_50_W18_C.wav', 'TS_50_W18_C.wav', 'WD_50_W18_C.wav', 'WS_50_W18_C.wav']
# ns_files = ['FM_50_W15.wav', 'FM_50_W15_C.wav', 'TD_50_W15_C.wav', 'TS_50_W15_C.wav', 'WD_50_W15_C.wav', 'WS_50_W15_C.wav']
# ns_files = ['FM_50_W13.wav', 'FM_50_W13_C.wav', 'TD_50_W13_C.wav', 'TS_50_W13_C.wav', 'WD_50_W13_C.wav', 'WS_50_W13_C.wav']

# ns_files = ['FM_250_W20.wav', 'FM_250_W20_C.wav', 'TD_250_W20_C.wav', 'TS_250_W20_C.wav', 'WD_250_W20_C.wav', 'WS_250_W20_C.wav']
# ns_files = ['FM_250_W18.wav', 'FM_250_W18_C.wav', 'TD_250_W18_C.wav', 'TS_250_W18_C.wav', 'WD_250_W18_C.wav', 'WS_250_W18_C.wav']
# ns_files = ['FM_250_W15.wav', 'FM_250_W15_C.wav', 'TD_250_W15_C.wav', 'TS_250_W15_C.wav', 'WD_250_W15_C.wav', 'WS_250_W15_C.wav']
# ns_files = ['FM_250_W13.wav', 'FM_250_W13_C.wav', 'TD_250_W13_C.wav', 'TS_250_W13_C.wav', 'WD_250_W13_C.wav', 'WS_250_W13_C.wav']

# ns_files = ['FM_1000_W20.wav', 'FM_1000_W20_C.wav', 'TD_1000_W20_C.wav', 'TS_1000_W20_C.wav', 'WD_1000_W20_C.wav', 'WS_1000_W20_C.wav']
# ns_files = ['FM_1000_W18.wav', 'FM_1000_W18_C.wav', 'TD_1000_W18_C.wav', 'TS_1000_W18_C.wav', 'WD_1000_W18_C.wav', 'WS_1000_W18_C.wav']
# ns_files = ['FM_1000_W15.wav', 'FM_1000_W15_C.wav', 'TD_1000_W15_C.wav', 'TS_1000_W15_C.wav', 'WD_1000_W15_C.wav', 'WS_1000_W15_C.wav']
# ns_files = ['FM_1000_W13.wav', 'FM_1000_W13_C.wav', 'TD_1000_W13_C.wav', 'TS_1000_W13_C.wav', 'WD_1000_W13_C.wav', 'WS_1000_W13_C.wav']







# For HR and LR
ns_R_ES_20_files = ['FM_ES_W20_C.wav', 'FM_ES_W20.wav', 'HR_ES_W20.wav', 'LR_ES_W20.wav']
ns_R_LS_20_files = ['FM_LS_W20_C.wav', 'FM_LS_W20.wav', 'HR_LS_W20.wav', 'LR_LS_W20.wav']
ns_R_ES_13_files = ['FM_ES_W13_C.wav', 'FM_ES_W13.wav', 'HR_ES_W13.wav', 'LR_ES_W13.wav']
ns_R_LS_13_files = ['FM_LS_W13_C.wav', 'FM_LS_W13.wav', 'HR_LS_W13.wav', 'LR_LS_W13.wav']

n_R_20_files = ['FM_W20_C.wav', 'FM_W20.wav', 'HR_W20.wav', 'LR_W20.wav']
n_R_13_files = ['FM_W13_C.wav', 'FM_W13.wav', 'HR_W13.wav', 'LR_W13.wav']

s_R_ES_files = ['FM_ES_C.wav', 'FM_ES.wav', 'HR_ES.wav', 'LR_ES.wav']
s_R_LS_files = ['FM_LS_C.wav', 'FM_LS.wav', 'HR_LS.wav', 'LR_LS.wav']

ns_HR_ES_files = ['HR_ES_W20.wav', 'HR_ES_W18.wav', 'HR_ES_W15.wav', 'HR_ES_W13.wav']
ns_LR_ES_files = ['LR_ES_W20.wav', 'LR_ES_W18.wav', 'LR_ES_W15.wav', 'LR_ES_W13.wav']
ns_HR_LS_files = ['HR_LS_W20.wav', 'HR_LS_W18.wav', 'HR_LS_W15.wav', 'HR_LS_W13.wav']
ns_LR_LS_files = ['LR_LS_W20.wav', 'LR_LS_W18.wav', 'LR_LS_W15.wav', 'LR_LS_W13.wav']

n_HR_files = ['HR_W20.wav', 'HR_W18.wav', 'HR_W15.wav', 'HR_W13.wav']
n_LR_files = ['LR_W20.wav', 'LR_W18.wav', 'LR_W15.wav', 'LR_W13.wav']

# Flight Files
fl_files = [['P1_10_-2.wav', 'P1_10_-1.wav', 'P1_10_0.wav', 'P1_10_1.wav', 'P1_10_2.wav'],
            ['P2_10_-2.wav', 'P2_10_-1.wav', 'P2_10_0.wav', 'P2_10_1.wav', 'P2_10_2.wav'],
            ['P3_10_-2.wav', 'P3_10_-1.wav', 'P3_10_0.wav', 'P3_10_1.wav', 'P3_10_2.wav'],
            ['P4_10_-2.wav', 'P4_10_-1.wav', 'P4_10_0.wav', 'P4_10_1.wav', 'P4_10_2.wav'],
            ['P5_10_-2.wav', 'P5_10_-1.wav', 'P5_10_0.wav', 'P5_10_1.wav', 'P5_10_2.wav'],
            ['P6_10_-2.wav', 'P6_10_-1.wav', 'P6_10_0.wav', 'P6_10_1.wav', 'P6_10_2.wav'],
            ['P7_10_-2.wav', 'P7_10_-1.wav', 'P7_10_0.wav', 'P7_10_1.wav', 'P7_10_2.wav'],
            ['P8_10_-2.wav', 'P8_10_-1.wav', 'P8_10_0.wav', 'P8_10_1.wav', 'P8_10_2.wav'],
            ['P9_10_-2.wav', 'P9_10_-1.wav', 'P9_10_0.wav', 'P9_10_1.wav', 'P9_10_2.wav'],
            ['P1_5_-4.wav', 'P1_5_-3.wav', 'P1_5_-2.wav', 'P1_5_-1.wav', 'P1_5_0.wav', 'P1_5_1.wav', 'P1_5_2.wav', 'P1_5_3.wav', 'P1_5_4.wav'],
            ['P2_5_-4.wav', 'P2_5_-3.wav', 'P2_5_-2.wav', 'P2_5_-1.wav', 'P2_5_0.wav', 'P2_5_1.wav', 'P2_5_2.wav', 'P2_5_3.wav', 'P2_5_4.wav'],
            ['P3_5_-4.wav', 'P3_5_-3.wav', 'P3_5_-2.wav', 'P3_5_-1.wav', 'P3_5_0.wav', 'P3_5_1.wav', 'P3_5_2.wav', 'P3_5_3.wav', 'P3_5_4.wav'],
            ['P4_5_-4.wav', 'P4_5_-3.wav', 'P4_5_-2.wav', 'P4_5_-1.wav', 'P4_5_0.wav', 'P4_5_1.wav', 'P4_5_2.wav', 'P4_5_3.wav', 'P4_5_4.wav'],
            ['P5_5_-4.wav', 'P5_5_-3.wav', 'P5_5_-2.wav', 'P5_5_-1.wav', 'P5_5_0.wav', 'P5_5_1.wav', 'P5_5_2.wav', 'P5_5_3.wav', 'P5_5_4.wav'],
            ['P6_5_-4.wav', 'P6_5_-3.wav', 'P6_5_-2.wav', 'P6_5_-1.wav', 'P6_5_0.wav', 'P6_5_1.wav', 'P6_5_2.wav', 'P6_5_3.wav', 'P6_5_4.wav'],
            ['P7_5_-4.wav', 'P7_5_-3.wav', 'P7_5_-2.wav', 'P7_5_-1.wav', 'P7_5_0.wav', 'P7_5_1.wav', 'P7_5_2.wav', 'P7_5_3.wav', 'P7_5_4.wav'],
            ['P8_5_-4.wav', 'P8_5_-3.wav', 'P8_5_-2.wav', 'P8_5_-1.wav', 'P8_5_0.wav', 'P8_5_1.wav', 'P8_5_2.wav', 'P8_5_3.wav', 'P8_5_4.wav'],
            ['P9_5_-4.wav', 'P9_5_-3.wav', 'P9_5_-2.wav', 'P9_5_-1.wav', 'P9_5_0.wav', 'P9_5_1.wav', 'P9_5_2.wav', 'P9_5_3.wav', 'P9_5_4.wav'],
            ['P1_1_-9.wav', 'P1_1_-8.wav', 'P1_1_-7.wav', 'P1_1_-6.wav', 'P1_1_-5.wav', 'P1_1_-4.wav', 'P1_1_-3.wav', 'P1_1_-2.wav', 'P1_1_-1.wav', 'P1_1_0.wav', 'P1_1_1.wav', 'P1_1_2.wav', 'P1_1_3.wav', 'P1_1_4.wav', 'P1_1_5.wav', 'P1_1_6.wav', 'P1_1_7.wav', 'P1_1_8.wav', 'P1_1_9.wav'],
            ['P2_1_-9.wav', 'P2_1_-8.wav', 'P2_1_-7.wav', 'P2_1_-6.wav', 'P2_1_-5.wav', 'P2_1_-4.wav', 'P2_1_-3.wav', 'P2_1_-2.wav', 'P2_1_-1.wav', 'P2_1_0.wav', 'P2_1_1.wav', 'P2_1_2.wav', 'P2_1_3.wav', 'P2_1_4.wav', 'P2_1_5.wav', 'P2_1_6.wav', 'P2_1_7.wav', 'P2_1_8.wav', 'P2_1_9.wav'],
            ['P3_1_-9.wav', 'P3_1_-8.wav', 'P3_1_-7.wav', 'P3_1_-6.wav', 'P3_1_-5.wav', 'P3_1_-4.wav', 'P3_1_-3.wav', 'P3_1_-2.wav', 'P3_1_-1.wav', 'P3_1_0.wav', 'P3_1_1.wav', 'P3_1_2.wav', 'P3_1_3.wav', 'P3_1_4.wav', 'P3_1_5.wav', 'P3_1_6.wav', 'P3_1_7.wav', 'P3_1_8.wav', 'P3_1_9.wav'],
            ['P4_1_-9.wav', 'P4_1_-8.wav', 'P4_1_-7.wav', 'P4_1_-6.wav', 'P4_1_-5.wav', 'P4_1_-4.wav', 'P4_1_-3.wav', 'P4_1_-2.wav', 'P4_1_-1.wav', 'P4_1_0.wav', 'P4_1_1.wav', 'P4_1_2.wav', 'P4_1_3.wav', 'P4_1_4.wav', 'P4_1_5.wav', 'P4_1_6.wav', 'P4_1_7.wav', 'P4_1_8.wav', 'P4_1_9.wav'],
            ['P5_1_-9.wav', 'P5_1_-8.wav', 'P5_1_-7.wav', 'P5_1_-6.wav', 'P5_1_-5.wav', 'P5_1_-4.wav', 'P5_1_-3.wav', 'P5_1_-2.wav', 'P5_1_-1.wav', 'P5_1_0.wav', 'P5_1_1.wav', 'P5_1_2.wav', 'P5_1_3.wav', 'P5_1_4.wav', 'P5_1_5.wav', 'P5_1_6.wav', 'P5_1_7.wav', 'P5_1_8.wav', 'P5_1_9.wav'],
            ['P6_1_-9.wav', 'P6_1_-8.wav', 'P6_1_-7.wav', 'P6_1_-6.wav', 'P6_1_-5.wav', 'P6_1_-4.wav', 'P6_1_-3.wav', 'P6_1_-2.wav', 'P6_1_-1.wav', 'P6_1_0.wav', 'P6_1_1.wav', 'P6_1_2.wav', 'P6_1_3.wav', 'P6_1_4.wav', 'P6_1_5.wav', 'P6_1_6.wav', 'P6_1_7.wav', 'P6_1_8.wav', 'P6_1_9.wav'],
            ['P7_1_-9.wav', 'P7_1_-8.wav', 'P7_1_-7.wav', 'P7_1_-6.wav', 'P7_1_-5.wav', 'P7_1_-4.wav', 'P7_1_-3.wav', 'P7_1_-2.wav', 'P7_1_-1.wav', 'P7_1_0.wav', 'P7_1_1.wav', 'P7_1_2.wav', 'P7_1_3.wav', 'P7_1_4.wav', 'P7_1_5.wav', 'P7_1_6.wav', 'P7_1_7.wav', 'P7_1_8.wav', 'P7_1_9.wav'],
            ['P8_1_-9.wav', 'P8_1_-8.wav', 'P8_1_-7.wav', 'P8_1_-6.wav', 'P8_1_-5.wav', 'P8_1_-4.wav', 'P8_1_-3.wav', 'P8_1_-2.wav', 'P8_1_-1.wav', 'P8_1_0.wav', 'P8_1_1.wav', 'P8_1_2.wav', 'P8_1_3.wav', 'P8_1_4.wav', 'P8_1_5.wav', 'P8_1_6.wav', 'P8_1_7.wav', 'P8_1_8.wav', 'P8_1_9.wav'],
            ['P9_1_-9.wav', 'P9_1_-8.wav', 'P9_1_-7.wav', 'P9_1_-6.wav', 'P9_1_-5.wav', 'P9_1_-4.wav', 'P9_1_-3.wav', 'P9_1_-2.wav', 'P9_1_-1.wav', 'P9_1_0.wav', 'P9_1_1.wav', 'P9_1_2.wav', 'P9_1_3.wav', 'P9_1_4.wav', 'P9_1_5.wav', 'P9_1_6.wav', 'P9_1_7.wav', 'P9_1_8.wav', 'P9_1_9.wav']]