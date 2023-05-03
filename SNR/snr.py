from scipy import signal
from matplotlib import pyplot as plt
from constants import *
import numpy as np
import pandas as pd
import librosa, os

def main():
    # os.chdir('SNR')
    
    # s_list, ns_list, n_list, ts_list = get_sigs(s_ES_files_2, ns_ES_20_files_2, n_20_files_2, ts_files)
    # get_SNR_mounts(s_list, n_list, 'Pure', '20 m/s Winds', nc_mounts)
    # get_SNR_mounts(s_list, ns_list, 'Given Signal', '20 m/s Winds', nc_mounts)
    # get_SNR_mounts(ns_list, n_list, 'Given Noise', '20 m/s Winds', nc_mounts)
    # s_list, ns_list, n_list, ts_list = get_sigs(s_ES_files_2, ns_FM_ES_files_2, n_FM_files_2, ts_files)
    # get_SNR_winds(s_list[5], n_list, 'Pure', 'Flat Mount (No Cotton)')
    # get_SNR_winds(s_list[1], ns_list, 'Given Signal', 'Flat Mount (Cotton)')
    # get_SNR_winds(ns_list, n_list, 'Given Noise', 'Flat Mount (Cotton)')
    # s_list, ns_list, n_list, ts_list = get_sigs(s_R_ES_files, ns_R_ES_13_files, n_R_13_files, ts_files)
    # get_SNR_mounts(s_list, n_list, 'Pure', '13 m/s Winds', ramp_mounts)
    # get_SNR_mounts(s_list, ns_list, 'Given Signal', '20 m/s Winds', ramp_mounts)
    # get_SNR_mounts(ns_list, n_list, 'Given Noise', '20 m/s Winds', ramp_mounts)
    # s_list, ns_list, n_list, ts_list = get_sigs(s_R_ES_files, ns_HR_ES_files, n_HR_files, ts_files)
    # get_SNR_winds(s_list[1], n_list, 'Pure', 'High Ramp (No Cotton)')
    # get_SNR_winds(s_list[1], ns_list, 'Given Signal', 'High Ramp (No Cotton)')
    # get_SNR_winds(ns_list, n_list, 'Given Noise', 'High Ramp (No Cotton)')
    
    # [noisy_signal, noise, true_signal]
    flight_list = get_sigs()
    snr_db = db_array_to_mean(get_SNR_flight(flight_list[0], flight_list[1], 'Given Noise', 'Angel (Air Speed: 25 m/s, Alt.: 40 m)'))
    dist_array, snr_dist_angel, snr_dist_angel_model = find_avg_snr_db_dist_array(dist_array_big, snr_db, angel_vel_array, angel_alt)
    dist_array, snr_dist_albatross, snr_dist_albatross_model = find_avg_snr_db_dist_array(dist_array_big, None, wt_vel_array, albatross_alt)
    plt.plot(dist_array, snr_dist_angel, label=r'Angel, Net Wind: 31 $\frac{m}{s}$', lw=0.75, alpha=0.75)
    plt.plot(dist_array, snr_dist_albatross_model, label=r'Albatross, Model: 14 $\frac{m}{s}$', lw=0.75, alpha=0.75)
    plt.plot(dist_array, snr_dist_angel_model, label=r'Angel, Model: 31 $\frac{m}{s}$', lw=0.75, alpha=0.75)
    plt.title('Average SNR per Distance\n(Flat Mount with Cotton)')
    plt.xlabel('Distance (m)')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_sigs(s_files=None, ns_files=None, n_files=None, ts_files=None, mono=True):
    if s_files:
        return retrieve_files(s_files, mono), retrieve_files(ns_files, mono), retrieve_files(n_files, mono), retrieve_files(ts_files, mono)
    else:
        return retrieve_files([flight_names[0], flight_names[1]], mono)

def retrieve_files(file_list, mono=True):
    s_list = []
    if file_list:
        for f in file_list:
            dat, _ = librosa.load(f, sr=big_sr, mono=mono)
            s_list.append(dat)
    return s_list

def get_SNR_flight(data_1, data_2, snr_type, titl):
    freq, _, _, db_rolled_both = get_SNR_arrays(data_1, data_2, snr_type)
    low_count = len(freq[freq < 2500])
    freq = freq[freq < 2500]
    db_rolled_both = db_rolled_both[0:low_count]
    db_rolled_both = db_rolled_both[np.isfinite(db_rolled_both)]
    # plt.plot(freq, db_rolled_both, label='', lw=1, alpha=0.75)
    # plt.title(titl + ' SNR')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('SNR (dB)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    return db_rolled_both

def get_SNR_winds(data_1, data_2, snr_type, mount):
    wnds = ['20', '18', '15', '13']
    if snr_type == 'Given Noise':
        for d1, d2, wd in zip(data_1, data_2, wnds):
            freq, _, _, db_rolled_both = get_SNR_arrays(d1, d2, snr_type)
            low_count = len(freq[freq < 2500])
            freq = freq[freq < 2500]
            db_rolled_both = db_rolled_both[0:low_count]
            plt.plot(freq, db_rolled_both, label=wd + ' m/s', lw=1, alpha=0.5)
    else:
        for d2, wd in zip(data_2, wnds):
            freq, _, _, db_rolled_both = get_SNR_arrays(data_1, d2, snr_type)
            low_count = len(freq[freq < 2500])
            freq = freq[freq < 2500]
            db_rolled_both = db_rolled_both[0:low_count]
            plt.plot(freq, db_rolled_both, label=wd + ' m/s', lw=1, alpha=0.75)
    plt.title(mount + ' SNR')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_SNR_mounts(data_1, data_2, snr_type, name, mounts):
    for l1, l2, mt in zip(data_1, data_2, mounts):
        freq, _, _, db_rolled_both = get_SNR_arrays(l1, l2, snr_type)
        low_count = len(freq[freq < 2500])
        freq = freq[freq < 2500]
        db_rolled_both = db_rolled_both[0:low_count]
        plt.plot(freq, db_rolled_both, label=mt, lw=1, alpha=0.75)
    
    plt.title(name + ' SNR')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_SNR_arrays(data_1, data_2, snr_type):
    # snr_type: Pure, Given Signal, Given Noise, System
    # data_1: sig, sig, noisy_sig, sig
    # data_2: noise, noisy_sig, noise, noisy_sig
    
    data_1_freq, data_1_data = signal.welch(x=data_1, fs=big_sr, nperseg=32768, average='mean')
    data_2_freq, data_2_data = signal.welch(x=data_2, fs=big_sr, nperseg=32768, average='mean')
    
    data_1_roll = pd.Series(data_1_data).rolling(window, center=True).mean().to_numpy()
    data_2_roll = pd.Series(data_2_data).rolling(window, center=True).mean().to_numpy()
    
    if snr_type == 'System':
        data_2_data = data_2_data - data_1_data
        data_2_roll = data_2_roll - data_1_roll
    
    snr_plain = []
    db_plain = []
    snr_rolled_before = []
    db_rolled_before = []
    for l1, l2, l1r, l2r in zip(data_1_data, data_2_data, data_1_roll, data_2_roll):
        if snr_type == 'Given Signal':
            this_data_ratio = 1/(((l2*1.25)/l1) - 1)
            this_roll_ratio = 1/(((l2r*1.25)/l1r) - 1)
        elif snr_type == 'Given Noise':
            this_data_ratio = ((l1)/l2) - 1
            this_roll_ratio = ((l1r)/l2r) - 1
        else:
            this_data_ratio = l1/l2
            this_roll_ratio = l1r/l2r
        snr_plain.append(this_data_ratio)
        db_plain.append(10*np.log10(this_data_ratio))
        snr_rolled_before.append(this_roll_ratio)
        db_rolled_before.append(10*np.log10(this_roll_ratio))
    
    # db_rolled_after = 10*np.log10(pd.Series(snr_plain).rolling(win, center=True).mean().to_numpy())
    db_rolled_both = 10*np.log10(pd.Series(snr_rolled_before).rolling(window, center=True).mean().to_numpy())
    
    return data_1_freq, db_plain, db_rolled_before, db_rolled_both

def value_db_conv(val, val_rat, val_type, result_type, ref=10**-12):
    # val_rat options: value, ratio
    # val_type options: intensity, power, pressure, voltage, current
    # result_type options: db, value
    factor = 10
    if val_type == 'pressure' or val_type == 'voltage' or val_type == 'current':
        factor = 20
        ref = 2*(10**-5) if val_type == 'pressure' else ref
    if result_type == 'value' and val_rat == 'value':
        return np.power(10, (val/factor))*ref
    elif result_type == 'value' and val_rat == 'ratio':
        return np.power(10, (val/factor))
    elif result_type == 'db' and val_rat == 'value':
        return factor*np.log10(val/ref)
    else: # result_type == 'db' and val_rat == 'ratio'
        return factor*np.log10(val)

def db_array_to_mean(db_array):
    avg_ratio = np.mean(value_db_conv(db_array, 'ratio', 'power', 'value'))
    return value_db_conv(avg_ratio, 'ratio', 'power', 'db')

def find_avg_snr_db_dist_array(dist_array, snr_db=None, vel_array=angel_vel_array, special_dist=angel_alt):
    if not type(dist_array) == np.ndarray:
        dist_array = [dist_array]
    snr_pred_db = calc_snr_pred(freqs, spl_src, dir_fact, vel_array, special_dist, angel_temp, angel_rel_hum, angel_p_bar, angel_p_ref)
    diff = db_array_to_mean(snr_pred_db) - snr_db if snr_db else 0
    
    snr_avg_db_dist_model = []
    snr_avg_db_dist = []
    for dist in dist_array:
        snr_avg_db_model = db_array_to_mean(calc_snr_pred(freqs, spl_src, dir_fact, vel_array, dist, angel_temp, angel_rel_hum, angel_p_bar, angel_p_ref))
        snr_avg_db_dist_model.append(snr_avg_db_model)
        snr_avg_db_dist.append(snr_avg_db_model - diff)
    if diff == 0:
        snr_avg_db_dist = None
    return dist_array, snr_avg_db_dist, snr_avg_db_dist_model

def calc_snr_pred(freqs, src_spl, dir_fact, vel_array, distance, temperature, rel_hum, p_bar, p_ref):
    abs_coeff_db, sound_speed_const, air_dens = calc_coeff(freqs, 1, temperature, rel_hum, p_bar, p_ref)
    sos_wind = sound_speed_const + vel_array[1]
    src_p_acc = value_db_conv(src_spl, 'value', 'pressure', 'value')
    src_intensity = p_acc_intensity_conv(src_p_acc, 'pressure', sos_wind, air_dens)
    src_pwr = pwr_intensity_conv(src_intensity, 'intensity', dst_from_src, dir_fact)
    src_pwr_db = value_db_conv(src_pwr, 'value', 'power', 'db')
    
    src_spl_from_dist = np.empty(shape=len(freqs))
    for i in range(len(src_spl_from_dist)):
        src_spl_from_dist[i] = src_pwr_db - 10*np.log10(4*np.pi*(distance**2)/dir_fact) - abs_coeff_db[i]*distance
    src_p_acc_dist = value_db_conv(src_spl_from_dist, 'value', 'pressure', 'value')
    src_int_dist = p_acc_intensity_conv(src_p_acc_dist, 'pressure', sos_wind, air_dens)
    src_pwr_dist = pwr_intensity_conv(src_int_dist, 'intensity', 1, dir_fact)
    src_pwr_db_dist = value_db_conv(src_pwr_dist, 'value', 'power', 'db')    
    src_pwr_db_dist = db_array_to_mean(src_pwr_db_dist)
    
    # new_noise_spl = noise_spl + 50*np.log10((vel_array[1] + vel_array[2])/vel_array[1])
    # noise_p_acc = value_db_conv(new_noise_spl, 'value', 'pressure', 'value')
    # noise_intensity = p_acc_intensity_conv(noise_p_acc, 'pressure', sound_speed_const, air_dens)
    # noise_pwr = pwr_intensity_conv(noise_intensity, 'intensity', 1, dir_fact)
    # noise_pwr_db = value_db_conv(noise_pwr, 'value', 'power', 'db')
    pow_wind = 0.5*air_dens*cs_area*np.power(vel_array[0] + vel_array[2], 3)
    noise_pwr_db = 10*np.log10(pow_wind/(10**(-12)))
    
    return src_pwr_db_dist - noise_pwr_db

def p_acc_intensity_conv(val, in_type, sound_speed, air_dens):
    if in_type == 'pressure':
        return np.power(val, 2)/(air_dens*sound_speed)
    else: # in_type == 'intensity'
        return np.sqrt(val*air_dens*sound_speed)

def pwr_intensity_conv(val, in_type, distance, dir_fact):
    if in_type == 'power':
        return val*dir_fact/(4*np.pi*np.square(distance))
    else: # in_type == 'intensity'
        return val*4*np.pi*np.square(distance)/dir_fact

def calc_coeff(freqs, distance, temperature, rel_hum, p_bar, p_ref):
    p_sat_ref = p_sat_ref_easy(temperature)
    mol_conc_wv = mol_conc_water_vapor(rel_hum, p_sat_ref, p_bar, p_ref)
    oxy_freq = oxy_relax_freq(p_bar, p_ref, 100*mol_conc_wv)
    nit_freq = nit_relax_freq(temperature, p_bar, p_ref, 100*mol_conc_wv)
    abs_coeff_db = distance*absorption_coeff(temperature, p_bar, p_ref, freqs, oxy_freq, nit_freq)
    
    mol_mix = mol_mass_mix(mol_conc_wv)
    hcr_mix = heat_cap_ratio_mix(mol_conc_wv)
    sound_speed = speed_of_sound(temperature, mol_mix, hcr_mix)
    air_dens = air_density(temperature, p_bar, mol_mix)
    return abs_coeff_db, sound_speed, air_dens

def p_sat_ref_easy(temperature):
    return np.power(10, -6.8346*np.power(273.16/temperature, 1.261) + 4.6151)

def mol_conc_water_vapor(rel_hum, p_sat_ref, p_bar, p_ref):
    return (100*rel_hum*(p_sat_ref/(p_bar/p_ref)))/100

def mol_mass_mix(mol_conc_water_vapor):
    return mol_conc_water_vapor*0.018016 + (1 - mol_conc_water_vapor)*0.02897

def heat_cap_ratio_mix(mol_conc_water_vapor):
    return 1/(mol_conc_water_vapor/(1.33 - 1) + (1 - mol_conc_water_vapor)/(1.4 - 1)) + 1

def speed_of_sound(temperature, mol_mass_mix, heat_cap_ratio_mix):
    return np.sqrt(heat_cap_ratio_mix*8.314462*temperature/mol_mass_mix)

def air_density(temperature, p_bar, mol_mass_mix):
    return mol_mass_mix*p_bar/(8.314462*temperature)

def oxy_relax_freq(p_bar, p_ref, mol_conc_water_vapor):
    return (p_bar/p_ref)*(24 + 40400*mol_conc_water_vapor*((0.02 + mol_conc_water_vapor)/(0.391 + mol_conc_water_vapor)))

def nit_relax_freq(temperature, p_bar, p_ref, mol_conc_water_vapor):
    return (p_bar/p_ref)*np.power(temperature/293.15, -0.5)*(9 + 280*mol_conc_water_vapor*np.exp(-4.17*(np.power(temperature/293.15, -1/3) - 1)))

def absorption_coeff(temperature, p_bar, p_ref, freq, oxy_relax_freq, nit_relax_freq):
    return 10*np.log10(np.exp(np.power(freq, 2)*(1.84*(10**-11)*np.power(p_bar/p_ref, -1)*np.power(temperature/293.15, 1/2) + np.power(temperature/293.15, -5/2)*(0.01275*np.exp(-2239/temperature)*(oxy_relax_freq/(np.power(freq, 2) + np.power(oxy_relax_freq, 2))) + 0.1068*np.exp(-3352/temperature)*(nit_relax_freq/(np.power(freq, 2) + np.power(nit_relax_freq, 2)))))))

if __name__ == '__main__':
    main()