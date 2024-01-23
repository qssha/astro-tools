import numpy as np
import pandas as pd
from scipy import interpolate

import math

SPEED_OF_LIGHT_CGS = 2.99702547 * 10**10

def get_lines_from_file_by_number(file_to_process, line_numbers):
    with open(file_to_process) as current_file:
        return [line.strip() for i, line in enumerate(current_file) if i in line_numbers]

def parse_meanopac(file_dir, ND):
    file_path = file_dir + "/MEANOPAC"
    meanopac_file_lines = get_lines_from_file_by_number(file_path, list(range(ND + 1)))
    data_frame_header = meanopac_file_lines[0].split()
    meanopac_data = [entry.split() for entry in meanopac_file_lines[1:]] 
    meanopac_data_frame = pd.DataFrame(meanopac_data, columns=data_frame_header)
    return meanopac_data_frame.astype("float64")

def parse_direct_access_file(file_path, ND):
    dtype_string = "(" + str(ND + 1) + ",)float64"
    direct_access_file_array = np.fromfile(file_path, dtype=dtype_string)
    return direct_access_file_array

def calc_tau(radius, chi, f_clumping=None):
    # prob will be better to adjust chi units before passing to func
    chi = chi / 10**10

    if f_clumping is not None:
        chi *= f_clumping

    mean_chi = np.array([(chi[i - 1] + chi[i]) / 2 for i in range(1, len(chi))])
    delta_rad = np.array([radius[i - 1] - radius[i] for i in range(1, len(radius))]) # alternative is np.abs(np.diff(rad))
    delta_tau = mean_chi * delta_rad
    tau_at_outer_boundary = calc_tau_at_outer_boundary(radius, chi)
    tau_at_each_depth = np.cumsum(np.insert(delta_tau, 0, tau_at_outer_boundary))

    return tau_at_each_depth

def calc_tau_at_outer_boundary(radius, chi):
    t1_param_outer_boundary = np.log10(np.abs(chi[0] / chi[3])) / np.log10(radius[3] / radius[0])

    if t1_param_outer_boundary < 2:
        t1_param_outer_boundary = 2

    tau_outer_boundary = radius[0] * chi[0] / (t1_param_outer_boundary - 1)

    return tau_outer_boundary

def x_ray_mean_intensity(x_ray_lum, distance, delta_tau):
    return (x_ray_lum / (16 * np.pi**2 * distance**2)) * np.exp(-delta_tau)

def get_lum_norm_constant(full_x_ray_lum, gamma, low_freq_limit, high_freq_limit):
    if gamma != 1:
        return full_x_ray_lum * (-gamma + 1) / (high_freq_limit**(-gamma + 1) - low_freq_limit**(-gamma + 1))
    else:
        return full_x_ray_lum  / (np.log(high_freq_limit) - np.log(low_freq_limit))

def get_lum_at_freq(lum_norm_constant, gamma, freq):
    return lum_norm_constant * freq**-gamma

def irrad_mean_intesities_for_tau_es_mode(full_x_ray_lum, gamma, x_ray_source_orbit_radius, low_freq_limit, high_freq_limit, frequencies, wind_radius, tau_es):
    tau_es_interp = interpolate.interp1d(wind_radius, tau_es)
    delta_distance_array = np.abs(x_ray_source_orbit_radius - wind_radius)
    delta_tau_es_array = np.abs(tau_es_interp(x_ray_source_orbit_radius) - tau_es)

    lum_norm_constant = get_lum_norm_constant(full_x_ray_lum, gamma, low_freq_limit, high_freq_limit)
    lum_per_freq = get_lum_at_freq(lum_norm_constant, gamma, frequencies)

    calc_mean_intensities_for_scattering = lambda lum: x_ray_mean_intensity(lum, delta_distance_array, delta_tau_es_array)
    mean_intensities =  np.array(list(map(calc_mean_intensities_for_scattering, lum_per_freq)))

    return mean_intensities

def irrad_mean_intensities(full_x_ray_lum, gamma, x_ray_source_orbit_radius, low_freq_limit, high_freq_limit, freq_indexes, frequencies, wind_radius, chi_data):
    chi_selected = chi_data[freq_indexes, :-1]
    calc_tau_per_freq = lambda chi_per_freq: calc_tau(wind_radius, chi_per_freq)
    tau_data = np.array(list(map(calc_tau_per_freq, chi_selected)))

    calc_tau_at_source = lambda tau_data_per_freq: interpolate.interp1d(wind_radius, tau_data_per_freq)(x_ray_source_orbit_radius)
    tau_at_source = np.array(list(map(calc_tau_at_source, tau_data)))

    delta_distance = np.abs(x_ray_source_orbit_radius - wind_radius)
    delta_tau = np.array([np.abs(tau_at_source_per_freq - tau_data_per_freq) for tau_at_source_per_freq, tau_data_per_freq in zip(tau_at_source, tau_data)])

    lum_norm_constant = get_lum_norm_constant(full_x_ray_lum, gamma, low_freq_limit, high_freq_limit)
    lum_per_freq = get_lum_at_freq(lum_norm_constant, gamma, frequencies)

    calc_mean_intensities = lambda lum, delta_tau: x_ray_mean_intensity(lum, delta_distance, delta_tau)
    mean_intensities = np.array(list(map(calc_mean_intensities, lum_per_freq, delta_tau)))

    return mean_intensities

def modify_eddfactor(file_dir, full_x_ray_lum, gamma, x_ray_source_orbit_radius, ND, tau_es_mode=True, low_wavelength_limit_ang=1.0, high_wavelength_limit_ang=200):
    low_freq_limit = SPEED_OF_LIGHT_CGS / (high_wavelength_limit_ang * 10**-8)
    high_freq_limit = SPEED_OF_LIGHT_CGS / (low_wavelength_limit_ang * 10**-8)

    meanopac_df = parse_meanopac(file_dir, ND)
    wind_radius = meanopac_df['R'] * 10**10

    eddfactor_array = parse_direct_access_file(file_dir + "/EDDFACTOR", ND)
    freq_indexes = np.where((eddfactor_array[:, -1] > low_freq_limit / 10**15) & (eddfactor_array[:, -1] < high_freq_limit / 10**15))[0]
    frequencies = eddfactor_array[freq_indexes, -1] * 10**15

    if tau_es_mode:
        tau_es = meanopac_df['Tau(es)']
        mean_intensities = irrad_mean_intesities_for_tau_es_mode(full_x_ray_lum, gamma, x_ray_source_orbit_radius, low_freq_limit, high_freq_limit, frequencies, wind_radius, tau_es)
    else:
        chi_data = parse_direct_access_file(file_dir + "/CHI_DATA_FIN", ND)
        mean_intensities = irrad_mean_intensities(full_x_ray_lum, gamma, x_ray_source_orbit_radius, low_freq_limit, high_freq_limit, freq_indexes, frequencies, wind_radius, chi_data)

    eddfactor_array[freq_indexes, :-1] += mean_intensities
    return eddfactor_array

ND = 55
file_dir = "."

eddfactor_array_mod = modify_eddfactor(file_dir, 3 * 10**39, 0, 9.4 * 10**12, ND, False)
eddfactor_array_mod.tofile("EDDFACTOR_MOD")
