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

def parse_rvtj_file(file_dir):
    file_path = file_dir + "/RVTJ"
    rvtj_first_file_lines = get_lines_from_file_by_number(file_path, list(range(4)))
    ND = int(rvtj_first_file_lines[3].split()[1])
    num_of_columns = math.ceil(ND / 8)
    num_of_first_entry_line = 12
    num_of_params = 20

    rvtj_file_lines = get_lines_from_file_by_number(file_path,
                                                    list(range(num_of_first_entry_line,
                                                               num_of_first_entry_line + (num_of_columns + 1) * num_of_params)))
    rvtj_data_frame = pd.DataFrame()

    for i in range(0, (num_of_columns + 1) * num_of_params, num_of_columns + 1):
        extracted_data = []
        [extracted_data.extend(data.split()) for data in rvtj_file_lines[i + 1:i + num_of_columns + 1]]
        rvtj_data_frame[rvtj_file_lines[i].strip()] = extracted_data
    return ND, rvtj_data_frame.astype("float64")

def calc_tau(radius, chi, f_clumping):
    # prob will be better to adjust chi units before passing to func
    chi = chi * f_clumping / 10**10

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

def get_uniform_x_ray_lum(full_x_ray_lum, freq_range):
    return full_x_ray_lum / freq_range

def modify_eddfactor(file_dir, ND, full_x_ray_lum, x_ray_source_orbit_radius, low_wavelength_limit_ang=800, high_wavelength_limit_ang=912):
    low_freq_limit = SPEED_OF_LIGHT_CGS / (high_wavelength_limit_ang * 10**-8)
    high_freq_limit = SPEED_OF_LIGHT_CGS / (low_wavelength_limit_ang * 10**-8)

    meanopac_data_frame = parse_meanopac(file_dir, ND)
    meanopac_data_frame['R'] = meanopac_data_frame['R'] * 10**10
    tau_es_interp = interpolate.interp1d(meanopac_data_frame['R'], meanopac_data_frame['Tau(es)'])

    delta_distance_array = np.abs(x_ray_source_orbit_radius - meanopac_data_frame['R'])
    delta_tau_es_array = np.abs(tau_es_interp(x_ray_source_orbit_radius) - meanopac_data_frame['Tau(es)'])

    eddfactor_array = parse_direct_access_file(file_dir + "/EDDFACTOR", ND)

    # Uniform law
    mean_intensity_for_uniform_lum = lambda delta_distance, delta_tau: x_ray_mean_intensity(get_uniform_x_ray_lum(full_x_ray_lum, high_freq_limit - low_freq_limit), delta_distance, delta_tau)
    eddfactor_array[(eddfactor_array[:, -1] > low_freq_limit / 10**15) & (eddfactor_array[:, -1] < high_freq_limit / 10**15), :-1] += mean_intensity_for_uniform_lum(delta_distance_array, delta_tau_es_array) 
    #

    return eddfactor_array

"""
file_dir = "."
ND = 55

eddfactor_array_mod = modify_eddfactor(file_dir, ND, 5 * 10**40, 9.4 * 10**12)
eddfactor_array_mod.tofile("EDDFACTOR_MOD")
"""
