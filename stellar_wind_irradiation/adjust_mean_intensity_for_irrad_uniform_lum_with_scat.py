import numpy as np
import pandas as pd
from scipy import interpolate

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

def parse_eddfactor_file(file_dir, ND):
    file_path = file_dir + "/EDDFACTOR"
    dtype_string = "(" + str(ND + 1) + ",)float64"
    eddfactor_array = np.fromfile(file_path, dtype=dtype_string)
    return eddfactor_array

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

    eddfactor_array = parse_eddfactor_file(file_dir, ND)

    # Uniform law
    mean_intensity_for_uniform_lum = lambda delta_distance, delta_tau: x_ray_mean_intensity(get_uniform_x_ray_lum(full_x_ray_lum, high_freq_limit - low_freq_limit), delta_distance, delta_tau)
    eddfactor_array[(eddfactor_array[:, -1] > low_freq_limit / 10**15) & (eddfactor_array[:, -1] < high_freq_limit / 10**15), :-1] += mean_intensity_for_uniform_lum(delta_distance_array, delta_tau_es_array) 
    #

    return eddfactor_array

file_dir = "."
ND = 55

eddfactor_array_mod = modify_eddfactor(file_dir, ND, 5 * 10**40, 9.4 * 10**12)
eddfactor_array_mod.tofile("EDDFACTOR_MOD")
