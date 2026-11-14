import numpy as np
import pandas as pd
import math
from formal_integral import compute_spectrum

SPEED_OF_LIGHT_CGS = 2.99792458 * 10**10

def get_lines_from_file_by_number(file_to_process, line_numbers):
    with open(file_to_process) as current_file:
        return [line.strip() for i, line in enumerate(current_file) if i in line_numbers]

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
    return ND, rvtj_data_frame.astype("float64").iloc[::-1]

def parse_direct_access_file(file_path, ND):
    dtype_string = "(" + str(ND + 1) + ",)float64"
    direct_access_file_array = np.fromfile(file_path, dtype=dtype_string)
    return direct_access_file_array

L_star = 10**6

file_dir = "."
ND, rvtj_df = parse_rvtj_file(file_dir)

r_vector = rvtj_df["Radius (10^10 cm)"].to_numpy() * 10**10
v_wind = rvtj_df["Velocity (km/s)"].to_numpy() * 10**5

low_wavelength_limit_ang = 3000
high_wavelength_limit_ang = 8000

low_freq_limit = SPEED_OF_LIGHT_CGS / (high_wavelength_limit_ang * 10**-8)
high_freq_limit = SPEED_OF_LIGHT_CGS / (low_wavelength_limit_ang * 10**-8)

chi_data = parse_direct_access_file(file_dir + "/CHI_DATA_FIN", ND)
eta_data = parse_direct_access_file(file_dir + "/ETA_DATA_FIN", ND)
es_j_data = parse_direct_access_file(file_dir + "/ES_J_CONV_FIN", ND)

freq_indexes = np.where((chi_data[:, -1] > low_freq_limit / 10**15) & (chi_data[:, -1] < high_freq_limit / 10**15))[0]
frequencies = chi_data[freq_indexes, -1] * 10**15

f_cl = rvtj_df["Clumping Factor"].to_numpy()

chi_selected = chi_data[freq_indexes, :-1][:, ::-1] * f_cl[None, :] / 10**10
eta_selected = eta_data[freq_indexes, :-1][:, ::-1] * f_cl[None, :] / 10**10
es_j_selected = es_j_data[freq_indexes, :-1][:, ::-1] * f_cl[None, :]

SIGMA_T = 6.6524587e-25
n_e = rvtj_df["Electron density"].to_numpy()

eta_selected += SIGMA_T * n_e[None, :] * es_j_selected
#chi_selected = np.clip(chi_selected - SIGMA_T * n_e[None, :], 1e-30, None)

### Macro clumping mod

h_scale = 0.03 * r_vector[0]
tau_cl = chi_selected * h_scale
chi_selected *= (1 - np.exp(-tau_cl)) / tau_cl

###

wave, flux = compute_spectrum(frequencies, chi_selected, eta_selected, r_vector, v_wind, L_star, v_unit=1.0)
np.savetxt("model_spec", np.transpose([wave, flux]))

