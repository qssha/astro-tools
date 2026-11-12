import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

def parse_direct_access_file(file_path, ND):
    dtype_string = "(" + str(ND + 1) + ",)float64"
    direct_access_file_array = np.fromfile(file_path, dtype=dtype_string)
    return direct_access_file_array

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

# init sound speed
ND, rvtj_data_frame = parse_rvtj_file(".")
vel_inv = rvtj_data_frame["Velocity (km/s)"].to_numpy()[::-1]
r_inv = rvtj_data_frame["Radius (10^10 cm)"].to_numpy()[::-1]

dfr_array = parse_direct_access_file("." + "/dFR_DATA_FIN", ND)
SPEED_OF_LIGHT_CGS = 2.99702547 * 10**10

wave_lower = 5750
wave_upper = 5765

low_freq_limit = SPEED_OF_LIGHT_CGS / (wave_upper * 10**-8)
high_freq_limit = SPEED_OF_LIGHT_CGS / (wave_lower * 10**-8)
freq_indexes = np.where((dfr_array[:, -1] > low_freq_limit / 10**15) & (dfr_array[:, -1] < high_freq_limit / 10**15))[0]
dfr_optic = dfr_array[freq_indexes, :]

#plt.imshow(dfr_optic[:, :-1])
plt.figure()
cp = plt.contourf(vel_inv, (SPEED_OF_LIGHT_CGS / (dfr_optic[:, -1] * 10**7)), dfr_optic[:, :-1])
plt.colorbar(cp)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("V(km/s)", fontsize=16)
plt.ylabel("Wavelength, Ang", fontsize=16)

plt.tight_layout()
plt.show()

plt.figure()
cp = plt.contourf(np.log10(r_inv / r_inv[-1]), (SPEED_OF_LIGHT_CGS / (dfr_optic[:, -1] * 10**7)), dfr_optic[:, :-1])
plt.colorbar(cp)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("R/Rstar", fontsize=16)
plt.ylabel("Wavelength, Ang", fontsize=16)

plt.tight_layout()
plt.show()
