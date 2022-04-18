import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r):
    return (V0 + (Vinf - V0) * (1 - Rstar / r)**beta) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht)) + 20 * (1 - Rstar / r)**50

rvtj_file = open('RVTJ')
rvtj_lines = rvtj_file.readlines()

raw_rad = []
[raw_rad.extend(r_list.strip().split()) for r_list in rvtj_lines[13:18]] 
rad = np.array(raw_rad, dtype=np.float)

raw_vel = []
[raw_vel.extend(v_list.strip().split()) for v_list in rvtj_lines[19:24]]
vel = np.array(raw_vel, dtype=np.float)

raw_dlnv = []
[raw_dlnv.extend(d_list.strip().split()) for d_list in rvtj_lines[25:30]]
dlnv = np.array(raw_dlnv, dtype=np.float)

Vcore = 7.94 * 10**-2
V0 = 30
Vinf = 300
beta = 4.0
Rstar = 139.2
Scl_ht = 0.022
rmax = 200 * Rstar
R = np.linspace(Rstar, rmax, num = 1000)
velocity = np.array([simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r) for r in rad])

beta_log = lambda x: np.log(simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, np.log(Rstar * 10**5), x))
dlnv_new = derivative(beta_log, np.log(rad * 10**5), dx=0.01) - 1

np.savetxt("RVSIG_COL", np.transpose([rad, vel, dlnv]), fmt='%.8f %1.6e %1.6e')

plt.plot(np.log10(rad / rad[-1]), vel, 'b')
plt.plot(np.log10(rad / rad[-1]), velocity, 'r')

plt.show()
