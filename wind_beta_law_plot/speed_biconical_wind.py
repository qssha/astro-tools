import numpy as np
import matplotlib.pyplot as plt

def simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r):
    return (V0 + (Vinf - V0) * (1 - Rstar / r)**beta) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def double_beta_law(Vcore, V0, Vinf1, Vinf2, beta1, beta2, Scl_ht, Rstar, r):
    Vext = Vinf2 - Vinf1
    Vinf = Vinf2
    return (V0 + (Vinf - Vext - V0) * (1 - Rstar / r)**beta1 + Vext * (1 - Rstar / r)**beta2) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def biconical_wind(v_0, v_inf, R_v, alpha, l):
    return v_0 + (v_inf - v_0) * (l / R_v)**alpha / ((l / R_v)**alpha + 1)

def main():
    v_0 = 300 * 10**5
    R_v = 10**10
    alpha = 1.0
    v_esc = 15000 * 10**5
    v_inf = 0.1 * v_esc
    r_init = 1.2 * 10**9
    r_max = 10**13
    l_arr = np.geomspace(r_init, r_max, num = 1000)
    velocity = np.array([biconical_wind(v_0, v_inf, R_v, alpha, l) for l in l_arr])
    velocity2 = np.array([biconical_wind(v_0, 1.0 * v_esc, 10**14, alpha, l) for l in l_arr])

    plt.figure()
    plt.plot(l_arr / r_init, velocity / 10**5, 'r')
    plt.plot(l_arr / r_init, velocity2 / 10**5, 'b')

    #plt.plot(l_arr / R_v, velocity / v_inf, 'r')
    #plt.plot(np.log10(R / Rstar), velocity2, 'b')

    #plt.xlim(Rstar - 5, rmax + 10)
    #plt.ylim(0, 400)
    plt.show()

if __name__ == '__main__':
    main()

