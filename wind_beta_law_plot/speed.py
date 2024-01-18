import numpy as np
import matplotlib.pyplot as plt

def simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r):
    return (V0 + (Vinf - V0) * (1 - Rstar / r)**beta) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def double_beta_law(Vcore, V0, Vinf1, Vinf2, beta1, beta2, Scl_ht, Rstar, r):
    Vext = Vinf2 - Vinf1
    Vinf = Vinf2
    return (V0 + (Vinf - Vext - V0) * (1 - Rstar / r)**beta1 + Vext * (1 - Rstar / r)**beta2) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def main():
    Vcore = 0.07
    V0 = 30
    Vinf = 185
    beta = 2.5
    Rstar = 528.9 
    Scl_ht = 0.022 * Rstar
    rmax = 200 * Rstar
    R = np.geomspace(Rstar, rmax, num = 1000)
    velocity = np.array([simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r) for r in R])
    velocity2 = np.array([simple_beta_law(Vcore, 15, Vinf, 2.0, 0.015 * Rstar, Rstar, r) for r in  R])

    #velocity = np.array(map(lambda x: double_beta_law(Vcore, 50, 80, 250, 4, 50.0, Scl_ht, Rstar, x), R))
    #velocity2 = np.array([simple_beta_law(Vcore, 10, 200, 15, Scl_ht, Rstar, r) for r in R])
    #velocity3 = np.array([simple_beta_law(Vcore, 10, 200, 20, Scl_ht, Rstar, r) for r in R])
    #velocity3 = np.array(map(lambda x: double_beta_law(Vcore, 30, 300, 1000, beta, 300.0, Scl_ht, Rstar, x), R))

    plt.figure()
    plt.plot(np.log10(R / Rstar), velocity, 'r')
    plt.plot(np.log10(R / Rstar), velocity2, 'b')
    #plt.plot(np.log10(R / Rstar), velocity3, 'g')

    #plt.xlim(Rstar - 5, rmax + 10)
    #plt.ylim(0, 400)
    plt.show()

if __name__ == '__main__':
    main()

