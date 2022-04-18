import numpy as np
import matplotlib.pyplot as plt

def simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r):
    return (V0 + (Vinf - V0) * (1 - Rstar / r)**beta) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def double_beta_law(Vcore, V0, Vinf1, Vinf2, beta1, beta2, Scl_ht, Rstar, r):
    Vext = Vinf2 - Vinf1
    Vinf = Vinf2
    return (V0 + (Vinf - Vext - V0) * (1 - Rstar / r)**beta1 + Vext * (1 - Rstar / r)**beta2) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def main():
    Vcore = 0.8
    V0 = 30
    Vinf = 250
    beta = 30.0
    Rstar = 139.2
    Scl_ht = 0.022
    rmax = 1000 * Rstar
    R = np.linspace(Rstar, rmax, num = 1000)
    velocity = np.array([simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar * 10**5, r * 10**5) for r in R])
    #velocity2 = np.array([simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar * 10**5, r * 10**5) for r in  R])

    velocity3 = np.array(map(lambda x: double_beta_law(Vcore, 100, 150, 450, beta, 700.0, Scl_ht, Rstar * 10**5, x * 10**5), R))

    plt.figure()
    plt.plot(R / Rstar, velocity, 'r')
    #plt.plot(np.log10(R / Rstar), velocity2, 'b')
    plt.plot(R / Rstar, velocity3, 'g')

    #plt.xlim(Rstar - 5, rmax + 10)
    #plt.ylim(0, 400)
    plt.show()

if __name__ == '__main__':
    main()

