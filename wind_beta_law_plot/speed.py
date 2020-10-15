import numpy as np
import matplotlib.pyplot as plt

def simple_beta_law(Vcore, V0, Vinf, beta, Scl_ht, Rstar, r):
    return (V0 + (Vinf - V0) * (1 - Rstar / r)**beta) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def double_beta_law(Vcore, V0, Vinf1, Vinf2, beta1, beta2, Scl_ht, Rstar, r):
    Vext = Vinf2 - Vinf1
    Vinf = Vinf2
    return (V0 + (Vinf - Vext - V0) * (1 - Rstar / r)**beta1 + Vext * (1 - (Rstar / r)**beta2)) / (1 + (V0 / Vcore) * np.exp((Rstar - r) / Scl_ht))

def main():
    Vcore = 0.06
    V0 = 10
    Vinf = 325
    beta = 4.0
    Rstar = 398
    Scl_ht = 0.022
    rmax = 200 * Rstar
    R = np.linspace(Rstar, rmax, num = 1000)
    velocity = np.array([simple_beta_law(Vcore, V0, 325, 4.0, Scl_ht, Rstar, r) for r in R])
    velocity2 = np.array([simple_beta_law(Vcore, V0, 388, 4.5, Scl_ht, Rstar, r) for r in  R])

    #velocity3 = np.array(map(lambda x: double_beta_law(Vcore, V0, 50, Vinf, 0.9, beta, Scl_ht, Rstar, x), R))

    plt.figure()
    plt.plot(np.log10(R / Rstar), velocity, 'r')
    plt.plot(np.log10(R / Rstar), velocity2, 'b')
    #plt.plot(np.log10(R / Rstar), velocity3, 'g')

    #plt.xlim(Rstar - 5, rmax + 10)
    #plt.ylim(0, 400)
    plt.show()

if __name__ == '__main__':
    main()

