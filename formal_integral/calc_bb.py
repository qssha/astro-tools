import numpy as np

def star_blackbody_spectrum(L_star_solar, R_star_cm, wavelengths_angstrom, distance_kpc=1.0):
    C_CGS = 2.99792458e10
    H_CGS = 6.62607015e-27
    K_B_CGS = 1.380649e-16
    SIGMA_SB = 5.670374419e-5
    L_SOL_CGS = 3.828e33
    PC_CGS = 3.0856775814913673e18

    wave = np.asarray(wavelengths_angstrom, dtype=np.float64)
    
    L_star_cgs = L_star_solar * L_SOL_CGS
    
    T_eff = (L_star_cgs / (4.0 * np.pi * R_star_cm**2 * SIGMA_SB)) ** 0.25
    
    wave_cm = wave * 1.0e-8
    nu = C_CGS / wave_cm
    
    x = H_CGS * nu / (K_B_CGS * T_eff)
    x = np.clip(x, 1.0e-300, 700.0)
    
    B_nu = (2.0 * H_CGS * nu**3 / C_CGS**2) / np.expm1(x)
    
    distance_cm = float(distance_kpc) * 1000.0 * PC_CGS
    
    flux_fnu = np.pi * B_nu * (R_star_cm / distance_cm) ** 2
    
    flux_out = flux_fnu * C_CGS * 1.0e8 / wave**2
    
    return wavelengths_angstrom, flux_out.astype(np.float64)
