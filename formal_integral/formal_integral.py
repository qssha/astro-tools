"""
formal_integral.py
==================
Observer-frame formal integral for a spherically symmetric, expanding
stellar wind in the (p, z) geometry.

    F_nu = (2 pi / d^2) * Int_0^inf  p I_nu(p) dp
    I_nu(p) = Int_zmin^zmax eta_nu(z) exp(-tau_nu(z)) dz  [+ B_nu(T_*) exp(-tau) on core rays]
    tau_nu(z) = Int_z^zmax chi_nu(z') dz'      (z axis points to the observer)

Opacities/emissivities are given in the comoving frame (CMF).  At every
point of every ray the observer frequency nu is mapped to the local CMF
frequency

    nu_cmf = nu * D,     D = gamma * (1 - mu * v(r)/c),   mu = z / r,

and the coefficients are transformed with the Lorentz invariants
(nu*chi and eta/nu^2):

    chi_obs = chi_cmf * D,       eta_obs = eta_cmf / D^2.

The main entry point is :func:`compute_spectrum`.

Parallelisation: the observer-frequency grid is split into chunks that are
processed by all cores (numba ``prange``; a threaded NumPy fallback is used
if numba is not installed -- ``pip install numba`` is strongly recommended).
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import numba as nb
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

# ----------------------------------------------------------------------------
# Physical constants (cgs)
# ----------------------------------------------------------------------------
C_CGS = 2.99792458e10        # cm / s
C_ANG = 2.99792458e18        # Angstrom / s
H_PLANCK = 6.62607015e-27    # erg s
K_BOLTZ = 1.380649e-16       # erg / K
SIGMA_SB = 5.670374419e-5    # erg cm^-2 s^-1 K^-4
L_SUN = 3.828e33             # erg / s
KPC = 3.0856775814913673e21  # cm


# ----------------------------------------------------------------------------
# Ray geometry
# ----------------------------------------------------------------------------
def _refine_z(z, p, r, v, dv_max, dlnr_max, max_sub):
    """Subdivide a ray so that the line-of-sight velocity changes by less
    than dv_max (resolves Doppler-shifted lines) and ln r by less than
    dlnr_max (resolves the density stratification) between consecutive points."""
    rr = np.sqrt(p * p + z * z)
    vlos = np.interp(rr, r, v) * z / np.maximum(rr, 1e-300)
    nsub = np.ceil(np.maximum(np.abs(np.diff(vlos)) / dv_max,
                              np.abs(np.diff(np.log(rr))) / dlnr_max)).astype(np.int64)
    nsub = np.clip(nsub, 1, max_sub)
    if nsub.max() == 1:
        return z
    seg = np.repeat(np.arange(nsub.size), nsub)
    first = np.cumsum(nsub) - nsub
    frac = (np.arange(seg.size) - first[seg]) / nsub[seg]
    zf = z[seg] + (z[seg + 1] - z[seg]) * frac
    return np.append(zf, z[-1])


def _build_rays(r, v, n_core, n_extra, dv_max, dlnr_max, max_sub, ls_chi, ls_eta):
    """Build impact parameters, quadrature weights and all ray points.

    Rays are stored point by point, ordered from the observer side inwards
    (decreasing z), concatenated in flat arrays.
    """
    N = r.size
    R_star, r_max = r[0], r[-1]
    lnr = np.log(r)

    # core rays: uniform in mu -> denser towards the limb
    mu_c = np.linspace(1.0, 0.0, n_core)
    p_core = R_star * np.sqrt(1.0 - mu_c**2)
    p_core[-1] = R_star * (1.0 - 1e-10)       # last ray just inside the limb
    # envelope rays: tangent to every shell + n_extra rays in between
    pieces = [np.geomspace(r[i], r[i + 1], n_extra + 2)[:-1] for i in range(N - 1)]
    # rays converging geometrically on the outer edge (I drops to 0 there)
    edge = r_max - (r_max - pieces[-1][-1]) * 0.5 ** np.arange(1, 13)
    p_env = np.concatenate(pieces + [edge, r[-1:]])
    p_all = np.concatenate([p_core, p_env])
    is_core = np.zeros(p_all.size, dtype=np.bool_)
    is_core[:n_core] = True

    # trapezoidal weights for Int p I(p) dp  (the step at p = R* is bracketed
    # by two nearly coincident rays, so it is integrated exactly)
    dp = np.diff(p_all)
    w = np.zeros_like(p_all)
    w[:-1] += 0.5 * dp
    w[1:] += 0.5 * dp
    wp = w * p_all

    starts, lens = [], []
    ir_l, wr_l, lnD_l, dz_l, fc_l, fe_l = [], [], [], [], [], []
    core_lnD = np.zeros(p_all.size)
    off = 0
    for j, p in enumerate(p_all):
        if is_core[j]:
            z = np.sqrt(np.maximum(r[::-1] ** 2 - p * p, 0.0))       # r_max ... R*
        else:
            zp = np.sqrt(r[r > p][::-1] ** 2 - p * p)
            z = np.concatenate([zp, [0.0], -zp[::-1]])
        if z.size >= 2:
            z = _refine_z(z, p, r, v, dv_max, dlnr_max, max_sub)
        rr = np.sqrt(p * p + z * z)
        rr = np.clip(rr, R_star, r_max)
        ir = np.clip(np.searchsorted(r, rr, side="right") - 1, 0, N - 2)
        wr = np.clip((np.log(rr) - lnr[ir]) / (lnr[ir + 1] - lnr[ir]), 0.0, 1.0)
        beta = np.interp(rr, r, v) / C_CGS
        D = (1.0 - beta * z / rr) / np.sqrt(1.0 - beta * beta)
        lnD = np.log(D)
        dz = np.append(np.abs(np.diff(z)), 0.0)
        if is_core[j]:
            core_lnD[j] = lnD[-1]
        starts.append(off)
        lens.append(z.size)
        off += z.size
        ir_l.append(ir); wr_l.append(wr); lnD_l.append(lnD); dz_l.append(dz)
        # Lorentz factors times the radial scale (power-law interpolated)
        fc_l.append(D * np.exp((1 - wr) * ls_chi[ir] + wr * ls_chi[ir + 1]))
        fe_l.append(np.exp((1 - wr) * ls_eta[ir] + wr * ls_eta[ir + 1]) / (D * D))

    return dict(
        wp=wp, is_core=is_core, core_lnD=core_lnD,
        start=np.asarray(starts, dtype=np.int64), length=np.asarray(lens, dtype=np.int64),
        ir=np.concatenate(ir_l).astype(np.int64), wr=np.concatenate(wr_l),
        lnD=np.concatenate(lnD_l), dz=np.concatenate(dz_l),
        fc=np.concatenate(fc_l), fe=np.concatenate(fe_l),
    )


# ----------------------------------------------------------------------------
# numba kernel
# ----------------------------------------------------------------------------
if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _eval_point(lnu_g, chi, eta, lnu_obs, k0, nk, ir, wr, lnD, fc, fe, outc, oute):
        """chi, eta (observer frame) at one ray point for a chunk of observer
        frequencies.  The CMF index is advanced by a pointer walk, because the
        CMF frequency is monotonic in the observer frequency."""
        nf = lnu_g.size
        wr1 = 1.0 - wr
        idx = np.searchsorted(lnu_g, lnu_obs[k0] + lnD) - 1
        if idx < 0:
            idx = 0
        elif idx > nf - 2:
            idx = nf - 2
        for k in range(nk):
            t = lnu_obs[k0 + k] + lnD
            while idx < nf - 2 and lnu_g[idx + 1] <= t:
                idx += 1
            wf = (t - lnu_g[idx]) / (lnu_g[idx + 1] - lnu_g[idx])
            if wf < 0.0:
                wf = 0.0
            elif wf > 1.0:
                wf = 1.0
            c = (1.0 - wf) * (wr1 * chi[idx, ir] + wr * chi[idx, ir + 1]) \
                + wf * (wr1 * chi[idx + 1, ir] + wr * chi[idx + 1, ir + 1])
            e = (1.0 - wf) * (wr1 * eta[idx, ir] + wr * eta[idx, ir + 1]) \
                + wf * (wr1 * eta[idx + 1, ir] + wr * eta[idx + 1, ir + 1])
            outc[k] = c * fc
            oute[k] = e * fe

    @njit(parallel=True, cache=True)
    def _kernel_numba(lnu_g, chi, eta, lnu_obs, nu_obs, kb,
                      ray_start, ray_len, ray_core, wp, core_lnD,
                      pt_ir, pt_wr, pt_lnD, pt_dz, pt_fc, pt_fe, T_star, out):
        nchunk = kb.size - 1
        nray = ray_start.size
        two_h_c2 = 2.0 * H_PLANCK / C_CGS**2
        h_kT = H_PLANCK / (K_BOLTZ * T_star)
        for ic in prange(nchunk):
            k0 = kb[ic]
            nk = kb[ic + 1] - k0
            tau = np.empty(nk)
            inten = np.empty(nk)
            ca = np.empty(nk); ea = np.empty(nk)
            cb = np.empty(nk); eb = np.empty(nk)
            floc = np.zeros(nk)
            for j in range(nray):
                n = ray_len[j]
                if n < 2:
                    continue
                s = ray_start[j]
                for k in range(nk):
                    tau[k] = 0.0
                    inten[k] = 0.0
                _eval_point(lnu_g, chi, eta, lnu_obs, k0, nk,
                            pt_ir[s], pt_wr[s], pt_lnD[s], pt_fc[s], pt_fe[s], ca, ea)
                for m in range(1, n):
                    q = s + m
                    _eval_point(lnu_g, chi, eta, lnu_obs, k0, nk,
                                pt_ir[q], pt_wr[q], pt_lnD[q], pt_fc[q], pt_fe[q], cb, eb)
                    dz = pt_dz[q - 1]
                    for k in range(nk):
                        # eta linear in z, chi = segment mean -> analytic integral
                        x = 0.5 * (ca[k] + cb[k]) * dz
                        if abs(x) < 1e-3:
                            w0 = 1.0 - 0.5 * x + x * x / 6.0
                            w1 = 0.5 - x / 3.0 + 0.125 * x * x
                        else:
                            ex = np.exp(-x)
                            w0 = (1.0 - ex) / x
                            w1 = (1.0 - (1.0 + x) * ex) / (x * x)
                        inten[k] += np.exp(-tau[k]) * dz * (ea[k] * w0 + (eb[k] - ea[k]) * w1)
                        tau[k] += x
                    ca, cb = cb, ca
                    ea, eb = eb, ea
                if ray_core[j]:
                    # opaque star: blackbody surface, Doppler-shifted (I/nu^3 invariant)
                    D = np.exp(core_lnD[j])
                    for k in range(nk):
                        nuc = nu_obs[k0 + k] * D
                        xb = h_kT * nuc
                        if xb < 700.0:
                            B = two_h_c2 * nuc**3 / np.expm1(xb)
                            inten[k] += np.exp(-tau[k]) * B / D**3
                w = wp[j]
                for k in range(nk):
                    floc[k] += w * inten[k]
            for k in range(nk):
                out[k0 + k] = floc[k]


# ----------------------------------------------------------------------------
# NumPy fallback (vectorised over frequency, threads over chunks)
# ----------------------------------------------------------------------------
def _chunk_numpy(args):
    (lnu_g, chi, eta, lnu_obs, nu_obs, k0, k1, rays, T_star) = args
    nf = lnu_g.size
    lo = lnu_obs[k0:k1]
    nu = nu_obs[k0:k1]
    floc = np.zeros(k1 - k0)

    def ev(q):
        ir, wr, lnD = rays["ir"][q], rays["wr"][q], rays["lnD"][q]
        t = lo + lnD
        idx = np.clip(np.searchsorted(lnu_g, t, side="right") - 1, 0, nf - 2)
        wf = np.clip((t - lnu_g[idx]) / (lnu_g[idx + 1] - lnu_g[idx]), 0.0, 1.0)
        c = (1 - wf) * ((1 - wr) * chi[idx, ir] + wr * chi[idx, ir + 1]) \
            + wf * ((1 - wr) * chi[idx + 1, ir] + wr * chi[idx + 1, ir + 1])
        e = (1 - wf) * ((1 - wr) * eta[idx, ir] + wr * eta[idx, ir + 1]) \
            + wf * ((1 - wr) * eta[idx + 1, ir] + wr * eta[idx + 1, ir + 1])
        return c * rays["fc"][q], e * rays["fe"][q]

    for j in range(rays["start"].size):
        n, s = rays["length"][j], rays["start"][j]
        if n < 2:
            continue
        tau = np.zeros_like(lo)
        inten = np.zeros_like(lo)
        ca, ea = ev(s)
        for m in range(1, n):
            cb, eb = ev(s + m)
            dz = rays["dz"][s + m - 1]
            x = 0.5 * (ca + cb) * dz
            small = np.abs(x) < 1e-3
            xs = np.where(small, 1.0, x)
            ex = np.exp(-xs)
            w0 = np.where(small, 1 - 0.5 * x + x * x / 6, (1 - ex) / xs)
            w1 = np.where(small, 0.5 - x / 3 + 0.125 * x * x, (1 - (1 + xs) * ex) / (xs * xs))
            inten += np.exp(-tau) * dz * (ea * w0 + (eb - ea) * w1)
            tau += x
            ca, ea = cb, eb
        if rays["is_core"][j]:
            D = np.exp(rays["core_lnD"][j])
            xb = np.minimum(H_PLANCK * nu * D / (K_BOLTZ * T_star), 700.0)
            B = 2 * H_PLANCK * (nu * D) ** 3 / C_CGS**2 / np.expm1(xb)
            inten += np.exp(-tau) * B / D**3
        floc += rays["wp"][j] * inten
    return k0, k1, floc


def _radial_scale(a):
    """log of a positive per-radius scale (frequency mean) and the scaled array."""
    m = np.abs(a).mean(axis=0)
    m = np.where(m > 0, m, 1.0)
    return np.log(m), np.ascontiguousarray(a / m[None, :])


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def compute_spectrum(frequencies, chi_selected, eta_selected, r_vector, v_wind, L_star,
                     distance_kpc=1.0, nu_obs=None,
                     r_unit=1.0, v_unit=1.0e5, nu_unit=1.0, chi_unit=1.0, eta_unit=1.0,
                     n_core=40, n_extra=2, dv_max_kms=None, dlnr_max=0.03, max_sub=400,
                     n_threads=None, clip_negative_chi=True, return_fnu=False):
    """Observer-frame formal integral -> emergent spectrum of a stellar wind.

    Parameters
    ----------
    frequencies : (Nfreq,) array
        Comoving-frame frequencies (descending, as produced by the model;
        ascending input is also accepted).  Units: Hz * ``nu_unit``.
    chi_selected, eta_selected : (Nfreq, N) arrays
        CMF absorption coefficient [cm^-1 * chi_unit] and emission coefficient
        [erg s^-1 cm^-3 Hz^-1 sr^-1 * eta_unit].
    r_vector : (N,) array
        Radii, increasing; ``r_vector[0]`` is the stellar (lower-boundary)
        radius.  Units: cm * ``r_unit`` (e.g. r_unit=6.957e10 for R_sun).
    v_wind : (N,) array
        Radial wind velocity.  Units: cm/s * ``v_unit`` (default: km/s).
    L_star : float
        Stellar luminosity in L_sun; T_* = (L / (4 pi R*^2 sigma))^(1/4).
    distance_kpc : float
        Distance to the observer (default 1 kpc).
    nu_obs : array, optional
        Observer-frame frequencies [Hz * nu_unit].  Default: the CMF grid
        restricted to the range that stays inside the CMF grid for every
        Doppler shift occurring in the wind.
    n_core : int
        Number of rays hitting the stellar disc.
    n_extra : int
        Additional impact parameters between consecutive shells.
    dv_max_kms : float, optional
        Maximum change of the line-of-sight velocity between two integration
        points on a ray.  Should be smaller than the intrinsic (thermal +
        turbulent) line width.  Default: 0.5 x the median velocity spacing of
        the CMF frequency grid.
    dlnr_max : float
        Maximum step in ln(r) between two integration points on a ray.
    max_sub : int
        Upper limit on the subdivision of one shell-to-shell ray segment.
    n_threads : int, optional
        Number of threads (default: all cores).
    clip_negative_chi : bool
        Set negative opacities (masing transitions) to zero.
    return_fnu : bool
        Additionally return observer frequencies [Hz] and F_nu.

    Returns
    -------
    wavelength : (Nobs,) array   -- Angstrom, ascending
    flux       : (Nobs,) array   -- F_lambda in erg s^-1 cm^-2 A^-1 at distance d
    [nu, F_nu] : if ``return_fnu`` -- Hz, erg s^-1 cm^-2 Hz^-1
    """
    # ---- input, units, ordering ------------------------------------------
    nu_g = np.asarray(frequencies, dtype=np.float64) * nu_unit
    r = np.asarray(r_vector, dtype=np.float64) * r_unit
    v = np.asarray(v_wind, dtype=np.float64) * v_unit
    chi = np.asarray(chi_selected, dtype=np.float64)
    eta = np.asarray(eta_selected, dtype=np.float64)
    if chi.shape != (nu_g.size, r.size) or eta.shape != chi.shape:
        raise ValueError("chi_selected / eta_selected must have shape (Nfreq, N)")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r_vector must be strictly increasing (r_vector[0] = stellar radius)")
    if nu_g.size < 2 or r.size < 2:
        raise ValueError("need at least two frequencies and two radial points")
    dnu = np.diff(nu_g)
    if np.all(dnu < 0):                       # descending -> ascending
        nu_g, chi, eta = nu_g[::-1], chi[::-1], eta[::-1]
    elif not np.all(dnu > 0):
        raise ValueError("frequencies must be strictly monotonic")
    chi = np.ascontiguousarray(chi) * chi_unit
    eta = np.ascontiguousarray(eta) * eta_unit
    if clip_negative_chi and chi.min() < 0:
        warnings.warn("negative opacities clipped to zero")
        chi = np.maximum(chi, 0.0)
    lnu_g = np.log(nu_g)

    # ---- star ---------------------------------------------------------------
    R_star = r[0]
    T_star = (L_star * L_SUN / (4.0 * np.pi * R_star**2 * SIGMA_SB)) ** 0.25
    d = distance_kpc * KPC

    # ---- observer frequency grid ---------------------------------------------
    beta_max = np.max(np.abs(v)) / C_CGS
    if beta_max >= 1:
        raise ValueError("|v| >= c: check v_unit")
    if nu_obs is None:
        gam = 1.0 / np.sqrt(1.0 - beta_max**2)
        lo = nu_g[0] / (gam * (1.0 - beta_max))      # nu*D >= nu_g[0] for all D
        hi = nu_g[-1] / (gam * (1.0 + beta_max))     # nu*D <= nu_g[-1]
        nuo = nu_g[(nu_g >= lo) & (nu_g <= hi)]
        if nuo.size == 0:
            raise ValueError("CMF frequency grid is narrower than the wind Doppler width")
    else:
        nuo = np.sort(np.asarray(nu_obs, dtype=np.float64) * nu_unit)
    lnu_obs = np.log(nuo)

    # ---- rays ---------------------------------------------------------------
    if dv_max_kms is None:
        dv_max = 0.5 * C_CGS * np.median(np.diff(lnu_g))
    else:
        dv_max = dv_max_kms * 1.0e5
    # Remove the bulk radial run (e.g. ~rho or rho^2) before interpolating:
    # chi = s(r) * chi_n(nu, r); s is interpolated as a power law in r, chi_n
    # linearly.  This keeps the interpolation accurate on coarse radial grids.
    ls_chi, chi = _radial_scale(chi)
    ls_eta, eta = _radial_scale(eta)
    rays = _build_rays(r, v, int(n_core), int(n_extra), dv_max, float(dlnr_max), int(max_sub), ls_chi, ls_eta)

    # ---- solve --------------------------------------------------------------
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    nobs = nuo.size
    nchunk = int(min(max(1, nobs // 64), 8 * n_threads))
    kb = np.linspace(0, nobs, nchunk + 1).astype(np.int64)
    acc = np.zeros(nobs)

    if HAVE_NUMBA:
        old = nb.get_num_threads()
        nb.set_num_threads(int(min(n_threads, nb.config.NUMBA_NUM_THREADS)))
        try:
            _kernel_numba(lnu_g, chi, eta, lnu_obs, nuo, kb,
                          rays["start"], rays["length"], rays["is_core"], rays["wp"],
                          rays["core_lnD"], rays["ir"], rays["wr"], rays["lnD"], rays["dz"],
                          rays["fc"], rays["fe"], T_star, acc)
        finally:
            nb.set_num_threads(old)
    else:
        warnings.warn("numba not found: using the (much slower) NumPy fallback")
        jobs = [(lnu_g, chi, eta, lnu_obs, nuo, kb[i], kb[i + 1], rays, T_star)
                for i in range(nchunk)]
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for k0, k1, f in ex.map(_chunk_numpy, jobs):
                acc[k0:k1] = f

    # ---- fluxes -------------------------------------------------------------
    f_nu = 2.0 * np.pi / d**2 * acc                  # erg s^-1 cm^-2 Hz^-1
    wavelength = C_ANG / nuo                         # Angstrom
    f_lam = f_nu * C_ANG / wavelength**2             # erg s^-1 cm^-2 A^-1
    order = np.argsort(wavelength)
    wavelength, f_lam = wavelength[order], f_lam[order]
    if return_fnu:
        return wavelength, f_lam, nuo[order], f_nu[order]
    return wavelength, f_lam
