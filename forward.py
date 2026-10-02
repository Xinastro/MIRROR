"""
MIRROR preliminary study: reduced reflected-light forward model.

NOT a line-by-line radiative transfer code. Band opacities are schematic
Gaussian profiles placed at published band centres, with peak optical
depths tuned so that a modern-Earth reference case reproduces the
approximate band depths of the disk-integrated terrestrial geometric
albedo spectrum. The purpose is to expose the *inference* problem
(which molecules are supported by the data) at negligible compute cost,
in the same way an analytic transmission model is used for rapid
pipeline development. Production work uses rfast + pyEDITH.

Abundance parameter convention: x_i = log10(column scaling relative to
the modern-Earth reference for that species). x = 0 means Earth-like.
"""

import numpy as np

# ----------------------------------------------------------------------
# Molecular band library
# Each entry: list of (centre_um, sigma_um, peak_tau_at_reference)
# ----------------------------------------------------------------------

BANDS = {
    "O2": [
        (0.6280, 0.0035, 0.020),
        (0.6880, 0.0040, 0.120),
        (0.7620, 0.0045, 0.750),
        (1.2700, 0.0090, 0.100),
    ],
    "O3": [
        (0.2550, 0.0280, 60.00),   # Hartley
        (0.3250, 0.0180, 0.600),   # Huggins
        (0.6000, 0.0700, 0.055),   # Chappuis
    ],
    "H2O": [
        (0.7200, 0.0090, 0.050),
        (0.8200, 0.0130, 0.120),
        (0.9400, 0.0220, 0.450),
        (1.1300, 0.0300, 0.650),
        (1.4100, 0.0450, 2.200),
    ],
    "CH4": [
        (0.7900, 0.0080, 0.004),
        (0.8890, 0.0110, 0.010),
        (1.0000, 0.0150, 0.012),
        (1.1650, 0.0200, 0.030),
        (1.4000, 0.0300, 0.020),
        (1.6700, 0.0400, 0.090),
    ],
    "CO2": [
        (1.0500, 0.0150, 0.002),
        (1.2100, 0.0180, 0.004),
        (1.4400, 0.0250, 0.012),
        (1.6000, 0.0300, 0.045),
    ],
    # --- species absent from the truth; spectrally confusable ---
    "CO": [
        (1.5700, 0.0180, 0.350),   # overlaps the CO2 1.6 um region
    ],
    "SO2": [
        (0.2900, 0.0250, 8.000),   # overlaps O3 Hartley/Huggins
        (0.3200, 0.0150, 1.500),
    ],
    "NO2": [
        (0.4200, 0.0600, 0.300),   # overlaps O3 Chappuis shortward
        (0.5000, 0.0500, 0.150),
    ],
    "N2O": [
        (1.5200, 0.0200, 0.200),   # overlaps CO2/CO/CH4 near 1.5-1.7
        (1.6600, 0.0250, 0.120),
    ],
}

SPECIES = list(BANDS.keys())
N_SPECIES = len(SPECIES)

# Proterozoic-like Earth reference state.
# x = log10 scaling relative to modern Earth for that species.
TRUTH_X = {
    "O2":  -1.0,   # ~10% PAL
    "O3":  -0.7,
    "H2O":  0.0,
    "CH4":  1.0,   # elevated relative to modern
    "CO2":  0.7,
}
TRUTH_SET = tuple(sorted(TRUTH_X.keys()))

# Continuum parameters (Rayleigh optical depth at 0.55 um, effective
# scattering albedo of the cloud/surface layer)
TRUTH_CONT = {"tau_r0": 0.10, "a_eff": 0.33}

X_PRIOR = (-3.0, 3.0)          # log10 scaling if present, uniform
TAUR_PRIOR = (0.01, 0.60)      # uniform
AEFF_PRIOR = (0.05, 0.80)      # uniform

AIRMASS = 2.0                  # crude double-pass factor


def species_opacity(name, wl):
    """Reference-column optical depth profile tau_ref(lambda) for one species."""
    tau = np.zeros_like(wl)
    for c, s, amp in BANDS[name]:
        tau += amp * np.exp(-0.5 * ((wl - c) / s) ** 2)
    return tau


def build_templates(wl):
    """Precompute tau_ref(lambda) for every species. Shape (N_SPECIES, n_wl)."""
    return np.array([species_opacity(n, wl) for n in SPECIES])


def continuum_albedo(wl, tau_r0, a_eff):
    """
    Two-stream-like continuum geometric albedo: conservative Rayleigh layer
    over a Lambertian cloud/surface with effective albedo a_eff.
    """
    tau_r = tau_r0 * (wl / 0.55) ** -4.0
    r_ray = tau_r / (tau_r + 4.0 / 3.0)             # Rayleigh layer reflectance
    trans = 1.0 - r_ray
    return 0.5 * (r_ray + trans ** 2 * a_eff / (1.0 - r_ray * a_eff))


def albedo(wl, templates, x_vec, mask, tau_r0, a_eff):
    """
    Geometric albedo for an included subset.

    templates : (N_SPECIES, n_wl) reference opacities
    x_vec     : (N_SPECIES,) log10 scalings (values where mask is False ignored)
    mask      : (N_SPECIES,) boolean inclusion vector
    """
    cont = continuum_albedo(wl, tau_r0, a_eff)
    if mask.any():
        scal = np.where(mask, 10.0 ** x_vec, 0.0)
        tau = scal @ templates
    else:
        tau = np.zeros_like(wl)
    return cont * np.exp(-AIRMASS * tau)


def truth_albedo(wl, templates):
    x = np.zeros(N_SPECIES)
    mask = np.zeros(N_SPECIES, dtype=bool)
    for i, n in enumerate(SPECIES):
        if n in TRUTH_X:
            x[i] = TRUTH_X[n]
            mask[i] = True
    return albedo(wl, templates, x, mask, **TRUTH_CONT)


# ----------------------------------------------------------------------
# Observing simulation
# ----------------------------------------------------------------------

def wavelength_grid(lam_min=0.20, lam_max=1.80, R=140.0):
    """Constant-resolving-power grid."""
    wl = [lam_min]
    while wl[-1] < lam_max:
        wl.append(wl[-1] * (1.0 + 1.0 / R))
    return np.array(wl[:-1])


def observe(wl, templates, snr_ref, rng, lam_ref=0.55):
    """
    Generate a noisy spectrum. snr_ref is the per-bin SNR at lam_ref;
    noise per bin scales as sqrt(albedo) (photon-limited on the planet)
    with a floor to avoid divergence in deep bands.
    """
    a_true = truth_albedo(wl, templates)
    a_ref = np.interp(lam_ref, wl, a_true)
    sigma = np.sqrt(np.maximum(a_true, 0.02 * a_ref) * a_ref) / snr_ref
    data = a_true + rng.normal(0.0, sigma)
    return data, sigma, a_true
