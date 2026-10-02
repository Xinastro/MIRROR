"""
PROTOTYPE trans-dimensional inference over molecular inclusion.

SCOPE.  This is the preliminary prototype that produced the Preliminary Results section,
not the production Route B implementation.  The proposal's Route B is exact enumeration
for N <= 12 with reversible-jump MCMC above that, on a 512-point quadrature validated to
0.03 nat against an 8000-point reference.  This file instead uses collapsed Gibbs over the
inclusion indicators on a coarse grid (default 80 points).  It is a test bed: it
demonstrates the mechanism and was the vehicle for the three implementation faults the
simulation-based calibration gate caught.  Do not read its defaults as production
settings.

Sampler: collapsed Gibbs over the inclusion indicators z_i. Conditional on
the continuum parameters and all other species, the joint conditional for
(z_i, x_i) is obtained by 1-D quadrature of the likelihood over x_i on a
fixed grid. It has no acceptance-rate tuning and mixes far better than
birth-death RJMCMC at this dimension, which is why it serves as the
prototype stand-in while Route B is built.

Quadrature grid: CELL-CENTERED, x_j = x_min + (j + 1/2) * dx with
dx = (x_max - x_min)/N. A grid inclusive of both endpoints, paired with a
rectangular rule of weight dx per node, integrates the prior to
(x_max - x_min) * N/(N-1) rather than (x_max - x_min), overcounting the
prior mass by 1/(N-1); and jittering by +/- dx/2 about an endpoint node can
place an abundance outside X_PRIOR. Cell centering removes both.

Prior on z: hierarchical Beta-Binomial (a=b=1), i.e. the expected number
of included species is itself inferred. This supplies automatic
multiplicity control rather than a post-hoc correction (Scott & Berger 2010).
"""

import numpy as np
from scipy.special import gammaln, logsumexp

import forward as F


def log_prior_z(k, n):
    """Beta-Binomial(1,1) prior on the inclusion vector: 1/((n+1) C(n,k))."""
    log_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return -np.log(n + 1) - log_binom


class Sampler:
    def __init__(self, wl, data, sigma, templates, n_grid=80, rng=None):
        self.wl = wl
        self.data = data
        self.inv_var = 1.0 / sigma ** 2
        self.templates = templates
        self.n = templates.shape[0]
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # cell-centered: N cells of width dx spanning X_PRIOR exactly, so the
        # rectangular rule integrates the prior to 1 and a +/- dx/2 jitter stays inside.
        lo, hi = F.X_PRIOR
        self.dx = (hi - lo) / n_grid
        self.xgrid = lo + (np.arange(n_grid) + 0.5) * self.dx
        self.log_prior_x = -np.log(F.X_PRIOR[1] - F.X_PRIOR[0])

    def _loglike_from_tau(self, tau, tau_r0, a_eff):
        """tau may be (n_wl,) or (m, n_wl)."""
        cont = F.continuum_albedo(self.wl, tau_r0, a_eff)
        model = cont * np.exp(-F.AIRMASS * tau)
        resid = self.data - model
        return -0.5 * np.sum(resid ** 2 * self.inv_var, axis=-1)

    def run(self, n_sweep=4000, burn=1000, thin=2, fixed_mask=None):
        n = self.n
        rng = self.rng

        # initialise
        if fixed_mask is None:
            mask = np.zeros(n, dtype=bool)
        else:
            mask = np.asarray(fixed_mask, dtype=bool).copy()
        x = rng.uniform(*F.X_PRIOR, size=n)
        tau_r0 = F.TRUTH_CONT["tau_r0"]
        a_eff = F.TRUTH_CONT["a_eff"]
        step = np.array([0.006, 0.010])

        scal = np.where(mask, 10.0 ** x, 0.0)
        tau_tot = scal @ self.templates

        z_chain, x_chain, c_chain = [], [], []
        n_acc = 0

        for it in range(n_sweep):
            # ---- inclusion / abundance updates -------------------------
            for i in rng.permutation(n):
                contrib = (10.0 ** x[i]) * self.templates[i] if mask[i] else 0.0
                tau_rest = tau_tot - contrib

                k_wo = int(mask.sum()) - (1 if mask[i] else 0)

                # z_i = 0 branch
                ll_out = self._loglike_from_tau(tau_rest, tau_r0, a_eff)
                log_w_out = ll_out + log_prior_z(k_wo, n)

                # z_i = 1 branch: 1-D quadrature over x_i
                tau_in = tau_rest[None, :] + (10.0 ** self.xgrid)[:, None] * self.templates[i][None, :]
                ll_in = self._loglike_from_tau(tau_in, tau_r0, a_eff)
                log_marg = logsumexp(ll_in + self.log_prior_x) + np.log(self.dx)
                log_w_in = log_marg + log_prior_z(k_wo + 1, n)

                if fixed_mask is not None:
                    take_in = bool(fixed_mask[i])
                else:
                    p_in = 1.0 / (1.0 + np.exp(np.clip(log_w_out - log_w_in, -700, 700)))
                    take_in = rng.random() < p_in

                if take_in:
                    logp = ll_in - ll_in.max()
                    p = np.exp(logp)
                    p /= p.sum()
                    x[i] = rng.choice(self.xgrid, p=p) + rng.uniform(-0.5, 0.5) * self.dx
                    mask[i] = True
                    tau_tot = tau_rest + (10.0 ** x[i]) * self.templates[i]
                else:
                    mask[i] = False
                    tau_tot = tau_rest

            # ---- continuum update (random-walk Metropolis) -------------
            prop = np.array([tau_r0, a_eff]) + rng.normal(0, step)
            if (F.TAUR_PRIOR[0] < prop[0] < F.TAUR_PRIOR[1]
                    and F.AEFF_PRIOR[0] < prop[1] < F.AEFF_PRIOR[1]):
                ll_new = self._loglike_from_tau(tau_tot, prop[0], prop[1])
                ll_old = self._loglike_from_tau(tau_tot, tau_r0, a_eff)
                if np.log(rng.random()) < ll_new - ll_old:
                    tau_r0, a_eff = prop
                    n_acc += 1
            if it < burn and it > 0 and it % 100 == 0:
                r = n_acc / it
                step *= 1.6 if r > 0.4 else (0.6 if r < 0.15 else 1.0)

            if it >= burn and (it - burn) % thin == 0:
                z_chain.append(mask.copy())
                x_chain.append(x.copy())
                c_chain.append([tau_r0, a_eff])

        self.acc_rate = n_acc / n_sweep
        return (np.array(z_chain), np.array(x_chain), np.array(c_chain))


def pip(z_chain):
    """Posterior inclusion probabilities."""
    return z_chain.mean(axis=0)


def bayesian_fdr(pips, threshold):
    """
    Expected false discovery rate among species called at >= threshold.
    E[FDR] = sum_{called} (1 - PIP_i) / n_called.
    """
    called = pips >= threshold
    if called.sum() == 0:
        return 0.0, called
    return float(np.mean(1.0 - pips[called])), called


def fdr_threshold(pips, alpha=0.05):
    """Largest call set whose expected FDR stays at or below alpha."""
    order = np.argsort(-pips)
    best = np.zeros_like(pips, dtype=bool)
    for m in range(1, len(pips) + 1):
        sel = order[:m]
        if np.mean(1.0 - pips[sel]) <= alpha:
            best = np.zeros_like(pips, dtype=bool)
            best[sel] = True
        else:
            break
    return best
