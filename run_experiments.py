"""Run all MIRROR preliminary experiments and cache results to .npz."""
import sys, time
import numpy as np
import forward as F
import inference as I

WL = F.wavelength_grid(R=140.0)
TPL = F.build_templates(WL)
NSW, NBURN = 800, 300


def one(keep_idx, snr, seed, fixed=None, truth_x=None, truth_mask=None):
    rng = np.random.default_rng(seed)
    if truth_x is None:
        d, s, a = F.observe(WL, TPL, snr, rng)
    else:
        a = F.albedo(WL, TPL, truth_x, truth_mask, **F.TRUTH_CONT)
        aref = np.interp(0.55, WL, a)
        s = np.sqrt(np.maximum(a, 0.02 * aref) * aref) / snr
        d = a + rng.normal(0, s)
    smp = I.Sampler(WL, d, s, TPL[keep_idx], rng=rng)
    z, x, c = smp.run(n_sweep=NSW, burn=NBURN, fixed_mask=fixed)
    p = I.pip(z)
    mask = p > 0.5
    mdl = F.albedo(WL, TPL[keep_idx], x.mean(0), mask, *c.mean(0))
    chi2 = np.sum(((d - mdl) / s) ** 2) / max(len(WL) - mask.sum() - 2, 1)
    xm = np.full(len(keep_idx), np.nan)
    xs = np.full(len(keep_idx), np.nan)
    for j in range(len(keep_idx)):
        sel = z[:, j]
        if sel.sum() > 40:
            xm[j], xs[j] = x[sel, j].mean(), x[sel, j].std()
    return p, chi2, xm, xs


ALL = list(range(F.N_SPECIES))
TRUTH_MASK = np.array([n in F.TRUTH_X for n in F.SPECIES])
TRUTH_XV = np.array([F.TRUTH_X.get(n, 0.0) for n in F.SPECIES])


def exp_ablation(nreal=8):
    out = {}
    for drop in [None, "CO2", "O3", "CH4"]:
        keep = [i for i in ALL if drop is None or F.SPECIES[i] != drop]
        P, C = [], []
        for k in range(nreal):
            p, c2, _, _ = one(keep, 30.0, 1000 + k)
            P.append(p); C.append(c2)
        key = drop or "none"
        out[f"abl_{key}_pip"] = np.array(P)
        out[f"abl_{key}_chi2"] = np.array(C)
        out[f"abl_{key}_names"] = np.array([F.SPECIES[i] for i in keep])
        print(f"  ablation {key} done", flush=True)
    return out


def exp_floor(snrs, nreal=8):
    P = np.zeros((len(snrs), nreal, F.N_SPECIES))
    XS = np.zeros((len(snrs), nreal, F.N_SPECIES))
    XSF = np.zeros((len(snrs), nreal, F.N_SPECIES))
    for a, snr in enumerate(snrs):
        for k in range(nreal):
            p, _, _, xs = one(ALL, snr, 2000 + 50 * a + k)
            P[a, k] = p; XS[a, k] = xs
            _, _, _, xsf = one(ALL, snr, 2000 + 50 * a + k, fixed=TRUTH_MASK)
            XSF[a, k] = xsf
        print(f"  floor snr={snr} done", flush=True)
    return {"floor_snr": np.array(snrs), "floor_pip": P,
            "floor_xs_open": XS, "floor_xs_fixed": XSF}


def exp_calibration_range(lo, hi, snr=30.0):
    pips, truths = [], []
    for k in range(lo, hi):
        rng = np.random.default_rng(90000 + k)
        # draw z from the SAME Beta-Binomial(1,1) prior the sampler assumes:
        # k_true uniform on {0..N}, then a uniform random subset of that size
        k_true = rng.integers(0, F.N_SPECIES + 1)
        m = np.zeros(F.N_SPECIES, dtype=bool)
        if k_true > 0:
            m[rng.choice(F.N_SPECIES, size=k_true, replace=False)] = True
        xv = rng.uniform(*F.X_PRIOR, size=F.N_SPECIES)
        p, _, _, _ = one(ALL, snr, 5000 + k, truth_x=xv, truth_mask=m)
        pips.append(p); truths.append(m)
        if (k + 1) % 15 == 0:
            print(f"  calib {k+1}", flush=True)
    return {"cal_pip": np.array(pips), "cal_truth": np.array(truths)}




def exp_ablation_snr(snrs, drop="CO2", nreal=6):
    keep = [i for i in ALL if F.SPECIES[i] != drop]
    names = [F.SPECIES[i] for i in keep]
    P = np.zeros((len(snrs), nreal, len(keep)))
    C = np.zeros((len(snrs), nreal))
    for a, snr in enumerate(snrs):
        for k in range(nreal):
            p, c2, _, _ = one(keep, snr, 7000 + 60 * a + k)
            P[a, k] = p; C[a, k] = c2
        print(f"  ablsnr {snr} done", flush=True)
    return {"as_snr": np.array(snrs), "as_pip": P, "as_chi2": C,
            "as_names": np.array(names)}



def merge(path, new):
    import os
    cur = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    cur.update(new)
    np.savez(path, **cur)
    print("saved", list(new.keys()), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    job = sys.argv[1]
    if job == "abl":
        merge("results.npz", exp_ablation(nreal=6))
    elif job == "floor":
        snrs = [float(v) for v in sys.argv[2].split(",")]
        r = exp_floor(snrs, nreal=6)
        tag = sys.argv[3]
        merge("results.npz", {f"{k}__{tag}": v for k, v in r.items()})
    elif job == "cal":
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
        r = exp_calibration_range(lo, hi)
        merge("results.npz", {f"{k}__{lo}": v for k, v in r.items()})
    elif job == "ablsnr":
        snrs = [float(v) for v in sys.argv[2].split(",")]
        r = exp_ablation_snr(snrs)
        merge("results.npz", {f"{k}__{sys.argv[3]}": v for k, v in r.items()})
    print("chunk %.1f min" % ((time.time() - t0) / 60), flush=True)