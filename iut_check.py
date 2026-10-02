"""
Calibration of the MIRROR pass procedure: marginal and project-level.

PROTOCOL (three stages, in this order, no step revisited after the next begins)

  1. TUNE   Critical rules are searched on tuning data only.  The search target is
            TUNE_FRAC x allocation, declared here before any audit is run, so that the
            frozen rules carry room rather than sitting on their limit.
  2. FREEZE The six critical values are written out and never touched again.
  3. AUDIT  A separate Generator with a different seed draws fresh samples.  The audit
            CANNOT trigger further tuning: its result is reported whatever it is.  If an
            audit component fails, this script prints FAIL and exits non-zero; retuning
            after seeing an audit result would require a predeclared sequential procedure
            and is not done here.

The statistic is a WEIGHTED proportion from a jointly stratified ensemble, so a binomial
critical value at an effective n is not exact.  Critical values come from Monte Carlo test
inversion of the weighted estimator under each boundary null, separately at the power
boundary (p0=0.90, a cell passes when the statistic is high) and the fpr/false-CO boundary
(p0=0.05, it passes when low); those do not share a size.

Every size bound reported is a ONE-SIDED 95% Clopper-Pearson upper bound.  Only the upper
limit governs calibration, so no lower limit and no two-sided interval is quoted.

The end-to-end screen-escalate-retest check is run SEPARATELY at all three boundary types,
not only at the power boundary, and the largest upper bound is the one that must clear.

It is evaluated at the SUPREMUM configuration: one bound sits at the boundary null and the
remaining J-1 bounds clear with probability one.  That is the worst case over all J, so the
reported rate is an upper bound for every J.  Adding further valid components multiplies
the rate by a factor at most one (the architecture must pass all of them), so the rate is
non-increasing in J by construction; earlier versions estimated that factor by simulation,
which is a poor idea because a probability near one raised to the 1727th power is dominated
by its own estimation error.

Design-power margins are computed from the same weighted simulation as the calibration,
using common random numbers so the root find is deterministic and monotone, for each
boundary type.  They retain one approximation, stated in the output and in the proposal:
the J candidate bounds are treated as independent.
"""
import numpy as np, math, json, sys
from scipy.stats import beta
from scipy.optimize import brentq

SEED_TUNE, SEED_AUDIT = 20261004, 777_000_1
ALPHA, NARCH = 0.05, 8
A_MARGINAL = 0.021
A_PROJECT  = ALPHA / NARCH / 2                  # 0.003125 per path
TUNE_FRAC  = 0.70                               # predeclared: tune to 70% of allocation,
#                                               chosen so the audit sample can certify it
SCREEN_PER_STATE, CONF_PER_STATE = 250, 5000
SIG = math.sqrt(math.log(2.0))                  # lognormal weights -> Kish n_eff/n ~ 0.5
NCAL, NAUD_S, NAUD_C, CHUNK = 60_000, 120_000, 50_000, 4_000
NEND, NMARG = 50_000, 6_000
MAX_TUNE_STEPS = 12

BOUNDARY = [("power  ", 0.90, True), ("fpr    ", 0.05, False), ("falseCO", 0.05, False)]
STAGE    = [("screen ", SCREEN_PER_STATE, NAUD_S), ("confirm", CONF_PER_STATE, NAUD_C)]
CLEAN    = {True: 0.9999, False: 0.0001}        # a comfortably compliant bound


def wstat(rng, n, p, nsim, want_neff=False):
    """nsim draws of the weighted proportion from n Bernoulli(p) with lognormal weights."""
    out = np.empty(nsim); nef = np.empty(nsim)
    for a in range(0, nsim, CHUNK):
        b = min(a + CHUNK, nsim); m = b - a
        w = rng.lognormal(-SIG**2 / 2, SIG, size=(m, n)).astype(np.float32)
        y = (rng.random((m, n), dtype=np.float32) < p)
        den = w.sum(1)
        out[a:b] = (w * y).sum(1) / den
        if want_neff: nef[a:b] = den**2 / (w**2).sum(1)
    return (out, nef) if want_neff else out


def wstat_crn(n, tv, nsim, seed):
    """Same statistic, but with common random numbers: deterministic given seed, and
    monotone in tv, so a root find over tv is smooth."""
    g = np.random.default_rng(seed)
    out = np.empty(nsim)
    for a in range(0, nsim, CHUNK):
        b = min(a + CHUNK, nsim); m = b - a
        w = g.lognormal(-SIG**2 / 2, SIG, size=(m, n)).astype(np.float32)
        u = g.random((m, n), dtype=np.float32)
        den = w.sum(1)
        out[a:b] = (w * (u < tv)).sum(1) / den
    return out


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper confidence bound on a binomial proportion."""
    if k >= n: return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def size_and_bound(sample, qv, high, n):
    k = int((sample >= qv).sum() if high else (sample <= qv).sum())
    return k / n, cp_upper(k, n), k


# ---------------------------------------------------------------- stage 1: TUNE
def tune_one(rng, p0, high, n, alloc):
    """Search the nominal target downward until the tuning upper bound clears
    TUNE_FRAC x alloc.  Tuning data only; the audit never enters this loop."""
    target = alloc * TUNE_FRAC
    for step in range(MAX_TUNE_STEPS):
        cal = wstat(rng, n, p0, NCAL)
        qv  = float(np.quantile(cal, 1 - target if high else target))
        tun = wstat(rng, n, p0, NCAL)                       # separate tuning replicates
        _, hi, _ = size_and_bound(tun, qv, high, NCAL)
        if hi <= alloc * TUNE_FRAC:
            return qv, target, step + 1
        target *= 0.82
    return qv, target, MAX_TUNE_STEPS


def tune(rng, alloc, label):
    print(f"\n{label}")
    print(f"  stage 1 TUNE: allocation {alloc:.6f} per path, "
          f"tuning target {TUNE_FRAC:.2f} x allocation = {alloc*TUNE_FRAC:.6f}")
    frozen = {}
    for lab, p0, high in BOUNDARY:
        for st, n, _ in STAGE:
            qv, tgt, steps = tune_one(rng, p0, high, n, alloc)
            frozen[(lab, st)] = dict(q=qv, high=high, p0=p0, n=n, nominal=tgt, steps=steps)
            print(f"    {lab} {st}  nominal {tgt:.6f}  critical value {qv:.6f}  "
                  f"({steps} search step{'s' if steps > 1 else ''})")
    print("  stage 2 FREEZE: the six critical values above are now fixed")
    return frozen


# ---------------------------------------------------------------- stage 3: AUDIT
def audit(rng, frozen, alloc):
    print(f"  stage 3 AUDIT on fresh samples, seed {SEED_AUDIT}, no retuning permitted")
    print(f"    {'boundary':8s} {'stage':8s} {'reps':>8s} {'size':>8s} {'CP U95':>9s} "
          f"{'<=alloc':>8s}")
    rows, worst, ok = {}, 0.0, True
    for lab, p0, high in BOUNDARY:
        for st, n, naud in STAGE:
            f = frozen[(lab, st)]
            s = wstat(rng, n, p0, naud)
            size, hi, k = size_and_bound(s, f["q"], high, naud)
            worst = max(worst, hi); ok &= hi <= alloc
            rows[f"{lab.strip()}/{st.strip()}"] = dict(reps=naud, hits=k, size=size, u95=hi,
                                                       clears=bool(hi <= alloc))
            print(f"    {lab} {st} {naud:8,d} {size:8.4f} {hi:9.4f} "
                  f"{'yes' if hi <= alloc else 'NO':>8s}")
    print(f"    worst component CP U95 {worst:.4f} against allocation {alloc:.6f}: "
          f"{'all clear' if ok else 'FAIL'}")
    return rows, worst, ok


# ------------------------------------------------- assembled procedure, per boundary
def end_to_end(rng, frozen, lab, high, p0, lim):
    """screen-escalate-retest false-pass rate at the least-favourable null for ONE boundary
    type, in the SUPREMUM configuration: this bound sits at the null and every other bound
    clears.  That is the worst case over the number of components J, so it bounds the rate
    for all J.  More valid components can only multiply it by a factor <= 1."""
    fs, fc = frozen[(lab, "screen ")], frozen[(lab, "confirm")]
    bs = wstat(rng, SCREEN_PER_STATE, p0, NEND)
    bc = wstat(rng, CONF_PER_STATE,  p0, NEND)
    bad_ok = ((bs >= fs["q"]) | (bc >= fc["q"])) if high else \
             ((bs <= fs["q"]) | (bc <= fc["q"]))
    k = int(bad_ok.sum())
    r, hi = k / NEND, cp_upper(k, NEND)
    print(f"    {lab} sup over J  false pass {r:.4f}  CP U95 {hi:.4f}  "
          f"vs {lim:.5f}: {'clears' if hi <= lim else 'DOES NOT CLEAR'}")
    return dict(rate=r, u95=hi, hits=k, reps=NEND)


# ------------------------------------------------------------ design-power margin
def margin_sim(frozen, lab, high, J, target=0.80):
    """True rate at which all J bounds clear with probability `target`, from the same
    weighted simulation as the calibration, with common random numbers.  The J bounds are
    treated as independent: that approximation is stated with the result."""
    fs, fc = frozen[(lab, "screen ")], frozen[(lab, "confirm")]
    def pr(tv):
        s = wstat_crn(SCREEN_PER_STATE, tv, NMARG, 101)
        c = wstat_crn(CONF_PER_STATE,  tv, NMARG, 202)
        ps = float((s >= fs["q"]).mean() if high else (s <= fs["q"]).mean())
        pc = float((c >= fc["q"]).mean() if high else (c <= fc["q"]).mean())
        return (ps + (1 - ps) * pc)**J - target
    lo, hi = (0.9005, 0.99990) if high else (0.00010, 0.04950)
    try:
        tv = brentq(pr, lo, hi, xtol=2e-5)
    except ValueError:
        return None
    return (tv - 0.90) if high else (0.05 - tv)


# ================================================================= run
rng_t = np.random.default_rng(SEED_TUNE)
rng_a = np.random.default_rng(SEED_AUDIT)
report = dict(protocol="tune/freeze/audit", seed_tune=SEED_TUNE, seed_audit=SEED_AUDIT,
              tune_frac=TUNE_FRAC, reps=dict(tune=NCAL, audit_screen=NAUD_S,
              audit_confirm=NAUD_C, end_to_end=NEND, margin=NMARG),
              bound="one-sided 95% Clopper-Pearson upper", levels={})

print("Kish effective sample size from pilot draws")
report["kish"] = {}
for n, lab in ((SCREEN_PER_STATE, "screening"), (CONF_PER_STATE, "confirmatory")):
    _, nef = wstat(rng_t, n, 0.9, 4_000, want_neff=True)
    med = float(np.median(nef))
    report["kish"][lab] = dict(n=n, n_eff=med, ratio=med / n)
    print(f"  {lab+', per state':26s} n={n:5d}  median n_eff={med:7.1f}  ratio {med/n:.2f}")

all_ok = True
for alloc, label, lim, key in (
        (A_MARGINAL, "MARGINAL (architecture-specific claim)", ALPHA, "marginal"),
        (A_PROJECT,  "PROJECT-LEVEL (family-wise 0.05 over 8 architectures)",
         ALPHA / NARCH, "project")):
    frozen = tune(rng_t, alloc, label)
    rows, worst, ok = audit(rng_a, frozen, alloc)
    all_ok &= ok

    print("  assembled screen-escalate-retest, EVERY boundary type, at the supremum over J:")
    e2e, worst_e2e = {}, 0.0
    for lab, p0, high in BOUNDARY:
        o = end_to_end(rng_a, frozen, lab, high, p0, lim)
        e2e[lab.strip()] = o
        worst_e2e = max(worst_e2e, o["u95"])
    print(f"    largest end-to-end CP U95 over all three boundaries: {worst_e2e:.4f} "
          f"against {lim:.5f}: {'clears' if worst_e2e <= lim else 'DOES NOT CLEAR'}")
    all_ok &= worst_e2e <= lim

    print("  design-power margin for an 80% chance of passing all 1728 bounds")
    print("  (weighted simulation; the 1728 bounds treated as independent)")
    marg = {}
    for lab, p0, high in BOUNDARY:
        m = margin_sim(frozen, lab, high, 1728)
        marg[lab.strip()] = None if m is None else round(100 * m, 2)
        print(f"    {lab} {'n/a' if m is None else f'{100*m:.2f} pp'}")
    # the three are margins on DIFFERENT quantities (a rate near 0.90 and rates near
    # 0.05), so they are reported side by side and not collapsed into one number.
    worst_m = marg.get("power")

    report["levels"][key] = dict(
        allocation=alloc, requirement=lim,
        frozen={f"{a.strip()}/{b.strip()}": dict(q=v["q"], nominal=v["nominal"],
                                                 steps=v["steps"])
                for (a, b), v in frozen.items()},
        audit=rows, worst_component_u95=worst,
        component_size_min=min(r["size"] for r in rows.values()),
        component_size_max=max(r["size"] for r in rows.values()),
        end_to_end=e2e, worst_end_to_end_u95=worst_e2e,
        margins_pp=marg, design_power_margin_pp=worst_m,
        all_clear=bool(ok and worst_e2e <= lim))

report["all_clear"] = bool(all_ok)
with open("iut_check.json", "w") as f:
    json.dump(report, f, indent=1, sort_keys=True)
print(f"\nkeyed results written to iut_check.json; overall: "
      f"{'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 1)
