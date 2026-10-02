"""
MIRROR design, sampling and compute accounting.  Every table row multiplies out
literally from the constants below.   Run:  python3 design.py
"""
import math
from fractions import Fraction as F
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------- cell structure ----------------
SHARED_SPECIES = 9          # species common to both libraries, each withheld in turn
NONOMIT_PRIMITIVE = 5       # cloud, surface, cross sections, correlated residuals,
                            # independent code
COMPOUND = 1                # pairs of the five above: C(5,2) = 10 pairs, no omission
NCOND = 1 + NONOMIT_PRIMITIVE + COMPOUND + SHARED_SPECIES   # 1 + 5 + 1 + 9 = 16
NATM, NLIB, NSPEC = 6, 2, 4
CELLS = NCOND*NATM*NLIB
GATED = CELLS*NSPEC
BOUNDS = 2*GATED + CELLS    # upper bound: power + fpr per (species,cell), plus one r_CO
                            # per cell, before the structural-exclusion mask

# ---------------- sample sizes ----------------
SCREEN_N, CONF_N, CAL_N = 500, 10_000, 4000
G0_N, G0_CAP = 10_000, 15_000
STRATA = 2                  # present / absent, balanced within a cell
DEFF = 2.0

# ---------------- architectures ----------------
OBS_CONFIGS   = 4           # 3 anchor EACs + 1 selected active configuration
S2_TOTAL      = 40          # 24 one-at-a-time cells + 16 factorial runs
S2_ANCHOR_REPS= 6           # reuse EAC-1's frozen threshold -> pure-error replicates
S2_DISTINCT   = S2_TOTAL - S2_ANCHOR_REPS
TRANSFER      = 8
S3_ARCH       = 8           # confirmatory architectures, each fully screened
PASS_ARCH     = OBS_CONFIGS + S3_ARCH      # architectures screened; only S3_ARCH are
#                                          # eligible for the project-level pass claim
S2_CELLS      = 3*3*1       # 3 conditions x Archean/Proterozoic/Phanerozoic x restrictive
S3_BASE_CELLS = 3*2*NLIB    # 3 conditions x Archean/Proterozoic x both libraries
S3_CELL_CAP   = 24

ensembles = OBS_CONFIGS*NLIB + S2_DISTINCT + TRANSFER + S3_ARCH

rows = [
 ("Gate G0 calibration", f"{G0_N} draws x 1 retrieval", G0_N),
 ("Threshold ensembles", f"{ensembles} ensembles x {CAL_N} planets", ensembles*CAL_N),
 ("Full-grid screen",    f"{PASS_ARCH} screened x {CELLS} cells x {SCREEN_N}", PASS_ARCH*CELLS*SCREEN_N),
 ("S2 response surface", f"{S2_TOTAL} arch x {S2_CELLS} cells x {SCREEN_N}", S2_TOTAL*S2_CELLS*SCREEN_N),
 ("Transfer validation", f"{TRANSFER} configs x {S2_CELLS} cells x {SCREEN_N}", TRANSFER*S2_CELLS*SCREEN_N),
 ("S3 confirmation",     f"{S3_ARCH} arch x {S3_BASE_CELLS} cells x {CONF_N}", S3_ARCH*S3_BASE_CELLS*CONF_N),
]
tot = sum(r[2] for r in rows)
s3_cap = S3_ARCH*S3_CELL_CAP*CONF_N
tot_cap = tot - rows[-1][2] + s3_cap

print(f"conditions {NCOND} (1 control + {NONOMIT_PRIMITIVE} primitives + {COMPOUND} compound + {SHARED_SPECIES} omission subcells)")
print(f"cells {NCOND}x{NATM}x{NLIB} = {CELLS};  gated (species,cell) pairs {GATED};  component bounds {BOUNDS}")
print(f"architectures screened (full grid):        {PASS_ARCH}")
print(f"architectures in project-level pass claim: {S3_ARCH}")
print(f"  the other {PASS_ARCH-S3_ARCH} are the anchor/active cases: diagnostic screens,\n  marginal pass reportable, excluded a priori from the project-level claim\n")
w=max(len(r[0]) for r in rows)
for n,f,v in rows: print(f"  {n:<{w}}  {f:<42} {v:>10,}")
print(f"  {'ROUTE A TOTAL':<{w}}  {'':<42} {tot:>10,}   ({tot/1e6:.2f}e6)")
print(f"  {'S3 at escalation cap':<{w}}  {f'{S3_ARCH} x {S3_CELL_CAP} x {CONF_N}':<42} {s3_cap:>10,}")
print(f"  {'WORST CASE':<{w}}  {'':<42} {tot_cap:>10,}   ({tot_cap/1e6:.2f}e6)")

EMU, TRAIN_N, TRAIN_C, RB_N, RB_C = 0.05, 300_000, 0.01, 5_000, 2.0
cpu = lambda n: n*EMU + TRAIN_N*TRAIN_C + RB_N*RB_C
print(f"\nCPU-hr baseline {cpu(tot):>10,.0f}  ({cpu(tot)/1e5:.2f}e5)")
print(f"CPU-hr cap case {cpu(tot_cap):>10,.0f}  ({cpu(tot_cap)/1e5:.2f}e5)")

cA_cells = NCOND*NATM*1
cA = (G0_N + (OBS_CONFIGS + S2_DISTINCT//2 + TRANSFER)*CAL_N
      + (OBS_CONFIGS + S3_ARCH//2)*cA_cells*SCREEN_N
      + 24*S2_CELLS*SCREEN_N + TRANSFER*S2_CELLS*SCREEN_N
      + (S3_ARCH//2)*(3*2*1)*CONF_N)
print(f"\nContingency A   {cA:>10,}  ({cA/1e5:.1f}e5), factor {tot/cA:.1f}")

# ---------------- precision from the actual endpoint denominators ----------------
def hw(p, n, z=1.96, deff=DEFF): return z*math.sqrt(p*(1-p)/(n/deff))*100
conf_per_state   = CONF_N//STRATA     # 5000 presences and 5000 absences per gated species
screen_per_state = SCREEN_N//STRATA   # 250 and 250
disc = int(3.2*CONF_N)
print(f"\nendpoint denominators: confirmatory {conf_per_state}/state, screening {screen_per_state}/state")
print(f"  power @80% confirmatory   {hw(.80,conf_per_state):.2f} pp")
print(f"  power @90% confirmatory   {hw(.90,conf_per_state):.2f} pp")
print(f"  fpr   @5%  confirmatory   {hw(.05,conf_per_state):.2f} pp")
print(f"  FDR   @5%  ({disc} disc) {hw(.05,disc):.2f} pp")
print(f"  power @90% screening      {hw(.90,screen_per_state):.2f} pp   <- sets the escalation band")

# ---------------- calibrated decision rule and the pass margin ----------------
# Thresholds come from iut_check.py, which resamples the actual WEIGHTED statistic
# under each boundary null; a binomial critical value at an effective n is not exact.
import numpy as np, math
ALPHA   = 0.05
# ---------------- calibration: single source of truth ----------------
# The decision-rule calibration lives in iut_check.py.  This script does NOT recompute it:
# two scripts with two answers is how a proposal ends up quoting a stale number.  Read the
# log and echo what it actually says.
import os, sys
LOG = "iut_check.log"
if not os.path.exists(LOG):
    sys.exit("design.py: run iut_check.py first; calibration is read from " + LOG)
cal = open(LOG).read()
print("\ncalibration, as logged by iut_check.py (not recomputed here)")
for line in cal.splitlines():
    t = line.strip()
    if (t.startswith(("MARGINAL", "PROJECT-LEVEL", "worst upper bound", "end-to-end upper bound",
                      "design-power margin")) or t.startswith("ratio") or "median n_eff" in t):
        print("  " + t)

# ---------------- populations ----------------
M = {"M1":[F(1,6)]*6,
     "M2":[F(8,25),F(8,25),F(4,25),F(1,15),F(1,15),F(1,15)],
     "M3":[F(2,25),F(2,25),F(1,25),F(4,15),F(4,15),F(4,15)]}
print()
for k,v in M.items():
    assert sum(v)==1
    print(f"{k} sum={sum(v)}  " + ", ".join(str(x) for x in v))
