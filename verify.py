"""
Build verification for the MIRROR proposal.  Runs on the FINAL assembled PDF,
prints the SHA-256 of the exact file it tested, and scans every page.
Usage: python3 verify.py [proposal.pdf]
"""
import subprocess, re, sys, collections, statistics, os, hashlib

PDF = sys.argv[1] if len(sys.argv) > 1 else "proposal.pdf"
LOG = os.path.splitext(PDF)[0] + ".log"
TEX = os.path.splitext(PDF)[0] + ".tex"
SPAGES = 7
fail = []

def check(ok, label, detail=""):
    print(("  OK   " if ok else "  FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not ok: fail.append(label)

def text(a=None, b=None):
    cmd = ["pdftotext", "-layout"]
    if a: cmd += ["-f", str(a), "-l", str(b or a)]
    return subprocess.run(cmd + [PDF, "-"], capture_output=True, text=True).stdout

sha = hashlib.sha256(open(PDF, "rb").read()).hexdigest()
print(f"FILE   {os.path.abspath(PDF)}\nSHA256 {sha}\n")

def must(cond, msg):
    """A precondition, not a check: if the evidence is missing the verifier stops rather
    than reporting OK on an empty string.  Every 'log reports no X' check below is
    vacuously true when the log is absent, which is exactly how a real defect once
    survived verification."""
    if not cond:
        print(f"  ABORT {msg}")
        sys.exit(2)

# --- preconditions: the evidence each section needs must actually exist -------------
must(os.path.exists(PDF), f"PDF not found: {PDF}")
must(os.path.exists(TEX), f"source not found: {TEX}")
must(os.path.exists(LOG), f"build log not found: {LOG}. The log-based integrity checks "
                          f"cannot run without it; rebuild with pdflatex and retry.")
pi = subprocess.run(["pdfinfo", PDF], capture_output=True, text=True)
must(pi.returncode == 0, f"pdfinfo failed (exit {pi.returncode}): {pi.stderr.strip()[:160]}")
mpages = re.search(r"Pages:\s+(\d+)", pi.stdout)
must(mpages is not None, "pdfinfo produced no page count")
pages = int(mpages.group(1))

log = open(LOG, errors="ignore").read()
must(len(log) > 200, f"build log is {len(log)} bytes, too short to be a real pdflatex log")
must(os.path.getmtime(LOG) >= os.path.getmtime(PDF) - 300,
     "build log is much older than the PDF; it does not describe this build")
tex = open(TEX).read()
whole = text()                      # EVERY page, not just the S/T/M
must(len(whole) > 2000, "pdftotext returned almost nothing; check the Poppler install")
body  = text(1, SPAGES)

print("integrity")
und = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
check(not und, "no undefined references", ",".join(und))
check("There were undefined references" not in log, "log reports no undefined references")
check("multiply defined" not in log, "no multiply-defined labels")
check("??" not in whole, "no '??' anywhere in the rendered document")
declared = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
gpaths = re.findall(r"\\graphicspath\{((?:\{[^}]*\})+)\}", tex)
searchdirs = [""] + ([p for p in re.findall(r"\{([^}]*)\}", gpaths[0])] if gpaths else [])
check(bool(declared), "figures declared", str(len(declared)))
for g in declared:
    found = any(os.path.exists(d + g) or os.path.exists(d + g + ".pdf") for d in searchdirs)
    check(found, f"graphic present: {g}")
n_tex = tex.count(r"\begin{figure}") + tex.count(r"\captionof{figure}")
n_pdf = len(re.findall(r"Figure \d+:", whole))
check(n_tex == n_pdf and n_tex == len(declared), "every declared figure renders with a caption",
      f"tex {n_tex}, pdf {n_pdf}, graphics {len(declared)}")

print("\npage limit")
first_ref = next((i for i in range(1, pages + 1) if text(i).lstrip().startswith("References")), None)
check(first_ref == SPAGES + 1, f"S/T/M is exactly {SPAGES} pages", f"references start on p{first_ref}")

print("\ncitations")
nums = []
for m in re.finditer(r"\[([0-9,\s]+)\]", body):
    nums += [int(t) for t in m.group(1).split(",") if t.strip().isdigit()]
seen = []
for n in nums:
    if n not in seen: seen.append(n)
breaks = [(seen[i], seen[i+1]) for i in range(len(seen)-1) if seen[i+1] != seen[i] + 1]
check(not breaks, "citations in first-appearance order", str(breaks))
i, j = tex.index("section*{References}"), tex.index("section*{Open Science")
nbib = tex[i:j].count("\n\\item ")
check(nbib == max(seen), "every reference cited, none missing", f"bib {nbib}, max cited {max(seen)}")

print("\ntypography")
ds = []
for p in (1, 2, 3, 6, 7):
    o = subprocess.run(["pdftotext", "-bbox", "-f", str(p), "-l", str(p), PDF, "-"], capture_output=True, text=True).stdout
    ys = sorted({round(float(m), 2) for m in re.findall(r'yMin="([0-9.]+)"', o)})
    ds += [round(ys[k+1] - ys[k], 2) for k in range(len(ys) - 1)]
lpi = 72.0 / collections.Counter(x for x in ds if 8 < x < 20).most_common(1)[0][0]
check(lpi <= 5.5, "lines per inch <= 5.5", f"{lpi:.3f}")
vals = []
for p in range(1, SPAGES + 1):
    o = subprocess.run(["pdftotext", "-bbox", "-f", str(p), "-l", str(p), PDF, "-"], capture_output=True, text=True).stdout
    rows = collections.defaultdict(list)
    for m in re.finditer(r'<word xMin="([0-9.]+)" yMin="([0-9.]+)" xMax="([0-9.]+)" yMax="[0-9.]+">([^<]*)</word>', o):
        rows[round(float(m.group(2)), 1)].append((float(m.group(1)), float(m.group(3)), m.group(4)))
    for y, ws in rows.items():
        ws.sort(); n = sum(len(w[2]) for w in ws) + len(ws) - 1; w = (ws[-1][1] - ws[0][0]) / 72.0
        if w > 4.0 and n > 50: vals.append(n / w)
check(statistics.median(vals) <= 15.0, "characters per inch <= 15", f"median {statistics.median(vals):.2f}")

print("\nanonymity and process language (WHOLE document)")
info = subprocess.run(["pdfinfo", PDF], capture_output=True, text=True).stdout
for k in ("Author", "Creator", "Producer"):
    m = re.search(rf"^{k}:[ \t]*(.*?)[ \t]*$", info, re.M)
    check((m.group(1) if m else "") == "", f"{k} metadata blank", repr(m.group(1) if m else ""))
t = re.search(r"^Title:[ \t]*(.*?)[ \t]*$", info, re.M)
check("Anonymized" not in (t.group(1) if t else ""), "Title metadata free of process language", repr(t.group(1) if t else ""))
own = sorted(set(re.findall(r"(?i)we have (?:built|developed|shown)|the PI's (?:expertise|experience)|our (?:previous|prior) (?:work|paper)|\bUniversity\b|Institute of Technology|\bDepartment of\b", whole)))
own += sorted(set(re.findall(r"\b(?:Dr|Prof)\.", whole)))
check(not own, "no ownership or affiliation language", ",".join(own))
proc = sorted(set(re.findall(r"(?i)DAPR|anonymized (?:fashion|technical proposal)|dual.anonymous", whole)))
check(not proc, "no review-process language", ",".join(proc))
check(b"/Outlines" not in open(PDF, "rb").read(), "no PDF outline")

print("\ncross-section consistency")
def nums_near(pat, s=whole):
    return sorted(set(re.findall(pat, s)))
store_tb = nums_near(r"(\d+(?:\.\d+)?)\s*TB working")
store_gb = nums_near(r"(\d+)\s*GB curated")
check(len(store_tb) == 1, "working-storage figure stated once", ",".join(store_tb))
check(len(store_gb) == 1, "curated-storage figure stated once", ",".join(store_gb))
for token in ("192", "1728", "2.57", "3.53"):
    check(token.replace(".", "") in whole.replace(".", "").replace(" ", ""),
          f"design figure {token} present")

# Every calibration figure in Section 3.5 is looked up BY KEY in iut_check.json and the
# string the proposal must contain is built from that value.  A prose log can be matched by
# coincidence; a keyed lookup cannot.  The audit must also have passed on its own terms.
print("\ncalibration numbers keyed to iut_check.json")
JS = "iut_check.json"
if os.path.exists(JS):
    import json
    J = json.load(open(JS))
    check(J.get("protocol") == "tune/freeze/audit", "protocol is tune/freeze/audit",
          str(J.get("protocol")))
    check(J.get("seed_tune") != J.get("seed_audit"), "audit seed differs from tuning seed",
          f'{J.get("seed_tune")} vs {J.get("seed_audit")}')
    check("Clopper" in J.get("bound", ""), "sizes are Clopper-Pearson bounds",
          J.get("bound", ""))
    check(J.get("all_clear") is True, "audit cleared on its own terms")

    def f4(x): return f"{x:.4f}"

    for key, alloc_s, req_s in (("marginal", "0.021", "0.05"),
                                ("project",  "0.003125", "0.00625")):
        L = J["levels"][key]
        # the audit must actually clear, not merely be reported
        check(L["worst_component_u95"] <= L["allocation"],
              f"{key}: worst component bound within allocation",
              f'{L["worst_component_u95"]:.4f} <= {L["allocation"]}')
        check(L["worst_end_to_end_u95"] <= L["requirement"],
              f"{key}: end-to-end bound within requirement",
              f'{L["worst_end_to_end_u95"]:.4f} <= {L["requirement"]}')
        check(len(L["end_to_end"]) == 3,
              f"{key}: end-to-end run at all three boundaries",
              ",".join(sorted(L["end_to_end"])))
        check(all(r["clears"] for r in L["audit"].values()),
              f"{key}: every audited component clears")
        check(len(L["audit"]) == 6, f"{key}: six boundary-by-stage components",
              str(len(L["audit"])))
        # and the proposal must quote exactly these values
        for lbl, val in (("size range low", L["component_size_min"]),
                         ("size range high", L["component_size_max"]),
                         ("worst component bound", L["worst_component_u95"]),
                         ("worst end-to-end bound", L["worst_end_to_end_u95"])):
            check(f4(val) in whole, f"{key}: text quotes {lbl} {f4(val)}")
        check(alloc_s in whole, f"{key}: text quotes allocation {alloc_s}")
        check(req_s in whole, f"{key}: text quotes requirement {req_s}")
        m = L["design_power_margin_pp"]
        check(f"{m:.1f}" in whole, f"{key}: text quotes design-power margin {m:.1f}")
    for tok, lbl in (("$6\\times10^4$", "tuning replicates"),
                     ("$1.2\\times10^5$", "screening audit replicates"),
                     ("$5\\times10^4$", "confirmatory audit replicates")):
        check(tok in tex, f"tex quotes {lbl}")
    for phrase in ("three-stag", "Clopper", "audit", "supremum"):
        check(phrase.lower() in whole.lower(), f"text describes the protocol: '{phrase}'")
else:
    check(False, "iut_check.json present (run iut_check.py)")

# design.py must not carry a second, competing calibration.  Any 4-decimal figure it prints
# has to come from the same log the proposal quotes; this is what catches a stale duplicate.
print("\ndesign.py carries no competing calibration")
if os.path.exists("design.py") and os.path.exists("iut_check.log"):
    d = subprocess.run([sys.executable, "design.py"], capture_output=True, text=True)
    check(d.returncode == 0, "design.py runs", d.stderr.strip()[:80])
    cal = open("iut_check.log").read()
    stray = sorted({t for t in re.findall(r"\b0\.\d{4}\b", d.stdout) if t not in cal})
    check(not stray, "every 4-decimal figure in design.py output is in the log", ",".join(stray))
    for tok in ("2,570,000", "3,530,000", "141,500", "189,500", "894,000"):
        check(tok in d.stdout, f"design.py reproduces {tok}")
    for tok in ("2.57", "3.53", "1.42", "1.90", "8.9"):
        check(tok in whole, f"text quotes compute figure {tok}")
else:
    check(False, "design.py and iut_check.log present")

print("\nRESULT:", "PASS" if not fail else f"{len(fail)} FAILED -> " + "; ".join(fail))
sys.exit(1 if fail else 0)
