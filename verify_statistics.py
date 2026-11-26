"""
verify_statistics.py - Recompute every derived statistic reported in the paper.

Each check recomputes a quantity from the reported summary data in
paper_values.py and compares it with the value printed in the manuscript.
A check passes when the two agree within the stated rounding tolerance
(input means and SDs are themselves reported to one decimal place, so
t-statistics and confidence limits are allowed a slightly wider tolerance).

Outputs:
  results/verification_report.md   - human-readable table of all checks
  results/verification_results.json - machine-readable results

Usage:
  python verify_statistics.py
Exit code is 0 when every check passes and 1 otherwise.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

import paper_values as pv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


class Checker:
    def __init__(self):
        self.rows = []

    def check(self, section, item, computed, reported, tol):
        ok = bool(abs(float(computed) - float(reported)) <= tol + 1e-9)
        self.rows.append({"section": section, "item": item,
                          "computed": round(float(computed), 4),
                          "reported": float(reported), "tolerance": tol,
                          "status": "PASS" if ok else "FAIL"})
        return ok

    @property
    def n_fail(self):
        return sum(r["status"] == "FAIL" for r in self.rows)


def pooled_sd(sd1, sd2):
    return np.sqrt((sd1 ** 2 + sd2 ** 2) / 2.0)


def cohens_d_ci(d, n1, n2):
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return d - 1.96 * se, d + 1.96 * se


def wilson(x, n, z=1.96):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff_ci(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def t_test_power(d, n, alpha=0.05):
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2.0)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


def pct_change(new, old):
    return (new - old) / old * 100.0


def run():
    c = Checker()
    n = pv.N_PER_GROUP
    df = 2 * n - 2
    tcrit = stats.t.ppf(0.975, df)

    # ---- Table I and Table II
    for subj, (pre_e, pre_c, post_e, post_c) in pv.TABLE_I.items():
        sp = pooled_sd(post_e[1], post_c[1])
        diff = post_e[0] - post_c[0]
        se = sp * np.sqrt(2.0 / n)
        t = diff / se
        t_rep, ci_rep, imp_rep = pv.TABLE_I_REPORTED[subj]
        c.check("Table I", f"{subj}: t(298)", t, t_rep, 0.05)
        c.check("Table I", f"{subj}: 95% CI lower", diff - tcrit * se, ci_rep[0], 0.03)
        c.check("Table I", f"{subj}: 95% CI upper", diff + tcrit * se, ci_rep[1], 0.03)
        c.check("Table I", f"{subj}: relative improvement (%)", pct_change(post_e[0], post_c[0]), imp_rep, 0.05)
        t_pre = (pre_e[0] - pre_c[0]) / (pooled_sd(pre_e[1], pre_c[1]) * np.sqrt(2.0 / n))
        c.check("Table I", f"{subj}: pre-test baseline equivalence (p > 0.05)", float(2 * stats.t.sf(abs(t_pre), df) > 0.05), 1, 0)
        d = diff / sp
        d_rep, dci_rep = pv.TABLE_II_REPORTED[subj]
        c.check("Table II", f"{subj}: Cohen's d", d, d_rep, 0.01)
        c.check("Table II", f"{subj}: d = t*sqrt(2/n)", t * np.sqrt(2.0 / n), d_rep, 0.01)
        lo, hi = cohens_d_ci(d, n, n)
        c.check("Table II", f"{subj}: d 95% CI lower", lo, dci_rep[0], 0.01)
        c.check("Table II", f"{subj}: d 95% CI upper", hi, dci_rep[1], 0.01)

    # ---- Table IX pre-to-post gains
    pre_e, pre_c, post_e, post_c = pv.TABLE_I["Overall"]
    c.check("Table IX", "EduMorph-Edge pre-to-post gain (%)", pct_change(post_e[0], pre_e[0]), pv.TABLE_IX_GAIN_REPORTED["EduMorph-Edge"], 0.05)
    c.check("Table IX", "Madrasati pre-to-post gain (%)", pct_change(post_c[0], pre_c[0]), pv.TABLE_IX_GAIN_REPORTED["Madrasati"], 0.05)

    # ---- Clustering sensitivity (Section IV-A)
    m = pv.N_TOTAL / pv.N_SCHOOLS
    t_overall = pv.TABLE_I_REPORTED["Overall"][0]
    for icc, (deff_rep, neff_rep, tadj_rep) in pv.DESIGN_EFFECT_REPORTED.items():
        deff = 1 + (m - 1) * icc
        tadj = t_overall / np.sqrt(deff)
        c.check("Sec. IV-A clustering", f"ICC={icc}: design effect", deff, deff_rep, 0.005)
        c.check("Sec. IV-A clustering", f"ICC={icc}: effective n per group", round(n / deff), neff_rep, 0)
        c.check("Sec. IV-A clustering", f"ICC={icc}: adjusted t", tadj, tadj_rep, 0.01)
        p = 2 * stats.t.sf(tadj, pv.N_SCHOOLS - 2)
        c.check("Sec. IV-A clustering", f"ICC={icc}: p < 0.001 (df = 7)", float(p < 0.001), 1, 0)
    deff30 = 1 + (m - 1) * 0.30
    subj_t = [pv.TABLE_I_REPORTED[s][0] / np.sqrt(deff30) for s in ["Mathematics", "Science", "English", "Social Studies"]]
    c.check("Sec. IV-A clustering", "ICC=0.30: min subject adjusted t", min(subj_t), pv.DESIGN_EFFECT_SUBJECT_RANGE_ICC030[0], 0.01)
    c.check("Sec. IV-A clustering", "ICC=0.30: max subject adjusted t", max(subj_t), pv.DESIGN_EFFECT_SUBJECT_RANGE_ICC030[1], 0.01)

    # ---- Power analysis (Section III-C)
    d0 = pv.POWER_REPORTED["d"]
    def n_required(target):
        k = 2
        while t_test_power(d0, k) < target:
            k += 1
        return k
    c.check("Sec. III-C power", "n per group for 80% power", n_required(0.80), pv.POWER_REPORTED["n80_per_group"], 0)
    c.check("Sec. III-C power", "n per group for 95% power", n_required(0.95), pv.POWER_REPORTED["n95_per_group"], 0)
    c.check("Sec. III-C power", "power at n = 150 per group", t_test_power(d0, 150), pv.POWER_REPORTED["power_at_150"], 0.0005)

    # ---- Table III / Table IV / cost
    for key, (e, b, rep) in pv.TABLE_III.items():
        if key in ("storage_iops", "max_users"):
            val = pct_change(e, b)
        else:
            val = (b - e) / b * 100.0
        c.check("Table III", f"{key}: change (%)", val, rep, 0.05)
    c.check("Table III", "uptime difference (pp)", pv.UPTIME[0] - pv.UPTIME[1], pv.UPTIME[2], 0.005)
    c.check("Table III", "p99 (276 ms) exceeds p95 (245 ms)", float(pv.P99_EDUMORPH_PEAK_MS > pv.TABLE_III["p95_rt_ms"][0]), 1, 0)
    rt50 = pv.TABLE_IV[50][0]
    c.check("Sec. IV-B", "matched-load (50 users) mean RT reduction (%)", (pv.TABLE_III["avg_rt_ms"][1] - rt50) / pv.TABLE_III["avg_rt_ms"][1] * 100, pv.MATCHED_LOAD_REDUCTION_REPORTED, 0.05)
    c.check("Table IV", "max mean RT below 160 ms", float(max(v[0] for v in pv.TABLE_IV.values()) < 160), 1, 0)

    # ---- Table V
    for k, (e, ctl, rep) in pv.TABLE_V.items():
        c.check("Table V", f"{k}: change (%)", pct_change(e, ctl), rep, 0.05)

    # ---- Table VI and VII
    c.check("Table VI", "D - C learning gain (pp)", pv.TABLE_VI["D"][0] - pv.TABLE_VI["C"][0], pv.TABLE_VI_REPORTED_DIFF[0], 0.005)
    c.check("Table VI", "D - C engagement score", pv.TABLE_VI["D"][1] - pv.TABLE_VI["C"][1], pv.TABLE_VI_REPORTED_DIFF[1], 0.005)
    for k, (e, ctl, rep) in pv.TABLE_VII.items():
        c.check("Table VII", f"{k}: difference", e - ctl, rep, 0.005)

    # ---- Table VIII subgroup sizes and sample description
    for k, groups in pv.TABLE_VIII_N.items():
        c.check("Table VIII", f"{k}: EduMorph-Edge n sums to 150", sum(g[0] for g in groups), 150, 0)
        c.check("Table VIII", f"{k}: control n sums to 150", sum(g[1] for g in groups), 150, 0)
    female = sum(pv.TABLE_VIII_N["gender"][0]) / pv.N_TOTAL * 100
    c.check("Sec. III-C sample", "female (%)", female, pv.FEMALE_PCT_REPORTED, 0.05)

    # ---- Retention (Section IV-G)
    R = pv.RETENTION
    a, b_ = R["E_retained"], R["E_completed"] - R["E_retained"]
    cc, d_ = R["C_retained"], R["C_completed"] - R["C_retained"]
    chi2, p, _, _ = stats.chi2_contingency(np.array([[a, b_], [cc, d_]]), correction=False)
    N = R["E_completed"] + R["C_completed"]
    rr = pv.RETENTION_REPORTED
    c.check("Sec. IV-G retention", "% retained, EduMorph-Edge", a / R["E_completed"] * 100, rr["pct_E"], 0.05)
    c.check("Sec. IV-G retention", "% retained, control", cc / R["C_completed"] * 100, rr["pct_C"], 0.05)
    c.check("Sec. IV-G retention", "chi-square(1)", chi2, rr["chi2"], 0.005)
    c.check("Sec. IV-G retention", "phi", np.sqrt(chi2 / N), rr["phi"], 0.005)
    p1, p2 = a / R["E_completed"], cc / R["C_completed"]
    c.check("Sec. IV-G retention", "difference in proportions (pp)", (p1 - p2) * 100, rr["diff_pp"], 0.05)
    lo, hi = newcombe_diff_ci(a, R["E_completed"], cc, R["C_completed"])
    c.check("Sec. IV-G retention", "Newcombe CI lower (pp)", lo * 100, rr["newcombe_ci"][0], 0.05)
    c.check("Sec. IV-G retention", "Newcombe CI upper (pp)", hi * 100, rr["newcombe_ci"][1], 0.05)
    orr = a * d_ / (b_ * cc)
    se_lor = np.sqrt(1 / a + 1 / b_ + 1 / cc + 1 / d_)
    c.check("Sec. IV-G retention", "odds ratio", orr, rr["or"], 0.005)
    c.check("Sec. IV-G retention", "OR 95% CI lower", np.exp(np.log(orr) - 1.96 * se_lor), rr["or_ci"][0], 0.005)
    c.check("Sec. IV-G retention", "OR 95% CI upper", np.exp(np.log(orr) + 1.96 * se_lor), rr["or_ci"][1], 0.005)
    risk_ratio = p1 / p2
    se_lrr = np.sqrt(1 / a - 1 / R["E_completed"] + 1 / cc - 1 / R["C_completed"])
    c.check("Sec. IV-G retention", "risk ratio", risk_ratio, rr["rr"], 0.005)
    c.check("Sec. IV-G retention", "RR 95% CI lower", np.exp(np.log(risk_ratio) - 1.96 * se_lrr), rr["rr_ci"][0], 0.005)
    c.check("Sec. IV-G retention", "RR 95% CI upper", np.exp(np.log(risk_ratio) + 1.96 * se_lrr), rr["rr_ci"][1], 0.005)

    # ---- Attrition and intention-to-treat sensitivity (Section IV-G)
    A = pv.ATTRITION
    ar = pv.ATTRITION_REPORTED
    chi2a, pa, _, _ = stats.chi2_contingency(np.array([[A["E_missing"], n - A["E_missing"]], [A["C_missing"], n - A["C_missing"]]]), correction=False)
    c.check("Sec. IV-G attrition", "attrition chi-square(1)", chi2a, ar["chi2"], 0.005)
    c.check("Sec. IV-G attrition", "attrition p-value", pa, ar["p"], 0.005)
    c.check("Sec. IV-G attrition", "attrition %, EduMorph-Edge", A["E_missing"] / n * 100, ar["pct_E"], 0.05)
    c.check("Sec. IV-G attrition", "attrition %, control", A["C_missing"] / n * 100, ar["pct_C"], 0.05)
    c.check("Sec. IV-G attrition", "overall attrition %", (A["E_missing"] + A["C_missing"]) / pv.N_TOTAL * 100, ar["overall_pct"], 0.05)

    def itt(rate_e, rate_c):
        e = (a + A["E_missing"] * rate_e) / n
        ctl = (cc + A["C_missing"] * rate_c) / n
        return e * 100, ctl * 100
    it = pv.ITT_REPORTED
    diffs = [itt(r, r)[0] - itt(r, r)[1] for r in np.linspace(0, 1, 101)]
    c.check("Sec. IV-G ITT", "common-rate difference, minimum (pp)", min(diffs), it["common_rate_range_pp"][0], 0.05)
    c.check("Sec. IV-G ITT", "common-rate difference, maximum (pp)", max(diffs), it["common_rate_range_pp"][1], 0.05)
    pooled = (a + cc) / (R["E_completed"] + R["C_completed"])
    c.check("Sec. IV-G ITT", "pooled completer retention rate (%)", pooled * 100, it["pooled_rate"], 0.05)
    pe, pc_ = itt(pooled, pooled)
    c.check("Sec. IV-G ITT", "pooled-rate imputation, EduMorph-Edge (%)", pe, it["pooled_E"], 0.05)
    c.check("Sec. IV-G ITT", "pooled-rate imputation, control (%)", pc_, it["pooled_C"], 0.05)
    c.check("Sec. IV-G ITT", "pooled-rate imputation, difference (pp)", pe - pc_, it["pooled_diff"], 0.05)
    # tipping point: smallest gap (rate_c - rate_e) that eliminates the difference
    gaps = []
    for re_ in np.linspace(0, min(1.0, (A["C_missing"] + cc - a) / A["E_missing"]), 2001):
        rc = (a - cc + A["E_missing"] * re_) / A["C_missing"]
        if rc <= 1:
            gaps.append(rc - re_)
    c.check("Sec. IV-G ITT", "tipping-point minimum gap (pp)", min(gaps) * 100, it["tipping_gap_pp"], 0.05)
    c.check("Sec. IV-G ITT", "tipping example: control rate when EduMorph rate = 0 (%)", (a - cc) / A["C_missing"] * 100, it["tipping_example"][1], 0.05)
    we, wc = itt(0.0, 1.0)
    c.check("Sec. IV-G ITT", "worst case, EduMorph-Edge (%)", we, it["worst_E"], 0.05)
    c.check("Sec. IV-G ITT", "worst case, control (%)", wc, it["worst_C"], 0.05)

    # ---- Table A.II
    f1s = []
    for k, (P, Rc, F) in pv.TABLE_A2.items():
        f = 2 * P * Rc / (P + Rc)
        f1s.append(F)
        c.check("Table A.II", f"{k}: F1 from precision and recall", f, F, 0.005)
    c.check("Table A.II", "macro-averaged F1", np.mean(f1s), pv.MACRO_F1_REPORTED, 0.005)
    cs = pv.CLASSIFIER_SET
    c.check("Sec. III-A classifier", "windows per student", cs["windows"] / cs["students"], 60, 0)
    return c


def write_reports(c):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "verification_results.json"), "w") as f:
        json.dump({"n_checks": len(c.rows), "n_fail": c.n_fail, "checks": c.rows}, f, indent=2)
    lines = ["# Verification report", "",
             f"Checks run: {len(c.rows)}; passed: {len(c.rows) - c.n_fail}; failed: {c.n_fail}.", "",
             "Each value is recomputed from the reported summary data (paper_values.py) and compared",
             "with the value printed in the manuscript, within the stated rounding tolerance.", "",
             "| Section | Item | Computed | Reported | Tolerance | Status |",
             "|---|---|---|---|---|---|"]
    for r in c.rows:
        lines.append(f"| {r['section']} | {r['item']} | {r['computed']} | {r['reported']} | {r['tolerance']} | {r['status']} |")
    with open(os.path.join(RESULTS_DIR, "verification_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    c = run()
    write_reports(c)
    for r in c.rows:
        if r["status"] == "FAIL":
            print(f"FAIL  {r['section']}: {r['item']} computed={r['computed']} reported={r['reported']}")
    print(f"{len(c.rows)} checks, {c.n_fail} failed. Report: results/verification_report.md")
    return 0 if c.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
