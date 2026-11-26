"""
paper_values.py - Summary statistics as reported in the manuscript.

These values are the inputs from which verify_statistics.py recomputes every
derived quantity in the paper (t-statistics, confidence intervals, effect
sizes, design-effect adjustments, retention and attrition analyses,
percentage changes, and cost and latency comparisons). They are not used
to generate or simulate data anywhere in this package.
"""

N_PER_GROUP = 150
N_SCHOOLS = 9
N_TOTAL = 300

# Table I - pre-test and post-test scores (mean, SD), n = 150 per group
TABLE_I = {
    #                 pre_E        pre_C        post_E       post_C
    "Mathematics":    ((61.3, 4.2), (61.1, 4.5), (85.2, 4.1), (64.7, 4.8)),
    "Science":        ((62.1, 4.8), (62.4, 4.6), (88.1, 4.3), (67.3, 4.6)),
    "English":        ((60.8, 4.1), (60.5, 4.3), (82.4, 4.5), (61.9, 4.2)),
    "Social Studies": ((59.7, 4.9), (59.3, 4.7), (79.8, 4.7), (59.2, 4.9)),
    "Overall":        ((61.0, 4.5), (60.9, 4.5), (83.9, 4.4), (63.3, 4.6)),
}
# Reported Table I statistics: t(298), 95% CI of the post-test difference,
# relative improvement (%)
TABLE_I_REPORTED = {
    "Mathematics":    (39.76, (19.49, 21.52), 31.7),
    "Science":        (40.46, (19.79, 21.81), 30.9),
    "English":        (40.79, (19.51, 21.49), 33.1),
    "Social Studies": (37.16, (19.51, 21.69), 34.8),
    "Overall":        (39.64, (19.58, 21.62), 32.5),
}
# Table II - Cohen's d [95% CI]
TABLE_II_REPORTED = {
    "Mathematics":    (4.59, (4.16, 5.02)),
    "Science":        (4.67, (4.23, 5.11)),
    "English":        (4.71, (4.27, 5.15)),
    "Social Studies": (4.29, (3.88, 4.70)),
    "Overall":        (4.58, (4.15, 5.01)),
}
# Section IV-A clustering sensitivity: ICC -> (DEFF, n_eff per group, adjusted overall t)
DESIGN_EFFECT_REPORTED = {
    0.05: (2.62, 57, 24.51),
    0.10: (4.23, 35, 19.27),
    0.20: (7.47, 20, 14.51),
    0.30: (10.70, 14, 12.12),
}
DESIGN_EFFECT_SUBJECT_RANGE_ICC030 = (11.36, 12.47)

# Power analysis (Section III-C): d = 0.45, two-tailed alpha = 0.05
POWER_REPORTED = {"d": 0.45, "n80_per_group": 79, "n95_per_group": 130, "power_at_150": 0.973}

# Table III - system performance (EduMorph-Edge, baseline, reported improvement)
TABLE_III = {
    "avg_rt_ms":   (141, 423, 66.7),
    "p95_rt_ms":   (245, 612, 60.0),
    "max_users":   (300, 50, 500.0),
    "cpu_pct":     (68.2, 84.7, 19.5),
    "memory_gb":   (17.8, 31.7, 43.8),
    "storage_iops": (12500, 8900, 40.4),
    "cost_usd":    (0.0031, 0.0041, 24.4),
}
UPTIME = (99.73, 97.21, 2.52)          # percentage-point difference
P99_EDUMORPH_PEAK_MS = 276             # at 280-300 concurrent users
# Table IV - controlled load tests: users -> (mean RT ms, CPU %, memory GB, success %)
TABLE_IV = {
    50: (123.0, 38.5, 12.3, 100.0),
    100: (129.0, 42.3, 13.1, 100.0),
    150: (138.0, 47.1, 14.2, 99.9),
    200: (144.0, 52.0, 15.0, 99.9),
    250: (151.0, 56.4, 15.6, 99.8),
    300: (157.0, 61.0, 16.2, 99.7),
}
MATCHED_LOAD_REDUCTION_REPORTED = 70.9  # % at 50 users, Table IV vs Table III baseline

# Table V - engagement (EduMorph-Edge, control, reported change %)
TABLE_V = {
    "Daily Active Time (min)":     (47.3, 32.1, 47.4),
    "Weekly Login Frequency":      (5.8, 3.9, 48.7),
    "Content Completion Rate":     (89.2, 67.4, 32.3),
    "Voluntary Extended Sessions": (34.6, 12.8, 170.3),
    "Peer Collaboration Events":   (23.7, 8.9, 166.3),
    "Help-Seeking Frequency":      (15.2, 21.7, -30.0),
    "Self-Correction Rate":        (78.9, 54.3, 45.3),
}
# Table VI - A/B cohort (learning gain %, engagement score)
TABLE_VI = {"C": (10.7, 59.4), "D": (29.8, 86.7)}
TABLE_VI_REPORTED_DIFF = (19.1, 27.3)
# Table VII - UX (EduMorph-Edge, control, reported difference)
TABLE_VII = {"SUS": (79.3, 61.2, 18.1), "NASA-TLX": (31.4, 52.3, -20.9),
             "NPS": (47, 12, 35), "Perceived transparency": (4.1, 2.9, 1.2)}
# Table VIII - subgroup sizes (EduMorph-Edge, control)
TABLE_VIII_N = {
    "gender": [(77, 75), (73, 75)],
    "geography": [(135, 135), (15, 15)],
    "prior_gpa": [(52, 51), (62, 63), (36, 36)],
}
FEMALE_PCT_REPORTED = 50.7
# Table IX - pre-to-post gains (%)
TABLE_IX_GAIN_REPORTED = {"EduMorph-Edge": 37.5, "Madrasati": 3.9}
VANLEHN_STEP_BASED_D = 0.76            # VanLehn (2011), step-based ITS vs no tutoring

# Section IV-G - six-month retention
RETENTION = {"E_retained": 62, "E_completed": 91, "C_retained": 27, "C_completed": 87}
RETENTION_REPORTED = {
    "chi2": 24.48, "phi": 0.37, "diff_pp": 37.1, "newcombe_ci": (22.6, 49.4),
    "or": 4.75, "or_ci": (2.52, 8.95), "rr": 2.20, "rr_ci": (1.56, 3.09),
    "pct_E": 68.1, "pct_C": 31.0,
}
ATTRITION = {"E_missing": 59, "C_missing": 63}
ATTRITION_REPORTED = {"chi2": 0.22, "p": 0.64, "pct_E": 39.3, "pct_C": 42.0, "overall_pct": 40.7}
ITT_REPORTED = {
    "common_rate_range_pp": (20.7, 23.3),
    "pooled_rate": 50.0, "pooled_E": 61.0, "pooled_C": 39.0, "pooled_diff": 22.0,
    "tipping_gap_pp": 52.5, "tipping_example": (0.0, 55.6),
    "worst_E": 41.3, "worst_C": 60.0,
}

# Table A.II - MSC-Trans classifier (precision, recall, F1)
TABLE_A2 = {"Engaged": (0.94, 0.91, 0.92), "Transitioning": (0.83, 0.87, 0.85),
            "Disengaged": (0.91, 0.89, 0.90)}
MACRO_F1_REPORTED = 0.89
CLASSIFIER_SET = {"students": 240, "windows": 14400, "train_frac": 0.68}
