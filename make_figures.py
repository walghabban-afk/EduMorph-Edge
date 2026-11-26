"""
make_figures.py - Regenerate the data figures of the paper.

  Figure 4: EduMorph-Edge vs. the baseline platform (Table III, plus the
            matched-load comparison at 50 users from Table IV)
  Figure 5: scalability under variable load (Table IV)
  Figure 3: learning outcomes by subject (Table I post-test means)
  Figure 6: student engagement metrics (Table V)
  Figure 7: qualitative capability comparison with related systems

Figures 1 and 2 are architecture diagrams and are provided in figures/.

  python make_figures.py [output_dir]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_values as pv
from config import FIGURES_DIR

E_COL, B_COL = "#2E6FB0", "#9A9A9A"


def figure4(path):
    t3 = pv.TABLE_III
    panels = [
        ("Mean response time (ms)\nEduMorph-Edge: deployment average", t3["avg_rt_ms"][0], t3["avg_rt_ms"][1], "{:.0f}"),
        ("Mean response time (ms)\nmatched load: 50 users", pv.TABLE_IV[50][0], t3["avg_rt_ms"][1], "{:.0f}"),
        ("95th percentile response time (ms)", t3["p95_rt_ms"][0], t3["p95_rt_ms"][1], "{:.0f}"),
        ("Maximum concurrent users", t3["max_users"][0], t3["max_users"][1], "{:.0f}"),
        ("CPU utilization (%)", t3["cpu_pct"][0], t3["cpu_pct"][1], "{:.1f}"),
        ("Memory usage (GB)", t3["memory_gb"][0], t3["memory_gb"][1], "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.1))
    for ax, (title, e, b, fmt) in zip(axes.ravel(), panels):
        bars = ax.bar(["EduMorph\n-Edge", "Baseline"], [e, b], color=[E_COL, B_COL], width=0.6)
        for bar, v in zip(bars, [e, b]):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, max(e, b) * 1.2)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    fig.suptitle("EduMorph-Edge vs. baseline platform (Table III; matched-load values from Table IV)",
                 fontsize=9.5, fontweight="bold")
    fig.text(0.5, 0.01, "Baseline: up to 50 concurrent users (capacity ceiling). EduMorph-Edge deployment averages span 50-300 users.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure5(path):
    users = sorted(pv.TABLE_IV)
    cols = list(zip(*[pv.TABLE_IV[u] for u in users]))
    titles = ["Response Time (ms)", "CPU (%)", "Memory (GB)", "Success Rate (%)"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, vals, title in zip(axes, cols, titles):
        ax.bar([str(u) for u in users], vals, color="#4E79A7")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Concurrent users", fontsize=9)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.5 if hi > lo else 1
        ax.set_ylim(max(0, lo - pad) if title.startswith("Success") else 0, hi + pad)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure3(path):
    subjects = ["Mathematics", "Science", "English", "Social Studies"]
    e = [pv.TABLE_I[s][2][0] for s in subjects]
    c = [pv.TABLE_I[s][3][0] for s in subjects]
    fig, ax = plt.subplots(figsize=(7.31, 5.0))
    ax.plot(subjects, e, marker="o", color=E_COL, label="EduMorph-Edge")
    ax.plot(subjects, c, marker="o", color="#E8871E", label="Control")
    ax.fill_between(range(len(subjects)), c, e, color="#CFE3F0", alpha=0.6)
    for i, (a, b) in enumerate(zip(e, c)):
        ax.annotate(f"{a:.1f}%", (i, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, color=E_COL)
        ax.annotate(f"{b:.1f}%", (i, b), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color="#E8871E")
    ax.set_ylim(50, 95)
    ax.set_ylabel("Mean post-test score (%)")
    ax.set_xlabel("Subject")
    ax.set_title("Comparative Learning Outcomes by Subject")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure6(path):
    labels = list(pv.TABLE_V)
    e = [pv.TABLE_V[k][0] for k in labels]
    c = [pv.TABLE_V[k][1] for k in labels]
    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.79, 5.0))
    ax.barh([i - 0.2 for i in y], e, height=0.38, color=E_COL, label="EduMorph-Edge")
    ax.barh([i + 0.2 for i in y], c, height=0.38, color=B_COL, label="Control")
    for i, (a, b) in enumerate(zip(e, c)):
        ax.text(a + 0.8, i - 0.2, f"{a:g}", va="center", fontsize=8, color=E_COL)
        ax.text(b + 0.8, i + 0.2, f"{b:g}", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Value (units as in Table V)")
    ax.set_title("Student Engagement Metrics: EduMorph-Edge vs. Control", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, max(e + c) * 1.12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# Capabilities described in each cited publication (1 = described, 0 = not reported).
CAPABILITIES = ["Multimodal\nengagement\ndetection", "On-device\nprivacy-preserving\ninference",
                "Adaptive\ncontent\nsequencing", "Contextual\nintervention\npolicy",
                "Fairness-aware\ndebiasing", "Cloud\nauto-scaling", "Classroom\ntrial\nevaluation"]
SYSTEMS = [
    ("EduMorph-Edge (this work)", [1, 1, 1, 1, 1, 1, 1]),
    ("MSC-Trans [30]",            [1, 0, 0, 0, 0, 0, 0]),
    ("HTM-MDICE [41]",            [1, 0, 0, 0, 0, 0, 0]),
    ("MixUCB [42]",               [0, 0, 0, 1, 0, 0, 0]),
    ("Step-based ITS\n(e.g., Cognitive Tutor) [21]", [0, 0, 1, 0, 0, 0, 1]),
]


def figure7(path, ref_labels=None):
    systems = SYSTEMS if ref_labels is None else [(ref_labels.get(n, n), v) for n, v in SYSTEMS]
    fig, ax = plt.subplots(figsize=(7, 7 / 1.427))
    nr, nc = len(systems), len(CAPABILITIES)
    for i, (name, vals) in enumerate(systems):
        for j, v in enumerate(vals):
            ax.add_patch(plt.Rectangle((j, nr - 1 - i), 1, 1, facecolor="#D6E6F5" if v else "white",
                                       edgecolor="#808080"))
            ax.text(j + 0.5, nr - 0.5 - i, "✓" if v else "–", ha="center", va="center",
                    fontsize=13 if v else 11, color="#1F4E79" if v else "#808080")
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)])
    ax.set_xticklabels(CAPABILITIES, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks([nr - 0.5 - i for i in range(nr)])
    ax.set_yticklabels([s[0] for s in systems], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.5, 0.02, "✓ = capability described in the cited publication;  "
             "– = not reported in the cited publication. Qualitative comparison; no scores implied.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else FIGURES_DIR
    os.makedirs(out, exist_ok=True)
    print(figure3(os.path.join(out, "fig3_learning_outcomes.png")))
    print(figure4(os.path.join(out, "fig4_system_performance.png")))
    print(figure5(os.path.join(out, "fig5_scalability.png")))
    print(figure6(os.path.join(out, "fig6_user_engagement.png")))
    print(figure7(os.path.join(out, "fig7_capability_comparison.png")))
