# EduMorph-Edge: reproducibility package

Reproducibility package for **"EduMorph-Edge: AI-Driven Adaptive Personalization for Student Engagement and Learning Outcomes in Cloud-Supported Classrooms"** (W. Alghabban and R. Salama, *IEEE Access*). Ethics approval: University of Tabuk Local Research Ethics Committee, protocol UT-683-432-2025.

The package has two parts:

1. **Statistical verification.** `verify_statistics.py` recomputes every derived statistic in the paper from the reported summary data and checks it against the manuscript. That covers t-statistics, confidence intervals, effect sizes, the clustering design-effect analysis, the power analysis, retention, attrition and intention-to-treat bounds, percentage changes, latency and cost comparisons, and classifier F1. It needs only numpy and scipy, and runs in seconds.
2. **Reference implementation** of the four AI components in Appendix A:
   - the MSC-Trans engagement classifier [30]
   - the MAML recommender [33]
   - the DRQN path optimizer with prioritized replay [34], [47]
   - LinUCB with adversarial debiasing [31]

   It also includes training and evaluation scripts, a synthetic surrogate of the classifier labelling set, and the privacy tooling.

The study data involve minors and are available on request under the conditions in [`data/README.md`](data/README.md). Run on the synthetic surrogate, the training code demonstrates the pipeline end to end, but its scores are **not** comparable with Table A.II.

## Quick start

```bash
pip install -r requirements-verify.txt
python verify_statistics.py          # 132 checks -> results/verification_report.md
python -m pytest -q tests            # model smoke tests run when torch is installed
python make_figures.py               # regenerates Figures 3-7
```

Full environment (training and evaluation):

```bash
pip install -r requirements.txt      # or: docker build -t edumorph . && docker run edumorph
python train.py                      # synthetic surrogate; add --data labelled.npz for the real set
python evaluate.py                   # held-out test students, macro-averaged F1
python inference.py                  # one adaptive-loop step: engagement state -> intervention
```

## Where each reported result is checked

| Paper item | Script / function | What is recomputed |
|---|---|---|
| Table I | `verify_statistics.py` | t(298), 95% CI of difference, relative improvement, pre-test equivalence |
| Table II | `verify_statistics.py` | Cohen's d = t·√(2/n), pooled-SD d, 95% CI of d |
| Sec. IV-A clustering | `verify_statistics.py` | DEFF = 1 + (m − 1)ρ, effective n, adjusted t (df = 7) for ICC 0.05–0.30 |
| Sec. III-C power | `verify_statistics.py` | n for 80% / 95% power and power at n = 150 (d = 0.45) |
| Table III, Sec. IV-B | `verify_statistics.py` | % improvements, uptime difference, 24.4% cost reduction, 70.9% matched-load latency reduction |
| Figures 3-7 | `make_figures.py` | Fig. 3 from Table I, Fig. 4 from Tables III/IV, Fig. 5 from Table IV, Fig. 6 from Table V, Fig. 7 capability matrix |
| Tables V–VIII | `verify_statistics.py` | % changes, A/B differences, UX differences, subgroup sizes |
| Table IX | `verify_statistics.py` | pre-to-post gains (+37.5%, +3.9%) |
| Sec. IV-G | `verify_statistics.py` | χ², φ, Newcombe CI, OR and RR with CIs, attrition χ², intention-to-treat bounds and tipping point |
| Table A.II | `verify_statistics.py`, `evaluate.py` | per-class F1 from precision and recall, macro-F1 = 0.89 |
| Sec. IV-H fairness | `utils.py` | demographic parity and equalized odds differences |
| Appendix B privacy | `anonymize.py` | k-anonymity (k = 5), Laplace mechanism (ε = 1.2) |

## Key values (as reported in the manuscript)

| Quantity | Value |
|---|---|
| Overall post-test, EduMorph-Edge vs. control | 83.9% vs. 63.3%; t(298) = 39.64; relative difference +32.5% |
| Cohen's d (overall) | 4.58 [4.15, 5.01]; subjects 4.29–4.71 |
| Pre-to-post gain | +37.5% (EduMorph-Edge), +3.9% (control) |
| Clustering-adjusted overall t, ICC = 0.30 | 12.12 (p < 0.001, df = 7) |
| Daily active time | 47.3 vs. 32.1 min (+47.4%) |
| Six-month retention (completers, n = 178) | 68.1% vs. 31.0%; OR = 4.75 [2.52, 8.95]; RR = 2.20 [1.56, 3.09] |
| Follow-up attrition | 40.7% (39.3% vs. 42.0%; χ²(1) = 0.22, p = 0.64) |
| Mean response time (Table III; Table IV at 300 users) | 141 ms; 157 ms |
| p99 at 280–300 users | 276 ms |
| Cloud cost per student-hour | USD 0.0031 vs. 0.0041 (−24.4%) |
| MSC-Trans macro-averaged F1 (held-out test students) | 0.89 |

## Package layout

```
paper_values.py          reported summary data (inputs to verification only)
verify_statistics.py     recomputes all derived statistics; writes results/
config.py                hyperparameters (Appendix A, Table A.I)
model.py                 MSCTrans, MAMLRecommender, DRQNPathOptimizer, DebiasedLinUCB
synthetic_data.py        synthetic labelling-set surrogate and 68/32 student-level split
dataset.py               PyTorch datasets and loaders (synthetic or --data .npz)
train.py / evaluate.py   classifier training (AdamW, warm-up + cosine) and test evaluation
inference.py             single adaptive-loop step
utils.py                 seeding, Cohen's d, fairness metrics
anonymize.py             k-anonymity and differential-privacy utilities
make_figures.py          Figures 3-7 from the reported tables
figures/                 fig1-fig7, numbered as in the paper (3-7 generated by make_figures.py)
results/                 verification_report.md, verification_results.json
tests/                   pytest suite
data/README.md           data-access conditions and file format
```

The classifier uses the 128-dimensional facial action-unit vector that the student device computes on-device (ResNet-18, TFLite INT8; Appendix B). Text inputs are DistilBERT embeddings; `model.finetune_text_encoder` fine-tunes the encoder on the engagement labels and `model.encode_text` produces embeddings from the fine-tuned model (Appendix A.3).

The following deployment components are tied to the production infrastructure and are not included: the on-device facial action-unit extractor, the Kubernetes and Temporal Fusion Transformer auto-scaling configuration, the variational-autoencoder cold-start module, the LLM hint service and its cross-encoder re-ranker. Analyses that require the individual-level study data (DML subgroup effects in Table VIII, the fairness audit in Section IV-H, and per-student tests) can be run only with the data obtained under the conditions in `data/README.md`.

## Tested environment

`results/` was produced with the versions in `environment-tested.txt` (Python 3.13). `requirements-verify.txt` pins the same versions; `requirements.txt` lists the full environment for model training. Pass `--deterministic` to `train.py` for deterministic GPU kernels.

## Licence and citation

Code: MIT License (see `LICENSE`). Please cite the article; citation metadata are in `CITATION.cff`.

## Synthetic missing-analysis suite (explicitly NOT empirical evidence)

Run `python synthetic_analysis.py` to generate the deterministic synthetic datasets and demonstration outputs in `synthetic_results/`. The script includes class counts, school ICC, attrition diagnostics, subgroup contrasts, and fairness metrics. Seeds are defined at the top of the script and written to `synthetic_results/summary.json`. Every dataset and result is synthetic; do not substitute these outputs into the manuscript as observed participant results.


### One-command reproduction

```bash
./run_all_reproducible.sh
```

This runs summary-statistic verification, synthetic analyses, figure regeneration, and tests. For the synthetic outputs alone, use `python synthetic_analysis.py`.
