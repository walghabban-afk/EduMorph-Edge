# Package completeness and reproducibility audit

This ZIP contains the original reproducibility package plus synthetic demonstration analyses.

## Reproduction
Run `./run_all_reproducible.sh` from this directory. Tested in this environment: summary verification, synthetic analyses, figure regeneration, and 11 tests.

## Synthetic analyses
`synthetic_analysis.py` generates CSV data and JSON/CSV outputs under `synthetic_results/`. Seeds are listed in the script and emitted in `summary.json`. The simulated class counts, ICC, attrition, subgroup effects, and fairness metrics are **not observed study results** and must not be represented as empirical findings in a paper.

## Contents
- Original model/reference implementation, data handling, verification, figures and tests
- Synthetic data generator and analysis outputs
- Fixed seeds and deterministic generation
- Test suite and one-command runner
- SHA-256 manifest for package files
