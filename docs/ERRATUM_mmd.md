# Erratum 2026-10-19: two minimum-meaningful-difference (MMD) figures in the preregistration

This is a record of a discrepancy, not a revision. No preregistered file was edited.

## What each file says

- `evidence/prereg/strata_v0_v2.md:52` (original): "MMD = 15 percentage points on the primary ... the power analysis must confirm or revise this number".
- `evidence/prereg/power_analysis_amendment.md:42` (revision after the power simulation): "**MMD = 20 percentage points** on the primary endpoint (overall live-verified recovery, V2 - V0)", with the outcome rules "in section 4 apply with MMD = 20" (line 46). The file contains no 7.2 figure.
- `evidence/prereg/test_protocol.md:42` (sealed TEST protocol, frozen 2026-10-18): "**Minimum meaningful difference 7.2 points** (`docs/prereg/power_analysis_amendment.md`; paired seed sd ~5.1 over 6 seeds)". It cites the amendment as the source, but the amendment states 20.

The cited file does not support 7.2, and `evidence/prereg/` records no derivation.

## What the sealed-TEST analysis applied

7.2. `scripts/analyze_test.py:22` sets `MMD = 7.2`; `evidence/results/test/TEST_REPORT_analysis.json` records `"mmd": 7.2`; `evidence/results/TEST_REPORT.md:35` and `evidence/results/test/TEST_REPORT_analysis.md:27` print "MMD 7.2 points" and classify the result "positive effect, size unresolved". 20 was not applied anywhere in the TEST outputs.

## Implication for V2 vs V0 +10.36 points

Two-level bootstrap 95% CI [+5.83, +15.26] (`evidence/results/TEST_REPORT.md:32`). Against 7.2 the CI excludes 0 and straddles the threshold (lower bound below, upper bound above): "positive effect, size unresolved". Against 20 the upper bound (+15.26) is below the threshold and the lower bound is above 0: under the section 4 rules this is "practically insignificant" (a real effect below the MMD). Against the original 15 the CI also straddles the threshold (15.26 is just above it).

## How to read the result

The point estimate (+10.36) is above 7.2 and below 20; classification depends on which conflicting frozen MMD is binding. The TEST run is exploratory (two task families, 6 seeds); neither figure was re-derived and the power simulation was not rerun.
