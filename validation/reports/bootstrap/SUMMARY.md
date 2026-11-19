# bootstrap vs real fingerprint

3 synthetic paths × 42787 hours, cut into 1-year chunks; status = synthetic chunk median vs real per-year [min, max].
Fit + generation + characterisation: 99 s.

**Overall: 100 PASS, 5 NEAR, 5 FAIL out of 110 graded facts.**

| group | PASS | NEAR | FAIL |
|---|---|---|---|
| bars | 7 | 1 | 0 |
| dependence | 14 | 1 | 2 |
| distribution | 18 | 0 | 0 |
| extremes | 7 | 0 | 0 |
| funding | 24 | 1 | 0 |
| joint | 10 | 0 | 0 |
| leverage | 3 | 0 | 0 |
| open_interest | 9 | 0 | 0 |
| seasonality | 5 | 0 | 0 |
| volume | 2 | 2 | 3 |
| zumbach | 1 | 0 | 0 |

## Not passing

| fact | status | real years [min, max] | synthetic chunk median |
|---|---|---|---|
| dependence.acf_abs_720 | FAIL | [0.0248, 0.04844] | 0.006178 |
| dependence.acf_absds_168 | NEAR | [0.01969, 0.07231] | 0.008874 |
| dependence.acf_absds_720 | FAIL | [0.01595, 0.0457] | -0.004344 |
| bars.acf_logrange_1 | NEAR | [0.6334, 0.6716] | 0.6729 |
| volume.logvol_std | FAIL | [0.7825, 0.8759] | 0.9335 |
| volume.acf_logvol_1 | FAIL | [0.7211, 0.7417] | 0.7851 |
| volume.acf_logvol_24 | NEAR | [0.3527, 0.4385] | 0.4647 |
| volume.acf_logvol_168 | FAIL | [0.4314, 0.4545] | 0.3567 |
| volume.corr_logvol_absr | NEAR | [0.5587, 0.6233] | 0.549 |
| funding.corr_funding_trailing_r_720h | NEAR | [0.149, 0.6487] | 0.1255 |

Full table: `score.csv`; overlays: `figures/`.
