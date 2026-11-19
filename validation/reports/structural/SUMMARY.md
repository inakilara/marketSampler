# structural vs real fingerprint

3 synthetic paths × 42787 hours, cut into 1-year chunks; status = synthetic chunk median vs real per-year [min, max].
Fit + generation + characterisation: 55 s.

**Overall: 96 PASS, 14 NEAR, 0 FAIL out of 110 graded facts.**

| group | PASS | NEAR | FAIL |
|---|---|---|---|
| bars | 4 | 4 | 0 |
| dependence | 15 | 2 | 0 |
| distribution | 15 | 3 | 0 |
| extremes | 6 | 1 | 0 |
| funding | 25 | 0 | 0 |
| joint | 10 | 0 | 0 |
| leverage | 1 | 2 | 0 |
| open_interest | 9 | 0 | 0 |
| seasonality | 5 | 0 | 0 |
| volume | 6 | 1 | 0 |
| zumbach | 0 | 1 | 0 |

## Not passing

| fact | status | real years [min, max] | synthetic chunk median |
|---|---|---|---|
| distribution.r1h_std | NEAR | [0.00454, 0.006805] | 0.004296 |
| distribution.r1h_skew | NEAR | [-0.2323, -0.06103] | -0.2768 |
| distribution.r1h_ann_vol | NEAR | [0.4251, 0.6371] | 0.4022 |
| dependence.acf_abs_720 | NEAR | [0.0248, 0.04844] | 0.0218 |
| dependence.acf_absds_powerlaw_slope | NEAR | [-1.043, -0.6282] | -0.4702 |
| leverage.leverage_mean_m24_m1 | NEAR | [0.0001097, 0.01004] | -0.001483 |
| leverage.leverage_1 | NEAR | [-0.02894, 0.007035] | 0.01027 |
| zumbach.zumbach_asymmetry | NEAR | [0.07231, 0.1234] | 0.05404 |
| extremes.exceed_interarrival_cv | NEAR | [1.196, 1.685] | 1.688 |
| bars.upper_wick_mean | NEAR | [0.2744, 0.2948] | 0.3048 |
| bars.lower_wick_mean | NEAR | [0.2808, 0.3021] | 0.3021 |
| bars.body_mean | NEAR | [0.4078, 0.4448] | 0.3924 |
| bars.corr_range_absr | NEAR | [0.8069, 0.8348] | 0.8023 |
| volume.acf_logvol_168 | NEAR | [0.4314, 0.4545] | 0.4227 |

Full table: `score.csv`; overlays: `figures/`.
