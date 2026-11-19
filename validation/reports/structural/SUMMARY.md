# structural vs real fingerprint

3 synthetic paths × 42787 hours, cut into 1-year chunks; status = synthetic chunk median vs real per-year [min, max].
Fit + generation + characterisation: 19 s.

**Overall: 92 PASS, 18 NEAR, 0 FAIL out of 110 graded facts.**

| group | PASS | NEAR | FAIL |
|---|---|---|---|
| bars | 3 | 5 | 0 |
| dependence | 15 | 2 | 0 |
| distribution | 16 | 2 | 0 |
| extremes | 5 | 2 | 0 |
| funding | 24 | 1 | 0 |
| joint | 10 | 0 | 0 |
| leverage | 2 | 1 | 0 |
| open_interest | 9 | 0 | 0 |
| seasonality | 3 | 2 | 0 |
| volume | 5 | 2 | 0 |
| zumbach | 0 | 1 | 0 |

## Not passing

| fact | status | real years [min, max] | synthetic chunk median |
|---|---|---|---|
| distribution.r1h_std | NEAR | [0.00454, 0.006805] | 0.004522 |
| distribution.r1h_ann_vol | NEAR | [0.4251, 0.6371] | 0.4234 |
| dependence.acf_abs_1 | NEAR | [0.238, 0.2829] | 0.2877 |
| dependence.pacf_abs_1 | NEAR | [0.238, 0.2829] | 0.2877 |
| leverage.leverage_mean_m24_m1 | NEAR | [0.0001097, 0.01004] | -0.001388 |
| zumbach.zumbach_asymmetry | NEAR | [0.07231, 0.1234] | 0.06432 |
| seasonality.hour_absr_max_over_min | NEAR | [1.988, 2.335] | 1.98 |
| seasonality.weekend_vol_rel | NEAR | [0.5497, 0.6315] | 0.5461 |
| extremes.exceed_interarrival_cv | NEAR | [1.196, 1.685] | 1.913 |
| extremes.exceed_within_24h_share | NEAR | [0.4253, 0.4943] | 0.5287 |
| bars.upper_wick_mean | NEAR | [0.2744, 0.2948] | 0.3023 |
| bars.lower_wick_mean | NEAR | [0.2808, 0.3021] | 0.3037 |
| bars.body_mean | NEAR | [0.4078, 0.4448] | 0.394 |
| bars.corr_range_absr | NEAR | [0.8069, 0.8348] | 0.8015 |
| bars.parkinson_over_cc_var | NEAR | [1.089, 1.222] | 1.23 |
| volume.acf_logvol_1 | NEAR | [0.7211, 0.7417] | 0.7475 |
| volume.acf_logvol_168 | NEAR | [0.4314, 0.4545] | 0.427 |
| funding.corr_funding_next_r_8h | NEAR | [-0.1326, -0.02148] | -0.02015 |

Full table: `score.csv`; overlays: `figures/`.
