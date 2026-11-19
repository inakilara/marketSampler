# structural+options vs real fingerprint

3 synthetic paths × 42787 hours, cut into 1-year chunks; status = synthetic chunk median vs real per-year [min, max].
Fit + generation + characterisation: 124 s.

**Overall: 129 PASS, 21 NEAR, 1 FAIL out of 151 graded facts.**

| group | PASS | NEAR | FAIL |
|---|---|---|---|
| bars | 5 | 3 | 0 |
| dependence | 15 | 2 | 0 |
| distribution | 15 | 3 | 0 |
| extremes | 5 | 2 | 0 |
| funding | 25 | 0 | 0 |
| joint | 25 | 2 | 1 |
| leverage | 1 | 2 | 0 |
| open_interest | 9 | 0 | 0 |
| options | 18 | 5 | 0 |
| seasonality | 5 | 0 | 0 |
| volume | 6 | 1 | 0 |
| zumbach | 0 | 1 | 0 |

## Not passing

| fact | status | real years [min, max] | synthetic chunk median |
|---|---|---|---|
| distribution.r1h_std | NEAR | [0.00454, 0.006805] | 0.004295 |
| distribution.r1h_skew | NEAR | [-0.2323, -0.06103] | -0.2756 |
| distribution.r1h_ann_vol | NEAR | [0.4251, 0.6371] | 0.4021 |
| dependence.acf_abs_720 | NEAR | [0.0248, 0.04844] | 0.0214 |
| dependence.acf_absds_powerlaw_slope | NEAR | [-1.043, -0.6282] | -0.477 |
| leverage.leverage_mean_m24_m1 | NEAR | [0.0001097, 0.01004] | -0.001564 |
| leverage.leverage_1 | NEAR | [-0.02894, 0.007035] | 0.01093 |
| zumbach.zumbach_asymmetry | NEAR | [0.07231, 0.1234] | 0.05046 |
| extremes.exceed_interarrival_cv | NEAR | [1.196, 1.685] | 1.688 |
| extremes.maxdd_30d_median | NEAR | [0.1053, 0.2322] | 0.1026 |
| bars.upper_wick_mean | NEAR | [0.2744, 0.2948] | 0.3047 |
| bars.body_mean | NEAR | [0.4078, 0.4448] | 0.3926 |
| bars.corr_range_absr | NEAR | [0.8069, 0.8348] | 0.8023 |
| volume.acf_logvol_168 | NEAR | [0.4314, 0.4545] | 0.4227 |
| options.beta_rr25_30d | NEAR | [0.3874, 0.609] | 0.3817 |
| options.corr_rr25_30d | NEAR | [0.2387, 0.3837] | 0.2081 |
| options.vrp30_share_positive | NEAR | [0.5237, 0.731] | 0.7494 |
| options.term_slope_share_inverted | NEAR | [0.1493, 0.3694] | 0.1476 |
| options.corr_term_slope_trailing_rv_24h | NEAR | [0.3814, 0.668] | 0.3044 |
| joint.corr_r_drr25_30d | NEAR | [0.2387, 0.3837] | 0.208 |
| joint.corr_dlogvol_dbf25_30d | NEAR | [-0.01633, 0.02061] | -0.01669 |
| joint.corr_datm_30d_dbf25_30d | FAIL | [-0.6036, -0.4876] | -0.3483 |

Full table: `score.csv`; overlays: `figures/`.
