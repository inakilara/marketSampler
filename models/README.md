# models/ — synthetic BTC perp paths (OHLCV + funding + OI)

Goal: hourly synthetic panels with the columns of `validation/panel.py` (open, high, low, close, volume,
taker_buy_volume, oi, funding) whose stylised facts match the real fingerprint in
`validation/reports/real/`. The option chain comes later and will hang off the same latent volatility state.

## What the real data says the model must reproduce

| fact (from `validation/reports/real`) | implication for the model |
|---|---|
| Hill tail index ≈ 2.9 both sides, excess kurtosis 8–26 by year | heavy-tailed shocks, not Gaussian |
| \|r\| ACF 0.26 at 1 h, still 0.05 at 30 days, power-law decay | long-memory volatility, more than one GARCH time scale |
| Zumbach asymmetry ≈ 0.11 | volatility driven by past coarse (multi-hour) moves → HAR-type cascade |
| weekly kurtosis ≈ 4 pooled but ≈ 0 within single years | slow volatility **regimes** across months, beyond clustering |
| leverage weak and sign-changing by year | leverage coefficient regime-dependent and small |
| intraday \|r\| max/min ≈ 2.1, weekends ≈ 0.7 | deterministic hour-of-week seasonality for volatility and volume |
| Parkinson / close-close variance ≈ 1.15, wick fractions ≈ 0.28 / 0.29 | intrabar paths must be generated, not just closes |
| log-volume ACF 0.79 (1 h), 0.58 (1 week), corr with \|r\| 0.54, taker imbalance corr with r 0.50 | volume = persistent level + coupling to the same-bar shock; signed flow tied to the return |
| ΔlogOI: corr with r +0.11, with \|r\| −0.17, −1.1 % in the worst 1 % bars | OI responds to returns, with liquidation drops in large moves |
| funding: ACF 0.77 per settlement, corr 0.29 with the trailing week, ~0 with the future | persistent funding that follows past price action and does not predict it |

## Candidate approaches

| approach | realism | control / scenarios | data needs | verdict |
|---|---|---|---|---|
| **A. Structural econometric model** (regime-switching multi-scale volatility + conditional sub-models for bars, volume, OI, funding) | high when the components are right; every fact maps to a parameter | full: set the regime, vol level, drift, funding stress | fine with 5 years of hourly data | **core** |
| B. Block bootstrap of real history (stationary bootstrap of joint multi-day blocks) | very high locally (copies the real data) | none; cannot create unseen scenarios; joins between blocks break the levels of OI and funding | — | baseline / benchmark only |
| C. Deep generative (TimeGAN, Quant GAN, diffusion, transformers) | can capture complex joint structure | poor; black box | ~1,800 days is small for long-memory, heavy-tailed series; overfitting/memorisation risk | not now; possible later on residuals |
| D. Agent-based market simulator | emergent facts | indirect | hard to calibrate | out of scope |

## Proposal: A, with empirical (bootstrapped) innovations

A hierarchical generator, one hour at a time, each layer conditioned on the ones above:

```
regime s_t (hidden Markov, ~2–3 states: calm / normal / stressed, dwell times of weeks)
  └─ volatility  log σ_t = seasonal(hour-of-week) + regime level μ_s
                          + HAR cascade of past |r| over 1 h, 1 d, 1 w, 1 month (long memory, Zumbach)
                          + regime-dependent leverage term
      └─ return    r_t = drift_s + σ_t · z_t
          └─ bar   open = previous close; high/low from an intrabar path (sub-steps, heavy-tailed) conditioned
                   on r_t and σ_t, calibrated to the real wick/body distribution and Parkinson ratio
          └─ volume  log V_t = seasonal + persistent level (AR/HAR on log V) + β·|z_t| + regime
                     taker share = logistic(γ·z_t + noise)
          └─ OI      ΔlogOI_t = a_s + b·r_t + c·|z_t| (liquidation, c < 0) + AR + noise, OI kept on a slow trend
          └─ funding (every 8 h) f = AR(1) toward a regime mean + k·(trailing returns) + m·(ΔOI) + noise,
                     clipped to the exchange caps; positive base rate
```

**Innovations:** instead of assuming a Student-t, the joint shock vector (z_r, z_volume, z_oi, z_funding) is taken
from the model's own standardised real residuals (filtered historical simulation), resampled in short
blocks. This keeps the empirical tails, the gain/loss asymmetry and the same-bar cross-correlations without
parametrising them, while the dynamics (clustering, regimes, seasonality) come from the model, so the paths
are new.

**Why this choice:** each stylised fact maps to an explicit component, so a failing check points to the
part to fix. The model calibrates robustly on 5 years of data, can generate scenarios (force a stressed
regime, set a starting vol or funding level), and exposes a latent vol/regime state that the option layer
(ATM ≈ vol forecast + risk premium, RR/BF tied to regime and returns) can reuse in step 2.

## Status (scored with `validation/compare.py`, 3 paths × 42,787 h, 110 graded facts)

| model | PASS | NEAR | FAIL | notes |
|---|---|---|---|---|
| `bootstrap` (benchmark) | 100 | 5 | 5 | copies real blocks: fails long-memory (\|r\| ACF at 30 d) and volume persistence; no new scenarios |
| `structural` | 96 | 14 | 0 | all groups without failures; funding 25/25, OI 9/9, seasonality 5/5, joint 10/10 |

Remaining NEARs (small, documented in `validation/reports/structural/SUMMARY.md`): a typical synthetic year
is slightly calmer than real ones (ann. vol 0.40 vs 0.43–0.64), slightly too negative 1h skew, Zumbach
asymmetry 0.054 vs 0.072–0.123, wick/body fractions ~0.01 off, leverage curve near zero at τ = ±1.

Lessons from fitting (kept in code comments):
- volume regressed on *deseasonalised* volatility (otherwise intraday seasonality is counted twice);
  its persistent level fitted within calendar years (pooled years look like a unit root);
- intrabar range needs 1-minute bridge steps and intrabar volatility linked to excess volume;
- funding is a one-sided dead band (continuous below 0.01 %, pinned at it, above only in bull phases);
  band width 0.01 % chosen by simulation; funding feeds back weakly and negatively on next returns.

Usage:
```python
import models
m = models.load("structural")            # saved state (0.2 s); fits on data/cache and saves it if missing (~1.5 min)
m = models.load("structural", refit=True) # after refreshing the data
path = m.generate(24 * 365, seed=42, start="2027-01-01")  # DataFrame: open high low close volume taker_buy_volume oi funding
```
The fitted state is `models/params/structural_state.npz` (gitignored: it holds residuals derived from the data);
`models/params/structural.json` is a readable parameter summary.

### Scenarios (keyword arguments of `generate`; "path" = scalar or one value per hour)

| argument | effect | checked (8 seeds × 1 y, medians) |
|---|---|---|
| `vol_level` (path) | long-run annualised vol (variance target); also rescales the starting history. A path is a slow regime change: the cascade moves to the new level over months | 0.30 / baseline / 0.80 → yearly vol 0.23 / 0.39 / 0.62; step 0.4→1.0 reaches ~0.78 after 6 months |
| `start_vol` | vol of the last 30 days before the start; `"calm"` / `"normal"` / `"stressed"` = 10/50/90th pct of real 30-day vol (0.32 / 0.46 / 0.67) | first month 0.32 / 0.41 / 0.56 (mean reversion inside the month) |
| `vol_mult` (path) | transient multiplier on conditional vol (not fed back to the cascade, which would be explosive) | ×2.5 for 2 weeks: 0.36 → 0.82 → 0.31 after |
| `drift` (path) | annualised log drift | ±0.5 → log return +0.39 / −0.42 |
| `jumps` | `{hour or timestamp: log return}` forced bars; the shock passed to volume / OI / range / vol cascade is capped at 8σ (real crash bars ~7–8σ) | −15 %: volume 36× prior day (real big bars 2–40×), OI −6.5 %, next-week vol ×1.45 (real 1.1–2×) |
| `funding_shift` (path) | shift of the long-run latent funding premium per 8 h; the funding→return feedback acts on deviations from the shifted level | +3e-4 → mean funding 2.9e-4 (baseline 0.6e-4) |
| `funding0`, `p0`, `oi0` | starting funding rate, price, open interest | funding0 = 5e-4 → 4.2e-4 over the first 3 days |

Note `vol_level` is the unconditional level; because vol is long-memory and right-skewed, the median single year
realises ~0.77 of it (same as real: pooled 0.50, typical year lower).

`python validation/compare.py --model structural [--refit]` re-scores; `python models/plot_sample.py` plots paths;
`python models/plot_scenarios.py` plots every scenario against the baseline
(`validation/reports/structural/figures/scenarios.png`).

## Options layer (`models/options.py`)

Hourly constant-maturity IV surface: ATM, RR25, BF25, RR10, BF10 at 7 / 14 / 30 / 60 / 90 / 180 days (the columns of
`data/cache/deribit_BTC_iv_surface_1h.parquet`). It is a function of the price path only (returns, funding), so
it attaches to the structural generator, the bootstrap, or real history.

```python
m = models.load("structural+options")                     # both saved states
path = m.generate(24 * 365, seed=1, start_vol="stressed", atm_shift=0.1)   # + atm_*, rr25_*, ... columns
surf = models.load("options").generate(real_or_synthetic_path, seed=1)     # surface for any path
```

Per series (log ATM; RR/ATM and BF/ATM; 30d modelled directly, other tenors as spreads to 30d):

| component | what it carries | fitted on |
|---|---|---|
| long-run level `a + β'F_t` | F = log realised vol over 1 d / 1 w / 30 d / 90 d, trailing 1 w / 30 d return, \|1 w return\|, funding: variance risk premium, skew follows price, short tenors track recent RV | hourly OLS |
| slow deviation (half-life ≤ 6 months) + medium deviation (days) | month-long skew / butterfly regimes; persistent spot-vol and skew response to the bar return | daily ACF up to 60 d + daily regression on the day's returns |
| fast transient (hours) | same-hour overshoot: 30d RR25 moves 0.8 × the bar return within the hour, ~0.4 stays | hourly impulse response |
| measurement noise | the reconstructed surface's hourly noise (10–90 % of hourly RR/BF changes), so synthetic surfaces are "measured" like the real one | hourly change variance |

Shocks are resampled jointly across the 30 series from real residual rows (slow: daily rows; medium + noise:
the same real hour), so tenors and quantities co-move as in the data.

Score (`python validation/compare.py --model structural+options`, all 151 facts): **130 PASS, 20 NEAR, 1 FAIL**;
options 19 / 4 / 0, the OHLCV / funding / OI part unchanged. Checked by hand: driven by the *real* price path the
layer reproduces the real options facts (ATM30 mean 0.50 vs 0.51, VRP +2.4 vs +2.9 vol, RR25 beta 0.49 vs 0.48,
ATM30 daily ACF at 30 d 0.63 vs 0.73); no calendar-arbitrage violations; negative BF25 2.2 % of hours (real
reconstruction 1.1 %). `python models/plot_options.py` → `validation/reports/structural+options/figures/surface_paths.png`.

Known gaps: same-hour corr(dATM30, dBF25) −0.35 vs −0.49…−0.60 (mostly a reconstruction artefact: BF = wings − ATM
shares ATM's measurement error); crash-time term-structure inversions are milder than real (7d − 90d peaks
~+10 vol vs +20…45); the real BF10 downtrend over 2022–2026 is not reproduced (stationary model).
Scenario knobs: `atm_shift` (log ATM level, e.g. 0.1 = +10 %), `rr_shift` (RR/ATM level), plus every price-path
scenario (vol, jumps, drift...) which the surface follows.

## Implementation plan

| step | content | validated by |
|---|---|---|
| 0 | `validation/compare.py`: score a synthetic panel against the real fingerprint (inside inter-year range? curve distances) + block-bootstrap benchmark (approach B) | — |
| 1 | returns core: seasonality, HAR volatility, regimes, FHS innovations | distribution, dependence, leverage, Zumbach, seasonality, extremes |
| 2 | intrabar OHLC | bars |
| 3 | volume + taker flow | volume |
| 4 | open interest | open_interest |
| 5 | funding | funding, joint |
| 6 | API + scenarios: `generate(n_hours, seed, start, regime=..., ...)` → panel; parameters saved to JSON | all |

Planned layout: `models/base.py` (interface), `models/calibrate.py` (fit from `data/cache`),
`models/structural.py` (the generator), `models/bootstrap.py` (benchmark), `models/params/*.json`.
