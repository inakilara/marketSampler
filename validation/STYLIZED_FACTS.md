# Stylised facts for validating synthetic BTC market paths

Goal: a fixed, reproducible *fingerprint* of the real hourly data (BTC perp OHLCV, funding, open interest,
Deribit IV surface) that synthetic paths must reproduce. Every fact below is computed by `validation/facts.py`
with the same code for real and synthetic panels, so they can later be compared number by number.

## 1. Is the proposed list a valid characterisation?

Yes — it covers the core of the classic stylised-facts literature (Cont 2001: heavy tails, absence of linear
autocorrelation, volatility clustering, slow decay of |r| autocorrelation, leverage effect, aggregational
Gaussianity, volume–volatility correlation) and adds the cross-asset part that matters for this project
(funding, OI, options). Recent work confirms these facts for Bitcoin, with two crypto-specific nuances:
a leverage effect that is weaker and regime-dependent (sign flips in bull phases), and somewhat slower decay
of linear autocorrelation than traditional assets.

Points to tighten:

| proposal | caveat | improvement |
|---|---|---|
| moments (mean, std, skew, kurtosis) | sample kurtosis/skew of heavy-tailed returns are dominated by a few bars and unstable; mean is statistically indistinguishable from 0 at 1h | report them, but validate on **robust** shape measures: quantiles, tail index (Hill), quantile skew/kurtosis, gain/loss asymmetry |
| ACF/PACF of \|r\| | — | also r², over long lags (up to ~1 month) and fit the power-law decay exponent; add ACF of r itself (should be ~0) |
| leverage effect | in BTC weak and regime-dependent | use the lagged correlation function L(τ)=corr(r_t, r²_{t+τ}) for τ = −48…+48 h, plus per-year values |
| aggregational Gaussianity | — | track kurtosis, tail index and a normality distance vs horizon (1h…1w), and the **volatility scaling** std(h)/√h (anomalous diffusion / mean reversion) |
| relations with range, volume, OI, funding | — | contemporaneous + **lead–lag cross-correlation functions** (both signs of lag), using |r| and signed r separately |
| price action vs option chain | — | specific, well-documented effects: variance risk premium, spot–vol beta, skew response, term-structure inversion in stress, IV persistence |

## 2. Extensions

Univariate (returns):
- **Intraday & weekly seasonality**: mean |r| and volume by UTC hour and weekday; 8-hour funding settlements
  (00/08/16 UTC) leave a footprint. A generator without seasonality fails this immediately.
- **Time-reversal asymmetry (Zumbach effect)**: past coarse-grained volatility predicts future fine
  volatility better than the reverse; GARCH-like models reproduce it, many ML generators do not.
- **Extremes**: tail index per side, clustering of exceedances (inter-arrival times of |r| > q99),
  drawdown distribution over fixed windows.
- **Non-stationarity**: compute everything per calendar year as well as pooled — the synthetic data should
  fall inside the range spanned by the real years, not only match the pooled average.

OHLC / volume:
- **Bar geometry**: upper/lower wick and body as fractions of range; Parkinson vs close-to-close variance ratio.
- **Volume**: log-volume distribution and persistence; taker-buy imbalance vs return (order-flow sign).

Derivatives:
- **Open interest**: OI changes vs returns (new longs vs short covering), large moves with OI drops
  (liquidation cascades), OI persistence.
- **Funding**: persistence, its response to trailing returns, its (weak) predictive power, funding vs OI.

Options (Deribit, constant maturity):
- **Variance risk premium**: 30d ATM IV vs the realised vol over the following 30 days.
- **IV vs trailing realised vol**; **spot–vol beta** dATM/dr by tenor (sign changes across regimes);
  **skew dynamics** dRR25/dr and RR level vs trailing return; **BF vs vol-of-vol**;
  **term structure**: ATM7 − ATM90 slope, inverted in stress; IV persistence (ACF / half-life).

Joint structure:
- Correlation matrix of hourly changes of all series and a lead–lag map (max |cross-correlation| and its lag).

## 3. How synthetic data will be scored (next step, `validation/compare.py`)

Stylised facts give interpretable checks but no single score; the literature has no consensus metric,
so the plan is to combine:
1. **Fact-by-fact distances**: Wasserstein / KS distance of marginals at several horizons, RMSE between ACF,
   leverage and cross-correlation curves, relative error of scalar facts — each judged against the
   spread across real calendar years (a synthetic value inside the real inter-year range passes).
2. **Discriminative score**: a classifier trained to tell real from synthetic windows (≈50 % accuracy = indistinguishable).
3. **Predictive score (TSTR)**: train a simple forecaster on synthetic, test on real.

## Sources
- Cont (2001), *Empirical properties of asset returns: stylized facts and statistical issues*, Quantitative Finance 1(2).
- [de Sousa Filho et al., The leverage effect and other stylized facts displayed by Bitcoin returns](https://arxiv.org/abs/2004.05870v2)
- [International Financial Markets Through 150 Years: Evaluating Stylized Facts](https://arxiv.org/pdf/2504.08611)
- [Can GANs Learn the Stylized Facts of Financial Time Series?](https://arxiv.org/pdf/2410.09850)
- [The ETS challenges: evaluation of simulated financial time series](https://arxiv.org/pdf/1811.07792)
- [A Data-driven Market Simulator for Small Data Environments](https://arxiv.org/pdf/2006.14498)
- [Evaluating generative models for synthetic financial data](https://arxiv.org/html/2512.21791v1)
- [Bitcoin options: finding edge in four years of volatility regimes (Deribit Insights)](https://insights.deribit.com/industry/bitcoin-options-finding-edge-in-four-years-of-volatility-regimes/)
- [Unveiling the Link Between Option Tails and Spot Volatility Correlations (Volmex)](https://volmex.finance/Unveiling-the-Link-Between-Option-Tails-and-Spot-Volatility-Correlations.pdf)
