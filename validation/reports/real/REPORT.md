# Real-data stylised facts — BTCUSDT perp (Binance) + Deribit IV

Sample: 2022-01-01 → 2026-11-18, 42787 hourly bars. Values: pooled (range across calendar years).

## Returns distribution
- 1h: annualised vol 0.500 (years: 0.425 … 0.637), skew -0.156 (years: -0.232 … -0.061), excess kurtosis 13.8 (years: 7.7 … 26.5)
- robust: quantile skew -0.013 (years: -0.041 … 0.020), quantile tail weight vs normal 2.274 (years: 2.013 … 2.370), Hill tail index left 2.91 (years: 2.57 … 3.24) / right 2.89 (years: 2.44 … 3.50)
- aggregational Gaussianity: excess kurtosis 1d 4.74 (years: 1.19 … 5.05), 1w 4.24 (years: -0.32 … 3.65); KS distance to normal 1d 0.078 (years: 0.068 … 0.108), 1w 0.099 (years: 0.050 … 0.173)
- volatility scaling std(h)/(std1·√h): 1d 1.001 (years: 0.940 … 1.049), 1w 1.020 (years: 0.819 … 1.209) (1 = random walk)

## Dependence
- ACF of r at lag 1 -0.010 (years: -0.044 … 0.005), max |ACF| lags 1–24 0.015 (years: 0.021 … 0.076), Ljung–Box(24) p-value 0.003 (years: 0.000 … 0.560)
- ACF of |r|: lag 1 0.282 (years: 0.238 … 0.283), 24 0.165 (years: 0.097 … 0.188), 168 0.139 (years: 0.084 … 0.184), 720 0.057 (years: 0.025 … 0.048)
- ACF of deseasonalised |r|: lag 1 0.263 (years: 0.225 … 0.266), 24 0.152 (years: 0.083 … 0.184), 168 0.084 (years: 0.020 … 0.072), 720 0.053 (years: 0.016 … 0.046); power-law slope (24–720 h) -0.458 (years: -1.043 … -0.628); seasonal bump at 24 h 0.013 (years: 0.000 … 0.039)
- leverage: mean corr(r_t, r²_{t+τ}) τ=1…24 -0.0118 (years: -0.0200 … 0.0178) vs τ=−24…−1 0.0034 (years: 0.0001 … 0.0100)
- Zumbach time-reversal asymmetry 0.1063 (years: 0.0723 … 0.1234)
- extremes (|r| > q99): inter-arrival CV 1.83 (years: 1.20 … 1.69) (1 = Poisson), share within 24 h 0.51 (years: 0.43 … 0.49) vs 0.21 (years: 0.21 … 0.21) if independent
- max drawdown in 7-day windows: median 0.057 (years: 0.047 … 0.083), q90 0.123 (years: 0.088 … 0.152) (log)

## Seasonality
- intraday |r| max/min hour ratio 2.13 (years: 1.99 … 2.34); weekend |r| relative 0.69 (years: 0.66 … 0.73); funding hours (00/08/16) |r| relative 1.02 (years: 1.00 … 1.05)

## Bars and volume
- wicks: upper 0.283 (years: 0.274 … 0.295), lower 0.292 (years: 0.281 … 0.302), body 0.425 (years: 0.408 … 0.445) of range; corr(log range, |r|) 0.826 (years: 0.807 … 0.835); Parkinson / close-close variance 1.151 (years: 1.089 … 1.222)
- log volume ACF lag 1 0.794 (years: 0.721 … 0.742), 168 0.577 (years: 0.431 … 0.455); corr(log vol, |r|) 0.542 (years: 0.559 … 0.623); corr(taker imbalance, r) 0.503 (years: 0.482 … 0.576), with next bar -0.0036 (years: -0.0180 … 0.0071)

## Open interest
- corr(bar return, ΔlogOI) 0.110 (years: 0.041 … 0.194); corr(|r|, ΔlogOI) -0.173 (years: -0.344 … -0.038); mean ΔlogOI in worst 1% bars -0.0113 (years: -0.0221 … -0.0015) / best 1% bars -0.0036 (years: -0.0116 … 0.0026) / all 0.00001 (years: -0.00004 … 0.00004)

## Funding
- mean 0.000060 (years: 0.000031 … 0.000109) per 8h (annualised 0.066 (years: 0.033 … 0.119)), share negative 0.15 (years: 0.08 … 0.24); ACF per settlement lag 1 0.772 (years: 0.553 … 0.821), 21 (1 week) 0.456 (years: 0.081 … 0.534)
- corr with trailing 24h / 168h return 0.160 (years: -0.003 … 0.266) / 0.293 (years: 0.088 … 0.364); with next 24h return -0.032 (years: -0.141 … 0.006)

## Options (Deribit constant-maturity IV)
- 30d ATM mean 0.507 (years: 0.402 … 0.656); VRP (ATM30 − next-30d RV) mean 0.029 (years: -0.011 … 0.046), positive 0.65 (years: 0.52 … 0.73) of the time
- spot–vol: corr(r, ΔATM30) -0.127 (years: -0.326 … 0.100), beta -0.170 (years: -0.468 … 0.161); corr(r, ΔRR25_30d) 0.292 (years: 0.239 … 0.384); corr(r, ΔBF25_30d) -0.068 (years: -0.104 … -0.030)
- RR25_30d mean -0.023 (years: -0.085 … 0.015), positive 0.37 (years: 0.02 … 0.72) of time, corr with trailing 7d return 0.472 (years: 0.417 … 0.555)
- term slope ATM7 − ATM90 mean -0.039 (years: -0.061 … -0.008), inverted 0.23 (years: 0.15 … 0.37) of time, corr with trailing 24h RV 0.494 (years: 0.381 … 0.668); ATM30 half-life 27.1 (years: 6.3 … 25.9) days

Full table: `fingerprint.csv`; curves: `curves/`; plots: `figures/`.
