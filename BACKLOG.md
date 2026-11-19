# Backlog

## Done
- [x] Data: Binance 1h OHLCV, funding, 1h open interest (`data/fetch_binance.py`)
- [x] `validation/`: stylised-facts research (`STYLIZED_FACTS.md`), facts library (`facts.py`), real-data
      characterisation (`characterize_real.py` -> `reports/real/`) for returns, bars, volume, OI, funding

- [x] Data: Deribit option trades since 2022 -> hourly constant-maturity IV surface (42,791 h; 30d ATM coverage 98 %,
      corr with DVOL 0.982, ATM 3.3 vol below DVOL on average) and options facts in `validation/reports/real`

## Open checks
- [ ] 2022 reconstruction bias: 30d ATM ~5.6 vol below DVOL in Jan-2022 vs ~3.3 average / ~2.2 live today
      (validate on a free Tardis first-of-month full chain)

## Next — synthetic OHLCV + funding + OI (`models/`, proposal in `models/README.md`)
- [x] 0. `validation/compare.py` (score vs real fingerprint) + block-bootstrap benchmark (88 PASS / 9 NEAR / 5 FAIL of 102)
- [x] 1. returns core: seasonality + HARCH cascade (1h…90d) + GJR leverage + FHS innovations
      (38 PASS / 11 NEAR / 0 FAIL of 49 return facts). Hidden regimes not needed so far; revisit if later
      layers (volume levels, funding regimes) need them
- [x] 2. intrabar OHLC (60-step t(6) bridge, intrabar vol linked to excess volume)
- [x] 3. volume + taker flow
- [x] 4. open interest
- [x] 5. funding (one-sided dead band, trend drivers, feedback on returns)
      -> structural: 96 PASS / 14 NEAR / 0 FAIL of 110 facts (bootstrap benchmark 100 / 5 / 5)
- [x] 6. scenarios on `generate()` (vol_level path, start_vol presets, vol_mult, drift, jumps, funding_shift,
      funding0) and `models.load()` with a saved state (0.2 s instead of refitting); baseline unchanged 96/14/0
- [ ] optional hidden-regime layer for the remaining NEARs (calm typical year, short funding bull episodes)
- [ ] options layer on the latent vol state (after the IV surface build)

## Later
- [ ] Joint synthetic generator (OHLCV + funding + OI + IV surface state)
- [ ] Chain reconstruction from ATM/RR/BF + forward curve, with no-arbitrage checks
- [ ] Production data source: Amberdata (constant-maturity delta surfaces)
