# Backlog

## Done
- [x] Data: Binance 1h OHLCV, funding, 1h open interest (`data/fetch_binance.py`)
- [x] `validation/`: stylised-facts research (`STYLIZED_FACTS.md`), facts library (`facts.py`), real-data
      characterisation (`characterize_real.py` -> `reports/real/`) for returns, bars, volume, OI, funding

## In progress
- [ ] Data: Deribit option trades since 2022 (downloading) -> hourly constant-maturity IV surface:
      ATM / RR25 / BF25 / RR10 / BF10 at 7-180 d (`data/fetch_deribit.py`, `data/smile.py`, `data/build_iv_surface.py`)
  - [ ] then re-run `validation/characterize_real.py` to fill the options section
  - [ ] Check 2022 reconstruction bias: 30d ATM sits ~5.6 vol below DVOL in Jan-2022 vs ~2.2 today
        (validate on a free Tardis first-of-month full chain)

## Next
- [ ] `validation/compare.py`: score a synthetic panel against the real fingerprint (inter-year ranges,
      curve distances, discriminative score, TSTR)

## Later
- [ ] Joint synthetic generator (OHLCV + funding + OI + IV surface state)
- [ ] Chain reconstruction from ATM/RR/BF + forward curve, with no-arbitrage checks
- [ ] Production data source: Amberdata (constant-maturity delta surfaces)
