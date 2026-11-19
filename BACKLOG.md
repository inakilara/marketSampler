# Backlog

## Done
- [x] Data: Binance 1h OHLCV, funding, 1h open interest (`data/fetch_binance.py`)
- [x] `validation/`: stylised-facts research (`STYLIZED_FACTS.md`), facts library (`facts.py`), real-data
      characterisation (`characterize_real.py` -> `reports/real/`) for returns, bars, volume, OI, funding

- [x] Data: Deribit option trades since 2022 -> hourly constant-maturity IV surface (42,791 h; 30d ATM coverage 98 %,
      corr with DVOL 0.982, ATM 3.3 vol below DVOL on average) and options facts in `validation/reports/real`

- [x] Full hourly option chain (shared read-only dataset, 2019-11 → 2026-10) -> `data/build_iv_surface_chain.py`;
      the panel's IV surface now comes from it (trade-based surface only after its end)
- [x] Own Python 3.12 venv (`.venv`, pinned `requirements.txt`)

## Open checks
- [x] 2022 reconstruction bias: not a reconstruction artefact. The full chain's own mark IVs give the same 30d ATM
      (within 0.3 vol in 2022) and the same gap to DVOL (-8.2 vol in 2022, -1.7…-3.6 later): DVOL sits above ATM
      when the wings are rich
- [ ] extend Binance data back to 2019-11 to use the chain's 2020–2021 (bull/crash regimes) in the fits

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
- [x] options layer (`models/options.py`): hourly ATM / RR / BF surface at 7–180 d driven by the price path;
      `models.load("structural+options")` = joint generator; 129 PASS / 21 NEAR / 1 FAIL of 151 facts (now 125–129 PASS over seeds, chain surface + venv)
- [ ] options gaps: crash-time term-structure inversions too mild; same-hour corr(dlogvolume, dATM30) too low (0.08 vs
      0.12…0.20); BF10 downtrend not reproduced. (The old dATM–dBF FAIL was the trade reconstruction: on the full
      chain the real value is −0.05…+0.18 and the model passes)
- [ ] options layer noise term was designed for the noisy trade-based surface; re-check its size on the chain

- [x] `cards/`: pre-built fitted models (whole history + each calendar year), `models.load_card(name)`;
      funding persistence capped at 0.95 and funding→return feedback measured against the trailing 30-day level
      (one-year fits otherwise turned bull years into crashes)

## Later
- [x] Joint synthetic generator (OHLCV + funding + OI + IV surface state)
- [ ] Chain reconstruction from ATM/RR/BF + forward curve, with no-arbitrage checks
- [ ] Production data source: Amberdata (constant-maturity delta surfaces)
