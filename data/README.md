# data/

Scripts that download the historical market data used to calibrate marketSampler.
Downloads go to `data/cache/`, which is git-ignored — re-create it with the scripts.

## Binance USD-M perpetual — BTC/USDT:USDT (`BTCUSDT`)

```
python data/fetch_binance.py all                    # everything below, since 2022-01-01, hourly
python data/fetch_binance.py ohlcv --interval 1m    # other kline intervals if ever needed (default 1h)
python data/fetch_binance.py funding
python data/fetch_binance.py oi
```

All datasets are incremental: re-running only fetches what is new. Needs pandas + pyarrow (Parquet).

| file in `cache/` | source | content |
|---|---|---|
| `binance_BTCUSDT_ohlcv_1h.parquet` | `GET /fapi/v1/klines` | open/high/low/close, volume, quote volume, trades, taker-buy volumes (closed bars only) |
| `binance_BTCUSDT_funding.parquet` | `GET /fapi/v1/fundingRate` | funding rate and mark price at each 8 h settlement |
| `binance_BTCUSDT_oi_1h.parquet` | data.binance.vision daily `metrics` files (5-min) + `GET /futures/data/openInterestHist` for the days not yet archived | open interest (BTC and USDT) at the start of every hour, plus top-trader / global long-short ratios and taker long-short volume ratio (archive days only) |

`raw/metrics/` keeps the daily 5-min archive zips so OI rebuilds never re-download them.

Timestamps are UTC milliseconds (`open_time`, `funding_time`, `timestamp`) plus a tz-aware `datetime` column.

## Deribit BTC options (inverse, BTC-settled)

```
python data/fetch_deribit.py all      # DVOL + all option trades since 2022 + hourly chain (takes a few hours the first time)
python data/fetch_deribit.py trades   # raw trades only (resumable, one file per UTC day)
python data/fetch_deribit.py chain    # rebuild the hourly chain from cached trades
python data/fetch_deribit.py dvol
```

Deribit publishes no historical snapshots of the full chain, so the hourly chain is **rebuilt from trades**
(history.deribit.com keeps every trade with the instrument's mark price, the trade IV and the index price).

| file in `cache/` | content |
|---|---|
| `raw/deribit_trades/BTC-option-trades-YYYY-MM-DD.parquet` | every BTC option trade of that UTC day |
| `deribit_BTC_option_chain_1h.parquet` | one row per (hour, instrument that traded in that hour): expiry, strike, C/P, time to expiry, moneyness, index, last mark price, last trade price/IV, VWAP, volume, number of trades |
| `deribit_BTC_dvol_1h.parquet` | DVOL (30-day implied volatility index) hourly OHLC |

Limits: the chain is **sparse** — only strikes that traded in a given hour appear, and `last_iv` is the IV of the
last trade (not the mark IV). Option prices are in BTC (multiply by `index_price` for USD).

## Full hourly option chain (shared, read-only)

`C:\ClaudeSandbox\shared-data\option-chain\option_chain_hourly.parquet` holds every listed BTC option every hour
(2019-11-25 → 2026-10-25, ~39M rows) with Deribit's mark IV and the expiry's forward. It is shared by all
projects: **read it in place, never modify it or write into that folder.**

```
python data/build_iv_surface_chain.py --check    # -> cache/deribit_BTC_iv_surface_chain_1h.parquet (~7 min)
```

Same columns as `deribit_BTC_iv_surface_1h.parquet` (atm / rr25 / bf25 / rr10 / bf10 at 7–180 d), built from all
OTM strikes' mark IVs instead of recent trades. `validation/panel.py` uses it and falls back to the trade-based
surface only for hours it does not cover (after 2026-10-25). On the 42,199 common hours the two agree within ~0.2
vol points ATM (corr 0.999) and ~0.1 RR25; BF agrees less closely (corr ~0.82), as expected from sparse trades.

If pip/requests fail with `CERTIFICATE_VERIFY_FAILED` on this machine, the scripts use the stdlib `ssl`
(OS certificate store); for pip, export the Windows root store to a PEM file and pass `--cert <file>`.
Note: the REST open-interest endpoint only serves the last 30 days, which is why history comes from the archive.
