"""Build the hourly panel that all stylised facts are computed on.

A *panel* is a DataFrame indexed by bar open time (UTC, hourly) with these columns (any subset may be
missing; facts that need a missing column are skipped):

    open, high, low, close      perp prices of the bar [t, t+1h)
    volume, taker_buy_volume    base-asset volume of the bar
    oi                          open interest (BTC) at the bar open
    funding                     last settled funding rate at the bar open (8h rate, decimal)
    atm_<T>d, rr25_<T>d, bf25_<T>d, rr10_<T>d, bf10_<T>d     IV surface snapshot at the bar open
    dvol                        DVOL at the bar open

Synthetic generators should emit the same columns so facts.py can be applied unchanged.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")


def _read(name):
    p = os.path.join(CACHE, name)
    return pd.read_parquet(p) if os.path.exists(p) else None


def load_real():
    k = _read("binance_BTCUSDT_ohlcv_1h.parquet")
    if k is None:
        raise SystemExit("missing OHLCV cache: run python data/fetch_binance.py all")
    p = k.set_index(pd.to_datetime(k["open_time"], unit="ms", utc=True))[
        ["open", "high", "low", "close", "volume", "taker_buy_volume"]]
    p.index.name = "time"

    oi = _read("binance_BTCUSDT_oi_1h.parquet")
    if oi is not None:
        s = oi.set_index(pd.to_datetime(oi["timestamp"], unit="ms", utc=True))["open_interest"]
        p["oi"] = s.where(s > 0).reindex(p.index)

    f = _read("binance_BTCUSDT_funding.parquet")
    if f is not None:
        # settlement times carry a few ms of jitter: floor to the hour; the rate settled at t is known from t on
        s = f.set_index(pd.to_datetime(f["funding_time"], unit="ms", utc=True).dt.floor("H"))["funding_rate"]
        s = s[~s.index.duplicated(keep="last")]
        p["funding"] = s.reindex(p.index.union(s.index)).ffill().reindex(p.index)

    iv = _read("deribit_BTC_iv_surface_1h.parquet")
    if iv is not None:
        cols = [c for c in iv.columns if c.split("_")[0] in ("atm", "rr25", "bf25", "rr10", "bf10")]
        s = iv.set_index(pd.to_datetime(iv["timestamp"], unit="ms", utc=True))[cols]
        p = p.join(s, how="left")

    dv = _read("deribit_BTC_dvol_1h.parquet")
    if dv is not None:
        p["dvol"] = dv.set_index(pd.to_datetime(dv["timestamp"], unit="ms", utc=True))["open"].reindex(p.index) / 100
    return p


def returns(p):
    """Close-to-close hourly log returns (NaN for the first bar)."""
    return np.log(p["close"]).diff()
