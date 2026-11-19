"""Benchmark: stationary block bootstrap of real hours (approach B in models/README.md).

Joint hourly records are resampled in blocks (geometric lengths, mean 7 days). Each block starts at a real
hour with the same hour-of-week as the output position, so the intraday/weekly profile survives. Prices,
OI are rebuilt from resampled log changes; funding is copied as a level, so it jumps at block joins.
It reproduces the real data by construction (except across joins) and is the bar to beat; it cannot
create unseen scenarios.
"""
import numpy as np
import pandas as pd

from models.common import hour_of_week, hourly_index, stationary_bootstrap_indices


class BlockBootstrap:
    name = "bootstrap"

    def __init__(self, mean_block_hours=168):
        self.mean_block = mean_block_hours

    def fit(self, p):
        prev_close = p["close"].shift(1)
        rec = pd.DataFrame({
            "r_gap": np.log(p["open"] / prev_close),          # open vs previous close (≈0 for a perp)
            "r_oc": np.log(p["close"] / p["open"]),
            "u": np.log(p["high"] / p["open"]),
            "d": np.log(p["low"] / p["open"]),
            "lv": np.log(p["volume"]),
            "tb": p["taker_buy_volume"] / p["volume"],
            "doi": np.log(p["oi"]).shift(-1) - np.log(p["oi"]),
            "funding": p["funding"],
        }, index=p.index).iloc[1:-1]
        rec = rec.dropna(subset=["r_oc", "u", "d", "lv"])
        rec["doi"] = rec["doi"].fillna(0.0)
        rec["funding"] = rec["funding"].ffill().bfill()
        rec["tb"] = rec["tb"].fillna(0.5)
        rec["r_gap"] = rec["r_gap"].fillna(0.0)
        self.rec = rec
        self.how = hour_of_week(rec.index)
        self.p0 = float(p["close"].iloc[-1])
        self.oi0 = float(p["oi"].dropna().iloc[-1])
        return self

    def generate(self, n_hours, seed=0, start="2030-01-07"):
        rng = np.random.default_rng(seed)
        idx = hourly_index(start, n_hours)
        j = stationary_bootstrap_indices(n_hours, len(self.rec), self.mean_block, rng,
                                         align=(self.how, hour_of_week(idx)))
        R = self.rec.iloc[j]
        lo = np.log(self.p0) + np.cumsum(R["r_gap"].values + R["r_oc"].values) - R["r_oc"].values  # log open
        o = np.exp(lo)
        out = pd.DataFrame(index=idx)
        out["open"] = o
        out["close"] = o * np.exp(R["r_oc"].values)
        out["high"] = o * np.exp(R["u"].values)
        out["low"] = o * np.exp(R["d"].values)
        out["volume"] = np.exp(R["lv"].values)
        out["taker_buy_volume"] = out["volume"] * R["tb"].values
        loi = np.log(self.oi0) + np.r_[0.0, np.cumsum(R["doi"].values)[:-1]]
        out["oi"] = np.exp(loi)
        out["funding"] = R["funding"].values
        return out
