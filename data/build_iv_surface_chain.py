"""Hourly constant-maturity BTC implied-vol surface (ATM, RR, BF at 25 and 10 delta) from the full hourly option
chain in the shared, read-only dataset C:\\ClaudeSandbox\\shared-data\\option-chain (never write there).

    python data/build_iv_surface_chain.py            # -> cache/deribit_BTC_iv_surface_chain_1h.parquet
    python data/build_iv_surface_chain.py --check    # also coverage and comparison with the trade-based surface

Unlike data/build_iv_surface.py (marks of the options that happened to trade in the last 2-12 h, IV inverted
from price with a parity-estimated forward), every listed strike is quoted every hour with Deribit's own mark IV
and the expiry's underlying (forward) price. Per hour and expiry: OTM options (calls K >= F, puts K < F) with
0 < markIv < 500, k = ln(K/F), then the same smile.smile_points (call-delta interpolation) and
smile.constant_maturity (total-variance interpolation) as the trade-based surface, so columns are identical.
Expiries closer than 2 days are skipped. Missing points stay NaN.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smile  # noqa: E402

CHAIN = r"C:\ClaudeSandbox\shared-data\option-chain\option_chain_hourly.parquet"
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "deribit_BTC_iv_surface_chain_1h.parquet")
TRADES_SURFACE = os.path.join(CACHE, "deribit_BTC_iv_surface_1h.parquet")
TENORS = (7, 14, 30, 60, 90, 180)
MIN_TAU_DAYS = 2
IV_CAP = 500.0
YEAR_S = 365.25 * 86400
COLS = ["timestamp", "expirationTimestamp", "strike", "putCall", "markIv", "underlyingPrice", "indexPrice"]


def prepare(tbl):
    d = pd.DataFrame({
        "t": tbl.column("timestamp").to_numpy().astype("datetime64[s]").astype("int64"),
        "exp": tbl.column("expirationTimestamp").to_numpy().astype("datetime64[s]").astype("int64"),
        "K": tbl.column("strike").to_numpy().astype(float),
        "call": tbl.column("putCall").to_numpy(zero_copy_only=False) == "C",
        "iv": tbl.column("markIv").to_numpy() / 100.0,
        "F": tbl.column("underlyingPrice").to_numpy(),
        "idx": tbl.column("indexPrice").to_numpy(),
    })
    d["tau"] = (d["exp"] - d["t"]) / YEAR_S
    ok = (d["iv"] > 0) & (d["iv"] < IV_CAP / 100 * 0.999) & (d["F"] > 0) & (d["tau"] >= MIN_TAU_DAYS / 365.25)
    ok &= np.where(d["call"], d["K"] >= d["F"], d["K"] < d["F"])
    d = d[ok]
    d["k"] = np.log(d["K"] / d["F"])
    return d


def surface(d):
    rows = []
    for t, g in d.groupby("t", sort=True):
        per_exp, n_pts = [], 0
        for exp, e in g.groupby("exp"):
            if len(e) < 2:
                continue
            tau = float(e["tau"].iloc[0])
            pt = smile.smile_points(e["k"].values, e["iv"].values, tau)
            pt["tau"] = tau
            per_exp.append(pt)
            n_pts += len(e)
        if not per_exp:
            continue
        res = smile.constant_maturity(per_exp, TENORS)
        res.update(timestamp=int(t) * 1000, index=float(g["idx"].iloc[0]), n_expiries=len(per_exp), n_points=n_pts)
        rows.append(res)
    return rows


def build(out=OUT):
    f = pq.ParquetFile(CHAIN)
    rows, t0, carry = [], time.time(), None
    for i in range(f.metadata.num_row_groups):
        d = prepare(f.read_row_group(i, columns=COLS))
        if carry is not None:
            d = pd.concat([carry, d], ignore_index=True)
        # an hour may straddle two row groups: hold back the last hour until the next group is read
        last = d["t"].max()
        carry = d[d["t"] == last] if i < f.metadata.num_row_groups - 1 else None
        rows += surface(d[d["t"] < last] if carry is not None else d)
        print("  row group %d/%d (%d snapshots, %.0fs)" % (i + 1, f.metadata.num_row_groups, len(rows),
                                                          time.time() - t0), flush=True)
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    first = ["timestamp", "index", "n_expiries", "n_points"]
    main = ["%s_%dd" % (fl, T) for T in TENORS for fl in ("atm", "rr25", "bf25", "rr10", "bf10")]
    df = df[first + main + [c for c in df.columns if c not in first + main]]
    df.insert(1, "datetime", pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    df.to_parquet(out, index=False)
    print("chain iv surface: %d hourly rows (%s .. %s) -> %s" % (len(df), df["datetime"].iloc[0],
                                                                  df["datetime"].iloc[-1], out))
    return df


def check(df):
    print("\ncoverage %% by year (atm / rr25 / bf10 at 30d, atm 180d):")
    y = df["datetime"].dt.year
    print((100 * df.groupby(y)[["atm_30d", "rr25_30d", "bf10_30d", "atm_180d"]]
           .apply(lambda x: x.notna().mean())).round(1).to_string())
    if os.path.exists(TRADES_SURFACE):
        tr = pd.read_parquet(TRADES_SURFACE)
        m = df.merge(tr, on="timestamp", suffixes=("_c", "_t"))
        print("\nchain vs trade-based surface (vol points, chain - trades), %d common hours:" % len(m))
        out = {}
        for T in (7, 30, 90, 180):
            for fl in ("atm", "rr25", "bf25", "rr10", "bf10"):
                a, b = 100 * m["%s_%dd_c" % (fl, T)], 100 * m["%s_%dd_t" % (fl, T)]
                ok = a.notna() & b.notna()
                out["%s_%dd" % (fl, T)] = {"corr": np.corrcoef(a[ok], b[ok])[0, 1], "mean_diff": (a - b)[ok].mean(),
                                           "mad": (a - b)[ok].abs().median()}
        print(pd.DataFrame(out).T.round(3).to_string())
        yy = m["datetime_c"].dt.year
        print("\nmean 30d ATM diff by year:", (100 * (m["atm_30d_c"] - m["atm_30d_t"])).groupby(yy).mean().round(2).to_dict())
    dv = os.path.join(CACHE, "deribit_BTC_dvol_1h.parquet")
    if os.path.exists(dv):
        dv = pd.read_parquet(dv)[["timestamp", "open"]]
        m = df.merge(dv, on="timestamp").dropna(subset=["atm_30d"])
        d = 100 * m["atm_30d"] - m["open"]
        print("\n30d ATM vs DVOL: corr %.3f, mean diff %.2f; by year %s" % (
            np.corrcoef(100 * m["atm_30d"], m["open"])[0, 1], d.mean(),
            d.groupby(pd.to_datetime(m["timestamp"], unit="ms").dt.year).mean().round(2).to_dict()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = build()
    if a.check:
        check(res)
