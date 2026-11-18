"""Hourly constant-maturity BTC implied-vol surface (ATM, RR, BF at 25 and 10 delta) rebuilt from the
cached Deribit option trades (run data/fetch_deribit.py trades first). Method: see data/smile.py.

    python data/build_iv_surface.py            # all cached days -> cache/deribit_BTC_iv_surface_1h.parquet
    python data/build_iv_surface.py --check    # also compare 30d ATM with DVOL and print coverage

Snapshot at each hour boundary t uses, per expiry, the last trade of every instrument in [t - w, t):
w = 2 h for expiries <= 35 d, 6 h for <= 100 d, 12 h beyond (long-dated options trade rarely and their vol
moves slowly). Expiries closer than 2 days are skipped. Missing points stay NaN (no extrapolation).
Columns: timestamp/datetime (snapshot time t), index, and for every tenor T in TENORS:
atm_T, rr25_T, bf25_T, rr10_T, bf10_T (vol as decimals, 0.45 = 45 %), plus c25/p25/c10/p10 and diagnostics.
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smile  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RAW = os.path.join(CACHE, "raw", "deribit_trades")
OUT = os.path.join(CACHE, "deribit_BTC_iv_surface_1h.parquet")
TENORS = (7, 14, 30, 60, 90, 180)
H = 3600000
YEAR_MS = 365.25 * 86400000
MIN_TAU_DAYS = 2


def window_ms(tau_days):
    return 2 * H if tau_days <= 35 else (6 * H if tau_days <= 100 else 12 * H)


def load_trades(files):
    cols = ["timestamp", "instrument_name", "mark_price", "index_price"]
    t = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    t = t.dropna(subset=["mark_price", "index_price"]).sort_values("timestamp").reset_index(drop=True)
    names = pd.Series(t["instrument_name"].unique())
    parts = names.str.split("-", expand=True)
    meta = pd.DataFrame({
        "instrument_name": names,
        "expiry_ms": ((pd.to_datetime(parts[1], format="%d%b%y") + pd.Timedelta(hours=8)).astype("int64") // 10 ** 6),
        "strike": parts[2].str.replace("d", ".").astype(float),
        "is_call": parts[3] == "C",
    })
    return t.merge(meta, on="instrument_name", how="left")


def snapshot(t_ms, win, basis_mem):
    """win: trades in [t - 12h, t). Returns dict of surface values at time t."""
    if win.empty:
        return None
    index_now = float(win["index_price"].iloc[-1])
    win = win[win["expiry_ms"] - t_ms >= MIN_TAU_DAYS * 86400000]
    per_exp, n_pts, missing_basis = [], 0, 0
    pending = []
    for exp, g in win.groupby("expiry_ms"):
        tau = (exp - t_ms) / YEAR_MS
        g = g[g["timestamp"] >= t_ms - window_ms(tau * 365.25)]
        if len(g) < 2:
            continue
        last = g.groupby("instrument_name").last()
        # basis ratio F/index from parity on this window (C and P at the same strike)
        c = last[last["is_call"]].set_index("strike")
        p = last[~last["is_call"]].set_index("strike")
        both = c.index.intersection(p.index)
        b = smile.basis_ratio([(K, c.at[K, "mark_price"], p.at[K, "mark_price"], c.at[K, "index_price"],
                                p.at[K, "index_price"]) for K in both]) if len(both) else float("nan")
        if np.isfinite(b):
            basis_mem[exp] = (t_ms, b)
        elif exp in basis_mem and t_ms - basis_mem[exp][0] <= 72 * H:
            b = basis_mem[exp][1]  # carry the last parity estimate forward (basis drifts slowly)
        pending.append((exp, tau, last, b))
    # expiries without any parity estimate: borrow the annualised basis of the others
    ann = [math.log(b) / tau for _, tau, _, b in pending if np.isfinite(b) and b > 0]
    ann_med = float(np.median(ann)) if ann else 0.0
    for exp, tau, last, b in pending:
        if not np.isfinite(b):
            b = math.exp(ann_med * tau)
            missing_basis += 1
        Fi = b * last["index_price"].values
        K = last["strike"].values
        isc = last["is_call"].values
        otm = np.where(isc, K >= Fi, K < Fi)
        if otm.sum() < 2:
            continue
        iv = smile.implied_vol_vec(last["mark_price"].values[otm], Fi[otm], K[otm], np.full(otm.sum(), tau), isc[otm])
        k = np.log(K[otm] / (b * index_now))
        pt = smile.smile_points(k, iv, tau)
        pt["tau"] = tau
        per_exp.append(pt)
        n_pts += int(np.isfinite(iv).sum())
    if not per_exp:
        return None
    res = smile.constant_maturity(per_exp, TENORS)
    res.update(index=index_now, n_expiries=len(per_exp), n_points=n_pts, n_basis_fallback=missing_basis)
    return res


def build(month_prefix=None, out=OUT):
    files = sorted(glob.glob(os.path.join(RAW, "BTC-option-trades-%s*.parquet" % (month_prefix or ""))))
    if not files:
        sys.exit("no raw trades; run: python data/fetch_deribit.py trades")
    # process month by month with a 12 h overlap so every window is complete
    by_month = {}
    for f in files:
        by_month.setdefault(os.path.basename(f)[18:25], []).append(f)
    months = sorted(by_month)
    rows, basis_mem, prev_tail = [], {}, None
    t0 = time.time()
    for m in months:
        tr = load_trades(by_month[m])
        if prev_tail is not None:
            tr = pd.concat([prev_tail, tr], ignore_index=True)
        first_day = pd.Timestamp(m + "-01", tz="UTC").value // 10 ** 6
        last_ts = int(tr["timestamp"].max())
        ts = tr["timestamp"].values
        t = max(first_day, (int(ts.min()) // H + 1) * H)
        while t <= last_ts // H * H + H:
            lo, hi = np.searchsorted(ts, t - 12 * H), np.searchsorted(ts, t)
            s = snapshot(t, tr.iloc[lo:hi], basis_mem)
            if s is not None:
                s["timestamp"] = t
                rows.append(s)
            t += H
        prev_tail = tr[tr["timestamp"] >= last_ts - 12 * H]
        print("  %s done (%d snapshots so far, %.0fs)" % (m, len(rows), time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    # the last snapshot may be built from a partial hour of the still-open day: keep only complete hours
    df = df[df["timestamp"] <= int(time.time() * 1000) // H * H]
    first = ["timestamp", "index", "n_expiries", "n_points", "n_basis_fallback"]
    main = ["%s_%dd" % (f, T) for T in TENORS for f in ("atm", "rr25", "bf25", "rr10", "bf10")]
    rest = [c for c in df.columns if c not in first + main]
    df = df[first + main + rest]
    df.insert(1, "datetime", pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    df.to_parquet(out, index=False)
    print("iv surface: %d hourly rows (%s .. %s) -> %s" % (len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1], out))
    return df


def check(df):
    print("\ncoverage (share of hours with a value):")
    cov = pd.DataFrame({T: [df["%s_%dd" % (f, T)].notna().mean() for f in ("atm", "rr25", "bf25", "rr10", "bf10")]
                        for T in TENORS}, index=["atm", "rr25", "bf25", "rr10", "bf10"])
    print((100 * cov).round(1).to_string())
    print("\nby year, atm_30d / rr25_30d coverage %:")
    y = df["datetime"].dt.year
    print((100 * df.groupby(y)[["atm_30d", "rr25_30d", "atm_180d"]].apply(lambda x: x.notna().mean())).round(1).to_string())
    p = os.path.join(CACHE, "deribit_BTC_dvol_1h.parquet")
    if os.path.exists(p):
        dv = pd.read_parquet(p)[["timestamp", "open"]].rename(columns={"open": "dvol"})  # DVOL at hour start
        m = df.merge(dv, on="timestamp", how="inner").dropna(subset=["atm_30d"])
        if len(m):
            d = 100 * m["atm_30d"] - m["dvol"]
            print("\n30d ATM vs DVOL (vol points): corr %.3f, mean diff %.2f, median abs diff %.2f, n=%d"
                  % (np.corrcoef(100 * m["atm_30d"], m["dvol"])[0, 1], d.mean(), d.abs().median(), len(m)))
            print("(DVOL is a 30d variance-swap-like index, so it sits slightly above ATM when the smile has wings)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", help="restrict to raw files starting with this date prefix, e.g. 2022-01 (test runs; "
                                   "written to a separate *_test.parquet)")
    a = ap.parse_args()
    out = build(a.only, OUT.replace(".parquet", "_test.parquet") if a.only else OUT)
    if a.check:
        check(out)
