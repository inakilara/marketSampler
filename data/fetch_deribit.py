"""Fetch Deribit BTC (inverse, BTC-settled) option data into data/cache/.

Deribit does not publish historical snapshots of the full option chain, so the hourly chain is
rebuilt from the complete public trade history (history.deribit.com): each trade carries the
instrument's mark price, the trade's implied volatility and the index price at that moment.

    python data/fetch_deribit.py all            # trades -> hourly chain, plus DVOL, since 2022-01-01
    python data/fetch_deribit.py trades         # raw trades, one Parquet file per UTC day (resumable)
    python data/fetch_deribit.py chain          # (re)build the hourly chain from the raw trades
    python data/fetch_deribit.py dvol           # DVOL volatility index, hourly OHLC

Output:
    cache/raw/deribit_trades/BTC-option-trades-YYYY-MM-DD.parquet
    cache/deribit_BTC_option_chain_1h.parquet   one row per (hour, instrument traded in that hour)
    cache/deribit_BTC_dvol_1h.parquet
Prices of inverse options are in BTC; multiply by index_price for USD.
"""
import argparse
import glob
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

HISTORY = "https://history.deribit.com/api/v2/public"
LIVE = "https://www.deribit.com/api/v2/public"
CURRENCY = "BTC"
DEFAULT_START = "2022-01-01"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RAW = os.path.join(CACHE, "raw", "deribit_trades")
UA = "marketSampler-data/0.1"
SSL_CTX = ssl.create_default_context()  # OS certificate store
DAY_MS = 86400000


def ms(d):
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def api(base, method, retries=6, **params):
    url = "%s/%s?%s" % (base, method, urllib.parse.urlencode(params))
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                return json.loads(r.read())["result"]
        except urllib.error.HTTPError as e:
            last = e
            body = e.read()[:300]
            if e.code == 429 or b"too_many_requests" in body:
                time.sleep(2 + 2 ** k)
                continue
            if e.code < 500:
                raise RuntimeError("%s -> %s %s" % (url, e.code, body))
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
        time.sleep(2 ** k)
    raise RuntimeError("giving up on %s: %s" % (url, last))


# ----------------------------------------------------------------- raw trades
TRADE_COLS = ["timestamp", "trade_id", "instrument_name", "direction", "price", "amount", "iv",
              "mark_price", "index_price", "tick_direction", "trade_seq", "block_trade_id", "liquidation"]


def fetch_day(d):
    """All BTC option trades of UTC day d (paginated, de-duplicated)."""
    start, end = ms(d), ms(d) + DAY_MS - 1
    rows, cursor = {}, start
    while True:
        res = api(HISTORY, "get_last_trades_by_currency_and_time", currency=CURRENCY, kind="option",
                  start_timestamp=cursor, end_timestamp=end, count=1000, sorting="asc")
        trades = res["trades"]
        for t in trades:
            rows[t["trade_id"]] = t
        if not res.get("has_more") or not trades:
            break
        nxt = trades[-1]["timestamp"]
        # several trades can share the last millisecond: re-request from it, duplicates are dropped by id
        cursor = nxt if nxt > cursor else nxt + 1
        time.sleep(0.05)
    df = pd.DataFrame(list(rows.values()))
    for c in TRADE_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[TRADE_COLS]
    df["block_trade_id"] = df["block_trade_id"].astype(str).where(df["block_trade_id"].notna(), None)
    df["liquidation"] = df["liquidation"].astype(str).where(df["liquidation"].notna(), None)
    return df.sort_values(["timestamp", "trade_id"]).reset_index(drop=True)


def fetch_trades(start=DEFAULT_START):
    os.makedirs(RAW, exist_ok=True)
    d = datetime.strptime(start, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    n_new = 0
    while d <= today:
        p = os.path.join(RAW, "BTC-option-trades-%s.parquet" % d.isoformat())
        # completed days are immutable; today (partial) is always refreshed
        if not os.path.exists(p) or d >= today - timedelta(days=1):
            df = fetch_day(d)
            df.to_parquet(p, index=False)
            n_new += 1
            if d.day == 1 or n_new % 50 == 0:
                print("  %s: %s trades" % (d, len(df)), flush=True)
        d += timedelta(days=1)
    print("trades: %s day files written in %s" % (n_new, RAW))


# ----------------------------------------------------------------- hourly chain
def parse_instruments(names):
    parts = names.str.split("-", expand=True)
    expiry = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)  # Deribit expiry 08:00 UTC
    strike = parts[2].str.replace("d", ".").astype(float)
    return expiry, strike, parts[3]


def build_chain():
    files = sorted(glob.glob(os.path.join(RAW, "BTC-option-trades-*.parquet")))
    if not files:
        sys.exit("no raw trades; run: python data/fetch_deribit.py trades")
    out = []
    for f in files:
        t = pd.read_parquet(f)
        if t.empty:
            continue
        t["hour"] = t["timestamp"] // 3600000 * 3600000
        t["notional_btc"] = t["amount"]
        t["premium_btc"] = t["price"] * t["amount"]
        g = t.sort_values("timestamp").groupby(["hour", "instrument_name"])
        h = g.agg(last_ts=("timestamp", "last"), mark_price=("mark_price", "last"), last_price=("price", "last"),
                  last_iv=("iv", "last"), index_price=("index_price", "last"),
                  vwap=("premium_btc", "sum"), volume=("amount", "sum"), trades=("trade_id", "count")).reset_index()
        h["vwap"] = h["vwap"] / h["volume"]
        out.append(h)
    c = pd.concat(out, ignore_index=True)
    c["expiry"], c["strike"], c["option_type"] = parse_instruments(c["instrument_name"])
    c["datetime"] = pd.to_datetime(c["hour"], unit="ms", utc=True)
    c["tau_years"] = (c["expiry"] - c["datetime"]).dt.total_seconds() / (365.25 * 86400)
    c["moneyness"] = c["strike"] / c["index_price"]
    c = c[["hour", "datetime", "instrument_name", "expiry", "strike", "option_type", "tau_years", "moneyness",
           "index_price", "mark_price", "last_price", "vwap", "last_iv", "volume", "trades", "last_ts"]]
    p = os.path.join(CACHE, "deribit_BTC_option_chain_1h.parquet")
    c.to_parquet(p, index=False)
    print("chain: %s rows, %s hours, %s instruments -> %s" % (len(c), c["hour"].nunique(), c["instrument_name"].nunique(), p))


# ----------------------------------------------------------------- DVOL
def fetch_dvol(start=DEFAULT_START):
    p = os.path.join(CACHE, "deribit_BTC_dvol_1h.parquet")
    old = pd.read_parquet(p) if os.path.exists(p) else None
    since = ms(datetime.strptime(start, "%Y-%m-%d").date()) if old is None else int(old["timestamp"].max()) + 1
    rows, end = [], int(time.time() * 1000)
    while since < end:
        # the endpoint returns the most recent points first and a continuation for older ones;
        # walking forward in 30-day windows keeps it simple
        stop = min(end, since + 30 * DAY_MS)
        res = api(LIVE, "get_volatility_index_data", currency=CURRENCY, start_timestamp=since,
                  end_timestamp=stop, resolution="3600")
        rows.extend(res["data"])
        cont = res.get("continuation")
        while cont and cont > since:
            r2 = api(LIVE, "get_volatility_index_data", currency=CURRENCY, start_timestamp=since,
                     end_timestamp=cont, resolution="3600")
            rows.extend(r2["data"])
            cont = r2.get("continuation")
        since = stop + 1
        time.sleep(0.1)
    new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df = pd.concat([old.drop(columns=["datetime"]) if old is not None else None, new])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("int64")
    df.insert(1, "datetime", pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    df.to_parquet(p, index=False)
    print("dvol: %s rows (%s .. %s) -> %s" % (len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1], p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "trades", "chain", "dvol"])
    ap.add_argument("--start", default=DEFAULT_START)
    a = ap.parse_args()
    if a.what in ("all", "dvol"):
        fetch_dvol(a.start)
    if a.what in ("all", "trades"):
        fetch_trades(a.start)
    if a.what in ("all", "trades", "chain"):
        build_chain()


if __name__ == "__main__":
    main()
