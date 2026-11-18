"""Fetch Binance USD-M perpetual data for BTC/USDT:USDT (symbol BTCUSDT) into data/cache/.

    python data/fetch_binance.py all                  # 1h ohlcv + funding + 1h open interest since 2022-01-01
    python data/fetch_binance.py ohlcv --interval 1m  # one dataset, any kline interval
    python data/fetch_binance.py funding
    python data/fetch_binance.py oi

Every dataset is incremental: re-running only downloads what is missing.
Output (Parquet, UTC timestamps in ms + a tz-aware `datetime` column):
    cache/binance_BTCUSDT_ohlcv_<interval>.parquet
    cache/binance_BTCUSDT_funding.parquet
    cache/binance_BTCUSDT_oi_1h.parquet
"""
import argparse
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

FAPI = "https://fapi.binance.com"
ARCHIVE = "https://data.binance.vision/data/futures/um/daily/metrics"
SYMBOL = "BTCUSDT"
DEFAULT_START = "2022-01-01"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
UA = "marketSampler-data/0.1"
# stdlib ssl uses the OS certificate store (works where an old certifi bundle does not)
SSL_CTX = ssl.create_default_context()


class Resp:
    def __init__(self, status, content, headers):
        self.status_code, self.content, self.headers = status, content, headers

    def json(self):
        return json.loads(self.content)


# ----------------------------------------------------------------- helpers
def ms(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def now_ms():
    return int(time.time() * 1000)


def get(url, params=None, retries=5):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last = None
    for k in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                return Resp(r.status, r.read(), r.headers)
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                return Resp(404, b"", e.headers)
            if e.code in (418, 429):  # rate limited: back off as Binance asks
                wait = int(e.headers.get("Retry-After", 60))
                print("  rate limited, sleeping %ss" % wait, file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2 ** k)
    raise RuntimeError("giving up on %s: %s" % (url, last))


def path(name, ext="parquet"):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, "binance_%s_%s.%s" % (SYMBOL, name, ext))


def load(name):
    p = path(name)
    if os.path.exists(p):
        return pd.read_parquet(p)
    legacy = path(name, "csv.gz")  # earlier cache format: migrated on the next save
    return pd.read_csv(legacy) if os.path.exists(legacy) else None


def save(name, df, key):
    df = df.drop(columns=["datetime"], errors="ignore")
    df = df.drop_duplicates(key).sort_values(key).reset_index(drop=True)
    df[key] = df[key].astype("int64")
    df.insert(1, "datetime", pd.to_datetime(df[key], unit="ms", utc=True))
    df.to_parquet(path(name), index=False)
    legacy = path(name, "csv.gz")
    if os.path.exists(legacy):
        os.remove(legacy)
    print("  saved %s rows -> %s (%s .. %s)" % (len(df), path(name), df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    return df


def drop_dt(df):
    return df.drop(columns=["datetime"], errors="ignore") if df is not None else None


# ----------------------------------------------------------------- OHLCV
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def fetch_ohlcv(interval="1h", start=DEFAULT_START):
    name = "ohlcv_%s" % interval
    old = load(name)
    since = ms(start) if old is None else int(old["open_time"].max()) + 1
    print("ohlcv %s from %s" % (interval, pd.to_datetime(since, unit="ms", utc=True)))
    rows, end = [], now_ms()
    while since < end:
        batch = get(FAPI + "/fapi/v1/klines",
                    dict(symbol=SYMBOL, interval=interval, startTime=since, limit=1500)).json()
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(rows) % 30000 < 1500:
            print("  ..%s rows, at %s" % (len(rows), pd.to_datetime(batch[-1][0], unit="ms", utc=True)))
        time.sleep(0.25)
    new = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["ignore"])
    # drop the still-open candle so the cache only holds closed bars
    new = new[new["close_time"] < now_ms()]
    df = pd.concat([drop_dt(old), new])
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    return save(name, df, "open_time")


# ----------------------------------------------------------------- funding
def fetch_funding(start=DEFAULT_START):
    old = load("funding")
    since = ms(start) if old is None else int(old["funding_time"].max()) + 1
    print("funding from %s" % pd.to_datetime(since, unit="ms", utc=True))
    rows = []
    while True:
        batch = get(FAPI + "/fapi/v1/fundingRate", dict(symbol=SYMBOL, startTime=since, limit=1000)).json()
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    new = pd.DataFrame(rows)
    if len(new):
        new = new.rename(columns={"fundingTime": "funding_time", "fundingRate": "funding_rate", "markPrice": "mark_price"})
        new = new[["funding_time", "funding_rate", "mark_price"]]
        new["mark_price"] = pd.to_numeric(new["mark_price"], errors="coerce")
        new["funding_rate"] = new["funding_rate"].astype(float)
    df = pd.concat([drop_dt(old), new])
    return save("funding", df, "funding_time")


# ----------------------------------------------------------------- open interest
OI_COLS = ["timestamp", "open_interest", "open_interest_value", "toptrader_ls_ratio_count",
           "toptrader_ls_ratio_sum", "ls_ratio_count", "taker_ls_vol_ratio"]


def _archive_day(day):
    """Daily 5-minute metrics file from data.binance.vision (None if not published yet)."""
    raw_dir = os.path.join(CACHE, "raw", "metrics")
    os.makedirs(raw_dir, exist_ok=True)
    zp = os.path.join(raw_dir, "%s-metrics-%s.zip" % (SYMBOL, day))
    if not os.path.exists(zp):
        r = get("%s/%s/%s-metrics-%s.zip" % (ARCHIVE, SYMBOL, SYMBOL, day))
        if r.status_code == 404:
            return None
        with open(zp, "wb") as f:
            f.write(r.content)
        time.sleep(0.1)
    with zipfile.ZipFile(zp) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df["timestamp"] = (pd.to_datetime(df["create_time"], utc=True).astype("int64") // 10 ** 6)
    df = df.rename(columns={"sum_open_interest": "open_interest", "sum_open_interest_value": "open_interest_value",
                            "count_toptrader_long_short_ratio": "toptrader_ls_ratio_count",
                            "sum_toptrader_long_short_ratio": "toptrader_ls_ratio_sum",
                            "count_long_short_ratio": "ls_ratio_count",
                            "sum_taker_long_short_vol_ratio": "taker_ls_vol_ratio"})
    # the archive has blocks of zero OI (e.g. 2022-03-07/08): bad records, drop them
    df = df[df["open_interest"] > 0]
    return df[OI_COLS]


def to_hourly(df):
    """Open interest is a level: keep the first 5-min snapshot of every hour (normally hh:00)."""
    df = df.sort_values("timestamp")
    df["timestamp"] = df["timestamp"].astype("int64") // 3600000 * 3600000
    return df.groupby("timestamp", as_index=False).first()


def fetch_oi(start=DEFAULT_START):
    old = load("oi_1h")
    first = datetime.strptime(start, "%Y-%m-%d").date()
    if old is not None:  # restart from the last complete archived day
        first = (pd.to_datetime(old["timestamp"].max(), unit="ms", utc=True) - timedelta(days=1)).date()
    today = datetime.now(timezone.utc).date()
    print("open interest from %s (daily archive)" % first)
    frames, day, missing = [], first, []
    while day < today:
        d = _archive_day(day.isoformat())
        if d is None:
            missing.append(day)
        else:
            frames.append(d)
        if day.day == 1:
            print("  ..%s" % day)
        day += timedelta(days=1)
    # recent days not yet archived: REST endpoint keeps the last 30 days
    last_archived = max(f["timestamp"].max() for f in frames) if frames else \
        (int(old["timestamp"].max()) if old is not None else ms(start))
    since = last_archived + 1
    rest = []
    if now_ms() - since < 29 * 86400 * 1000:
        # The endpoint answers with the *latest* `limit` rows unless endTime is given, and keeps returning the
        # newest row once startTime is past it: page with explicit windows of <= 500 five-minute points.
        step = 500 * 300000
        stop = now_ms()  # fixed: comparing against a moving clock never terminates
        while since < stop:
            end = min(since + step - 1, stop)
            batch = get(FAPI + "/futures/data/openInterestHist",
                        dict(symbol=SYMBOL, period="5m", startTime=since, endTime=end, limit=500)).json()
            rest.extend(b for b in batch if since <= b["timestamp"] <= end)
            since = end + 1
            time.sleep(0.25)
    if rest:
        r = pd.DataFrame(rest).rename(columns={"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"})
        r = r[["timestamp", "open_interest", "open_interest_value"]].astype(float)
        r["timestamp"] = r["timestamp"].astype("int64")
        frames.append(r)
    old_missing = [d for d in missing if d < today - timedelta(days=2)]
    if old_missing:
        print("  warning: %s archive days missing, e.g. %s" % (len(old_missing), old_missing[:5]), file=sys.stderr)
    new = to_hourly(pd.concat(frames)) if frames else None
    if old is not None and new is not None:  # re-fetched days replace what was cached
        old = old[old["timestamp"] < new["timestamp"].min()]
    df = pd.concat([drop_dt(old), new])
    return save("oi_1h", df, "timestamp")


# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "ohlcv", "funding", "oi"])
    ap.add_argument("--interval", action="append", help="kline interval(s) for ohlcv (default: 1h)")
    ap.add_argument("--start", default=DEFAULT_START, help="first day, YYYY-MM-DD (default %(default)s)")
    a = ap.parse_args()
    if a.what in ("all", "ohlcv"):
        for iv in a.interval or ["1h"]:
            fetch_ohlcv(iv, a.start)
    if a.what in ("all", "funding"):
        fetch_funding(a.start)
    if a.what in ("all", "oi"):
        fetch_oi(a.start)


if __name__ == "__main__":
    main()
