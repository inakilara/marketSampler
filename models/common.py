"""Shared helpers for the generators."""
import numpy as np
import pandas as pd


def hour_of_week(index):
    return np.asarray(index.dayofweek * 24 + index.hour)


def hourly_index(start, n):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq=pd.Timedelta(hours=1), name="time")


def seasonal_profile(values, how, smooth=1):
    """Mean of `values` per hour-of-week (168 buckets) relative to the overall mean, lightly smoothed
    with a circular moving average of +-smooth hours."""
    s = pd.Series(np.asarray(values, float)).groupby(np.asarray(how)).mean().reindex(range(168))
    s = s.interpolate().bfill().ffill().values
    if smooth:
        k = 2 * smooth + 1
        s = np.convolve(np.r_[s[-smooth:], s, s[:smooth]], np.ones(k) / k, mode="valid")
    return s / s.mean()


def stationary_bootstrap_indices(n_out, n_src, mean_block, rng, align=None):
    """Indices of a stationary bootstrap (geometric block lengths). With align=(src_key, out_key) blocks
    start at a source position whose key (e.g. hour-of-week) equals the key at the output position."""
    out = np.empty(n_out, dtype=np.int64)
    p = 1.0 / mean_block
    if align is not None:
        src_key, out_key = align
        buckets = {k: np.flatnonzero(src_key == k) for k in np.unique(src_key)}
    i = 0
    while i < n_out:
        if align is None:
            j = rng.integers(0, n_src)
        else:
            b = buckets[out_key[i]]
            j = b[rng.integers(0, len(b))]
        L = int(rng.geometric(p))
        L = min(L, n_out - i, n_src - j)
        out[i:i + L] = np.arange(j, j + L)
        i += L
    return out
