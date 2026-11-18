"""Smile reconstruction for Deribit inverse BTC options from trade-time mark prices:
mark price -> mark IV -> ATM / RR / BF by delta per expiry -> constant maturities.

Conventions
  * Black-76 on the expiry forward F, r = 0. Deribit price (BTC) = Black76_USD(F, K, sigma, tau) / F.
  * Each trade's mark is inverted with the forward *at that trade's time*: F_i = basis_ratio * index_i,
    where basis_ratio = F / index of the expiry, estimated from put-call parity (inverse: C - P = 1 - K/F).
    (Using the current forward for older marks biases the risk reversal toward zero.)
  * Deltas are forward deltas, not premium-adjusted: call N(d1), put N(d1) - 1. Only OTM options are used.
  * Smile points are linearly interpolated in call-delta space (no parametric fit, no extrapolation):
    ATM = call delta 0.5 (delta-neutral straddle), x-delta put = call delta 1 - x.
  * RR_x = IV(x call) - IV(x put);  BF_x = (IV(x call) + IV(x put)) / 2 - ATM.
  * Constant maturities: linear interpolation of total variance sigma^2 * tau between listed expiries.

Validated against the full live chain (mark_iv of every strike): with a 2-hour window of trades the
7-60d ATM / RR25 / BF25 / RR10 / BF10 match within ~0.3-1 vol point; long tenors need longer windows.
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

TARGET_DELTAS = (0.25, 0.10)
FIELDS = ("atm", "c25", "p25", "c10", "p10")


def black76_btc(F, K, sigma, tau, is_call):
    if sigma <= 0 or tau <= 0:
        intrinsic = max(F - K, 0) if is_call else max(K - F, 0)
        return intrinsic / F
    s = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    usd = F * norm.cdf(d1) - K * norm.cdf(d2) if is_call else K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return usd / F


def implied_vol(price_btc, F, K, tau, is_call):
    intrinsic = (max(F - K, 0) if is_call else max(K - F, 0)) / F
    if not (price_btc > intrinsic + 1e-9) or tau <= 0:
        return float("nan")
    try:
        return brentq(lambda s: black76_btc(F, K, s, tau, is_call) - price_btc, 1e-4, 10.0, xtol=1e-7)
    except ValueError:
        return float("nan")


def implied_vol_vec(price_btc, F, K, tau, is_call, lo=1e-3, hi=8.0, iters=50):
    """Vectorised bisection version of implied_vol (arrays of equal length). NaN where no solution."""
    from scipy.special import ndtr
    price_btc, F, K, tau = (np.asarray(x, float) for x in (price_btc, F, K, tau))
    is_call = np.asarray(is_call, bool)
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0)) / F
    a, b = np.full(price_btc.shape, lo), np.full(price_btc.shape, hi)
    sq = np.sqrt(np.maximum(tau, 1e-12))
    lnFK = np.log(F / K)

    def price(sig):
        s = sig * sq
        d1 = (lnFK + 0.5 * s * s) / s
        d2 = d1 - s
        c = F * ndtr(d1) - K * ndtr(d2)
        return np.where(is_call, c, c - F + K) / F  # put via parity (r = 0)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = price(m) < price_btc
        a, b = np.where(up, m, a), np.where(up, b, m)
    iv = 0.5 * (a + b)
    bad = ~(price_btc > intrinsic + 1e-9) | (tau <= 0) | (iv >= hi * 0.999) | (iv <= lo * 1.001)
    return np.where(bad, np.nan, iv)


def basis_ratio(pairs, tol=0.02):
    """pairs: (strike, call_mark_btc, put_mark_btc, index_at_call_trade, index_at_put_trade).
    Returns median F/index over pairs (outliers beyond tol dropped), or nan without pairs."""
    r = []
    for K, c, p, ic, ip in pairs:
        d = 1.0 - (c - p)
        if d > 0:
            r.append((K / d) / ((ic + ip) / 2))
    if not r:
        return float("nan")
    r = np.array(r)
    med = np.median(r)
    r = r[np.abs(r / med - 1) < tol]
    return float(np.median(r)) if len(r) else float(med)


def smile_points(k, iv, tau, max_gap=0.15):
    """k = ln(K/F), iv = OTM mark IVs of one expiry. Each option gets its own call delta from its own IV;
    ATM and the delta points are linearly interpolated in call-delta space. NaN where the observations
    do not bracket the point with neighbours closer than 2*max_gap in delta."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    ok = np.isfinite(k) & np.isfinite(iv)
    ok[ok] &= iv[ok] > 0
    k, iv = k[ok], iv[ok]
    out = {f: float("nan") for f in FIELDS + ("rr25", "bf25", "rr10", "bf10")}
    if len(k) < 2:
        return out
    s = iv * math.sqrt(tau)
    cd = norm.cdf((-k + 0.5 * s * s) / s)
    o = np.argsort(cd)
    cd, iv = cd[o], iv[o]

    def at(x):
        j = np.searchsorted(cd, x)
        if j == 0 or j == len(cd) or cd[j] - cd[j - 1] > max_gap * 2:
            return float("nan")
        return float(np.interp(x, cd, iv))
    out["atm"] = at(0.5)
    for d in TARGET_DELTAS:
        n = round(d * 100)
        out["c%d" % n], out["p%d" % n] = at(d), at(1 - d)
        out["rr%d" % n] = out["c%d" % n] - out["p%d" % n]
        out["bf%d" % n] = (out["c%d" % n] + out["p%d" % n]) / 2 - out["atm"]
    return out


def constant_maturity(per_expiry, tenors_days, fields=FIELDS):
    """per_expiry: dicts with 'tau' (years) and smile points. Total-variance interpolation in tau for each
    field; NaN outside the range of listed expiries that have that field."""
    pts = sorted([e for e in per_expiry if e["tau"] > 0], key=lambda e: e["tau"])
    res = {}
    for T in tenors_days:
        t = T / 365.25
        for f in fields:
            xs = [(e["tau"], e[f]) for e in pts if np.isfinite(e.get(f, np.nan))]
            v = float("nan")
            for (t0, s0), (t1, s1) in zip(xs, xs[1:]):
                if t0 <= t <= t1:
                    w = s0 * s0 * t0 + (s1 * s1 * t1 - s0 * s0 * t0) * (t - t0) / (t1 - t0)
                    v = math.sqrt(w / t) if w > 0 else float("nan")
                    break
            res["%s_%dd" % (f, T)] = v
        for n in (25, 10):
            c, p, a = res["c%d_%dd" % (n, T)], res["p%d_%dd" % (n, T)], res["atm_%dd" % T]
            res["rr%d_%dd" % (n, T)] = c - p
            res["bf%d_%dd" % (n, T)] = (c + p) / 2 - a
    return res
