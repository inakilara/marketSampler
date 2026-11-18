"""Stylised facts of an hourly panel (see panel.py for the column contract and STYLIZED_FACTS.md for the
rationale). Every fact function takes a panel and returns

    {"scalars": {name: float}, "curves": {name: pd.Series or pd.DataFrame}}

and silently returns empty results when the columns it needs are absent, so the same code characterises
real data and any synthetic generator.
"""
import numpy as np
import pandas as pd
from scipy import stats

HOURS_PER_YEAR = 24 * 365.25
HORIZONS = (1, 2, 4, 8, 12, 24, 72, 168)  # hours
TENORS = (7, 14, 30, 60, 90, 180)


# ======================================================================= helpers
def log_returns(p):
    return np.log(p["close"]).diff()


def acf(x, nlags):
    """Autocorrelation for lags 0..nlags with NaN-tolerant pairwise normalisation (FFT)."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < nlags + 10:
        return pd.Series(np.nan, index=range(nlags + 1))
    x = np.where(m, x - np.nanmean(x), 0.0)
    w = m.astype(float)
    n = len(x)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    fx, fw = np.fft.rfft(x, size), np.fft.rfft(w, size)
    num = np.fft.irfft(fx * np.conj(fx), size)[: nlags + 1]
    cnt = np.fft.irfft(fw * np.conj(fw), size)[: nlags + 1]
    cov = num / np.maximum(np.round(cnt), 1)
    return pd.Series(cov / cov[0], index=range(nlags + 1))


def pacf(x, nlags):
    """Partial autocorrelation via Durbin-Levinson on the sample ACF."""
    r = acf(x, nlags).values
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    v = 1.0
    for k in range(1, nlags + 1):
        a = (r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))) / v
        phi[k, k] = a
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - a * phi[k - 1, k - j]
        v *= (1 - a * a)
        out[k] = a
    return pd.Series(out, index=range(nlags + 1))


def xcorr(x, y, maxlag):
    """corr(x_t, y_{t+k}) for k = -maxlag..maxlag (k > 0: x leads y)."""
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    return pd.Series({k: x.corr(y.shift(-k)) for k in range(-maxlag, maxlag + 1)})


def hill(x, frac=0.025, min_k=10):
    """Hill tail-index estimate on the largest `frac` share of positive values of x."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    x = x[x > 0]
    k = max(min_k, int(frac * len(x)))
    if len(x) <= k + 1:
        return np.nan
    tail = x[-k:]
    return float(1.0 / np.mean(np.log(tail / x[-k - 1])))


def aggregate(r, h):
    """Non-overlapping h-hour log returns from hourly log returns (incomplete blocks dropped)."""
    r = pd.Series(r).dropna()
    n = len(r) // h * h
    return r.values[:n].reshape(-1, h).sum(axis=1)


def _corr(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return float(a.corr(b))


def _has(p, *cols):
    return all(c in p.columns and p[c].notna().sum() > 100 for c in cols)


def empty():
    return {"scalars": {}, "curves": {}}


# ======================================================================= 1. distribution & aggregation
def distribution(p):
    """Moments (as asked) plus robust shape measures, per horizon -> aggregational Gaussianity and scaling."""
    r = log_returns(p)
    rows, sc = {}, {}
    sd1 = r.std()
    for h in HORIZONS:
        x = aggregate(r, h)
        if len(x) < 30:
            continue
        q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        z = (x - x.mean()) / x.std()
        rows[h] = {
            "n": len(x), "mean": x.mean(), "std": x.std(), "skew": stats.skew(x), "exkurt": stats.kurtosis(x),
            "q01": q[0], "q05": q[1], "q50": q[3], "q95": q[5], "q99": q[6],
            # robust: Bowley-type skew on 5/95 and tail-weight (q99-q01)/(q75-q25) relative to a normal (=3.449)
            "qskew": (q[5] + q[1] - 2 * q[3]) / (q[5] - q[1]),
            "qtail": ((q[6] - q[0]) / (q[4] - q[2])) / 3.449,
            "hill_left": hill(-x), "hill_right": hill(x),
            "ks_normal": stats.kstest(z, "norm").statistic,
            "scaling": x.std() / (sd1 * np.sqrt(h)),  # 1 = random-walk scaling
            "gain_loss": q[6] / -q[0],  # >1: big gains larger than big losses
        }
    df = pd.DataFrame(rows).T
    if 1 in df.index:
        for c in ("mean", "std", "skew", "exkurt", "qskew", "qtail", "hill_left", "hill_right", "gain_loss"):
            sc["r1h_" + c] = float(df.at[1, c])
        sc["r1h_ann_vol"] = float(df.at[1, "std"] * np.sqrt(HOURS_PER_YEAR))
    for h in (24, 168):
        if h in df.index:
            sc["exkurt_%dh" % h] = float(df.at[h, "exkurt"])
            sc["qtail_%dh" % h] = float(df.at[h, "qtail"])
            sc["ks_normal_%dh" % h] = float(df.at[h, "ks_normal"])
            sc["scaling_%dh" % h] = float(df.at[h, "scaling"])
    return {"scalars": sc, "curves": {"by_horizon": df}}


# ======================================================================= 2. dependence & clustering
def deseasonalise_abs(p, r):
    """|r| divided by its mean for the same hour-of-week (168 buckets): removes the intraday/weekly
    profile so volatility clustering is not confused with seasonality."""
    a = r.abs()
    how = p.index.dayofweek * 24 + p.index.hour
    prof = a.groupby(how).transform("mean")
    return a / prof


def dependence(p, long_lag=720):
    r = log_returns(p)
    a_r, a_abs, a_sq = acf(r, 48), acf(r.abs(), long_lag), acf(r ** 2, long_lag)
    a_ds = acf(deseasonalise_abs(p, r), long_lag)
    pa = pacf(r.abs(), 48)
    n = r.notna().sum()
    lb = n * (n + 2) * sum(a_r[k] ** 2 / (n - k) for k in range(1, 25))
    lags = np.arange(24, long_lag + 1)
    y = a_ds.values[24:]
    ok = y > 0
    slope = np.polyfit(np.log(lags[ok]), np.log(y[ok]), 1)[0] if ok.sum() > 10 else np.nan
    sc = {"acf_r_1": a_r[1], "acf_r_max_abs_1_24": float(a_r[1:25].abs().max()),
          "ljung_box_24": float(lb), "ljung_box_24_p": float(1 - stats.chi2.cdf(lb, 24)),
          "acf_abs_1": a_abs[1], "acf_abs_24": a_abs[24], "acf_abs_168": a_abs[168], "acf_abs_720": a_abs[min(720, long_lag)],
          "acf_sq_1": a_sq[1], "pacf_abs_1": pa[1], "pacf_abs_24": pa[24],
          "acf_absds_1": a_ds[1], "acf_absds_24": a_ds[24], "acf_absds_168": a_ds[168],
          "acf_absds_720": a_ds[min(720, long_lag)],
          "acf_absds_powerlaw_slope": float(slope),
          # seasonal bump: how much the raw |r| ACF at 24 h exceeds the deseasonalised one
          "acf_abs_seasonal_bump_24": a_abs[24] - a_ds[24]}
    return {"scalars": {k: float(v) for k, v in sc.items()},
            "curves": {"acf_r": a_r, "acf_abs": a_abs, "acf_absds": a_ds, "acf_sq": a_sq, "pacf_abs": pa}}


def leverage(p, maxlag=48):
    """L(tau) = corr(r_t, r^2_{t+tau}); tau>0: returns lead future volatility (leverage), tau<0: reverse."""
    r = log_returns(p)
    L = xcorr(r, r ** 2, maxlag)
    return {"scalars": {"leverage_mean_1_24": float(L.loc[1:24].mean()), "leverage_mean_m24_m1": float(L.loc[-24:-1].mean()),
                        "leverage_1": float(L.loc[1])},
            "curves": {"leverage": L}}


def zumbach(p, window=24, maxlag=72):
    """Time-reversal asymmetry: corr(coarse vol past window, fine vol shifted by tau)."""
    r = log_returns(p)
    coarse = r.rolling(window).sum().abs()
    fine = np.sqrt((r ** 2).rolling(window).sum())
    C = xcorr(coarse, fine, maxlag)
    asym = float(C.loc[1:maxlag].mean() - C.loc[-maxlag:-1].mean())
    return {"scalars": {"zumbach_asymmetry": asym}, "curves": {"zumbach": C}}


def seasonality(p):
    r = log_returns(p)
    idx = p.index
    d = pd.DataFrame({"absr": r.abs(), "hour": idx.hour, "dow": idx.dayofweek})
    if "volume" in p:
        d["vol"] = p["volume"]
    hp = d.drop(columns="dow").groupby("hour").mean()
    wp = d.drop(columns="hour").groupby("dow").mean()
    hp_rel, wp_rel = hp / hp.mean(), wp / wp.mean()
    sc = {"hour_absr_max_over_min": float(hp["absr"].max() / hp["absr"].min()),
          "weekend_absr_rel": float(wp_rel["absr"].loc[[5, 6]].mean()),
          "funding_hours_absr_rel": float(hp_rel["absr"].loc[[0, 8, 16]].mean())}
    if "vol" in hp:
        sc["hour_vol_max_over_min"] = float(hp["vol"].max() / hp["vol"].min())
        sc["weekend_vol_rel"] = float(wp_rel["vol"].loc[[5, 6]].mean())
    return {"scalars": sc, "curves": {"by_hour": hp_rel, "by_weekday": wp_rel}}


def extremes(p, q=0.99):
    r = log_returns(p).dropna()
    thr = r.abs().quantile(q)
    t = np.flatnonzero(r.abs().values > thr)
    gaps = np.diff(t)
    lc = np.log(p["close"])
    dd = {}
    for w in (168, 720):
        mdd = []
        for i in range(0, len(lc) - w, w):
            s = lc.iloc[i:i + w].values
            mdd.append(float(np.max(np.maximum.accumulate(s) - s)))
        dd[w] = np.array(mdd)
    sc = {"exceed_interarrival_cv": float(gaps.std() / gaps.mean()) if len(gaps) > 2 else np.nan,
          "exceed_within_24h_share": float((gaps <= 24).mean()) if len(gaps) else np.nan,
          "exceed_within_24h_if_iid": float(1 - (1 - (1 - q)) ** 24),
          "maxdd_7d_median": float(np.median(dd[168])), "maxdd_7d_q90": float(np.quantile(dd[168], 0.9)),
          "maxdd_30d_median": float(np.median(dd[720])), "maxdd_30d_q90": float(np.quantile(dd[720], 0.9))}
    return {"scalars": sc, "curves": {"interarrival_hist": pd.Series(np.bincount(np.minimum(gaps, 200)))}}


# ======================================================================= 3. bars & volume
def bars(p):
    if not _has(p, "open", "high", "low", "close"):
        return empty()
    o, h, l, c = p["open"], p["high"], p["low"], p["close"]
    rng = h - l
    ok = rng > 0
    up = ((h - np.maximum(o, c)) / rng)[ok]
    lo = ((np.minimum(o, c) - l) / rng)[ok]
    body = ((c - o).abs() / rng)[ok]
    r_oc = np.log(c / o)
    logrange = np.log(h / l)
    park = (logrange ** 2).mean() / (4 * np.log(2))
    sc = {"upper_wick_mean": float(up.mean()), "lower_wick_mean": float(lo.mean()), "body_mean": float(body.mean()),
          "corr_range_absr": _corr(logrange, r_oc.abs()),
          "parkinson_over_cc_var": float(park / (log_returns(p) ** 2).mean()),
          "acf_logrange_1": float(acf(np.log(logrange.where(logrange > 0)), 1)[1]),
          "invalid_bars": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c))).sum())}
    curves = {"body_hist": pd.Series(np.histogram(body, bins=20, range=(0, 1))[0])}
    if _has(p, "volume"):
        sc["corr_logrange_logvol"] = _corr(np.log(logrange.where(logrange > 0)), np.log(p["volume"]))
    return {"scalars": sc, "curves": curves}


def volume(p):
    if not _has(p, "volume"):
        return empty()
    r = log_returns(p)
    lv = np.log(p["volume"].where(p["volume"] > 0))
    a = acf(lv, 168)
    sc = {"logvol_std": float(lv.std()), "acf_logvol_1": float(a[1]), "acf_logvol_24": float(a[24]),
          "acf_logvol_168": float(a[168]), "corr_logvol_absr": _corr(lv, r.abs())}
    curves = {"acf_logvol": a, "xcorr_absr_logvol": xcorr(r.abs(), lv, 24)}
    if _has(p, "taker_buy_volume"):
        imb = 2 * p["taker_buy_volume"] / p["volume"] - 1
        sc["corr_imbalance_r"] = _corr(imb, r)
        sc["corr_imbalance_next_r"] = _corr(imb, r.shift(-1))
        curves["xcorr_imbalance_r"] = xcorr(imb, r, 24)
    return {"scalars": sc, "curves": curves}


# ======================================================================= 4. open interest & funding
def open_interest(p):
    if not _has(p, "oi"):
        return empty()
    r = log_returns(p)  # return of bar t (close_t / close_{t-1}) ...
    loi = np.log(p["oi"])
    doi = loi.shift(-1) - loi  # ... OI change over bar t (oi is sampled at the bar open)
    rb = np.log(p["close"] / p["open"])  # in-bar return, aligned with doi
    q01, q99 = rb.quantile([0.01, 0.99])
    sc = {"doi_std": float(doi.std()), "acf_doi_1": float(acf(doi, 1)[1]),
          "corr_r_doi": _corr(rb, doi), "corr_absr_doi": _corr(rb.abs(), doi),
          "doi_mean_when_r_below_q01": float(doi[rb < q01].mean()),
          "doi_mean_when_r_above_q99": float(doi[rb > q99].mean()),
          "doi_mean_all": float(doi.mean()),
          "corr_doi_next_r": _corr(doi, rb.shift(-1)),
          "corr_trailing24_r_oi_level_change": _corr(r.rolling(24).sum(), loi.diff(24))}
    return {"scalars": sc, "curves": {"xcorr_r_doi": xcorr(rb, doi, 24), "xcorr_absr_doi": xcorr(rb.abs(), doi, 24)}}


def funding(p):
    if not _has(p, "funding"):
        return empty()
    f8 = p["funding"][p.index.hour % 8 == 0]  # one value per settlement
    r = log_returns(p)
    lc = np.log(p["close"])
    fwd = {h: (lc.shift(-h) - lc).reindex(f8.index) for h in (8, 24, 72)}
    back = {h: (lc - lc.shift(h)).reindex(f8.index) for h in (8, 24, 72, 168)}
    a = acf(f8, 90)
    sc = {"funding_mean": float(f8.mean()), "funding_std": float(f8.std()),
          "funding_ann_mean": float(f8.mean() * 3 * 365.25),
          "funding_share_negative": float((f8 < 0).mean()),
          "funding_q99": float(f8.quantile(0.99)), "funding_q01": float(f8.quantile(0.01)),
          "acf_funding_1settle": float(a[1]), "acf_funding_3settle": float(a[3]), "acf_funding_21settle": float(a[21])}
    for h, x in back.items():
        sc["corr_funding_trailing_r_%dh" % h] = _corr(f8, x)
    for h, x in fwd.items():
        sc["corr_funding_next_r_%dh" % h] = _corr(f8, x)
    if _has(p, "oi"):
        sc["corr_funding_oi_change_24h"] = _corr(f8, np.log(p["oi"]).diff(24).reindex(f8.index))
    return {"scalars": sc, "curves": {"acf_funding": a}}


# ======================================================================= 5. options
def options(p):
    if not _has(p, "atm_30d"):
        return empty()
    r = log_returns(p)
    rb = np.log(p["close"] / p["open"])  # bar return aligned with the IV change over the bar
    sc, curves, beta_rows = {}, {}, {}
    for T in TENORS:
        a = "atm_%dd" % T
        if not _has(p, a):
            continue
        row = {}
        for f in ("atm", "rr25", "bf25"):
            col = "%s_%dd" % (f, T)
            if not _has(p, col):
                continue
            d = p[col].shift(-1) - p[col]
            ok = d.notna() & rb.notna()
            if ok.sum() > 100:
                row["beta_" + f] = float(np.polyfit(rb[ok], d[ok], 1)[0])  # vol points (decimal) per unit log-return
                row["corr_" + f] = _corr(rb[ok], d[ok])
        beta_rows[T] = row
    if beta_rows:
        bt = pd.DataFrame(beta_rows).T
        curves["spot_vol_by_tenor"] = bt
        for c in bt.columns:
            if 30 in bt.index:
                sc["%s_30d" % c] = float(bt.at[30, c])
    atm = p["atm_30d"]
    rv_fwd = np.sqrt((r ** 2).rolling(720).sum().shift(-720) * HOURS_PER_YEAR / 720)
    rv_back = np.sqrt((r ** 2).rolling(720).sum() * HOURS_PER_YEAR / 720)
    vrp = atm - rv_fwd
    sc.update({"atm30_mean": float(atm.mean()), "atm30_std": float(atm.std()),
               "vrp30_mean": float(vrp.mean()), "vrp30_median": float(vrp.median()),
               "vrp30_share_positive": float((vrp.dropna() > 0).mean()),
               "corr_atm30_trailing_rv30": _corr(atm, rv_back),
               "corr_atm30_future_rv30": _corr(atm, rv_fwd)})
    d_atm = atm.resample("D").first()  # IV persistence on daily samples
    ad = acf(d_atm, 60)
    sc["acf_atm30_daily_1"] = float(ad[1])
    sc["atm30_halflife_days"] = float(np.log(0.5) / np.log(ad[1])) if 0 < ad[1] < 1 else np.nan
    curves["acf_atm30_daily"] = ad
    if _has(p, "rr25_30d"):
        rr = p["rr25_30d"]
        sc["rr25_30d_mean"] = float(rr.mean())
        sc["rr25_30d_share_positive"] = float((rr.dropna() > 0).mean())
        sc["corr_rr25_30d_trailing_r_7d"] = _corr(rr, np.log(p["close"]).diff(168))
    if _has(p, "bf25_30d"):
        volvol = (atm.diff()).rolling(168).std()
        sc["bf25_30d_mean"] = float(p["bf25_30d"].mean())
        sc["corr_bf25_30d_volvol_7d"] = _corr(p["bf25_30d"], volvol)
    if _has(p, "atm_7d", "atm_90d"):
        slope = p["atm_7d"] - p["atm_90d"]
        sc["term_slope_7_90_mean"] = float(slope.mean())
        sc["term_slope_share_inverted"] = float((slope.dropna() > 0).mean())
        sc["corr_term_slope_trailing_rv_24h"] = _corr(slope, np.sqrt((r ** 2).rolling(24).sum() * HOURS_PER_YEAR / 24))
    if _has(p, "dvol"):
        sc["dvol_minus_atm30_mean"] = float((p["dvol"] - atm).mean())
    curves["xcorr_r_datm30"] = xcorr(rb, atm.shift(-1) - atm, 24)
    return {"scalars": sc, "curves": curves}


# ======================================================================= 6. joint structure
def joint(p, maxlag=24):
    r = log_returns(p)
    cols = {"r": r, "absr": r.abs()}
    if _has(p, "volume"):
        cols["dlogvol"] = np.log(p["volume"]).diff()
    if _has(p, "oi"):
        cols["doi"] = np.log(p["oi"]).diff().shift(-1)
    if _has(p, "funding"):
        cols["dfunding"] = p["funding"].diff()
    for c in ("atm_30d", "rr25_30d", "bf25_30d"):
        if _has(p, c):
            cols["d" + c] = p[c].shift(-1) - p[c]
    d = pd.DataFrame(cols)
    C = d.corr()
    lead = {}
    names = list(d.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = xcorr(d[a], d[b], maxlag).drop(0)
            k = int(x.abs().idxmax())
            lead["%s->%s" % (a, b)] = {"lag": k, "corr": float(x[k]), "corr_lag0": float(d[a].corr(d[b]))}
    sc = {"corr_%s_%s" % (a, b): float(C.at[a, b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return {"scalars": sc, "curves": {"corr": C, "leadlag": pd.DataFrame(lead).T}}


ALL = {"distribution": distribution, "dependence": dependence, "leverage": leverage, "zumbach": zumbach,
       "seasonality": seasonality, "extremes": extremes, "bars": bars, "volume": volume,
       "open_interest": open_interest, "funding": funding, "options": options, "joint": joint}


def characterise(p, groups=None):
    """Run every fact group on panel p -> {group: {"scalars":..., "curves":...}}."""
    out = {}
    for name, fn in ALL.items():
        if groups and name not in groups:
            continue
        try:
            out[name] = fn(p)
        except Exception as e:  # one failing group must not hide the others
            out[name] = {"scalars": {"error": str(e)}, "curves": {}}
    return out


def fingerprint(results):
    """Flatten scalars -> {group.name: value}."""
    return {"%s.%s" % (g, k): v for g, res in results.items() for k, v in res["scalars"].items()}
