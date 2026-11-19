"""Structural generator (approach A in models/README.md): hourly BTC perp OHLCV + open interest + funding.

Layers (each conditioned on the ones above; all fitted on the real panel by fit()):

1. Returns core
     r_t = mu + s(how_t) * x_t,   x_t = sigma_t * z_t                  s: hour-of-week volatility profile
     sigma_t^2 = w + sum_k b_k * mean(x^2 over previous k hours) + g * min(x_{t-1}, 0)^2
     k in 1h, 4h, 1d, 1w, 30d, 90d (HARCH cascade: long memory, Zumbach asymmetry); g: leverage.
     Gaussian QML fit, variance targeting.
2. Intrabar OHLC: open = previous close; a 12-step heavy-tailed (Student-t) bridge from open to close with
     step scale kappa * s * sigma_t / sqrt(12); high/low = path extremes. kappa is calibrated so the
     Parkinson / close-to-close variance ratio matches the real one.
3. Volume: log V_t = season_v(how) + a + c1 log(s sigma_t) + c2 |z_t| + c3 z_t + eta_t,
     eta_t HAR(1h, 1d, 1w) â€” persistent volume level. Taker-buy share: logit = d0 + d1 z_t + e_t.
4. Open interest: dlogOI_t = a + b1 r_t + b2 |r_t| + phi dlogOI_{t-1} - kappa_oi (logOI_t - m) + e_t
     (positive coupling with returns, drops in large moves = liquidations, slow mean reversion of the level).
5. Funding (every 8 h) through a one-sided dead band on a latent rate L, I = 0.01 %, c = 0.01 %:
     F = L below I, F = I for I <= L <= I + c (real point mass ~28 %), F = L - c above (bull phases).
     L_k = a + phi1 L_{k-1} + phi2 mean(L_{k-1..k-21}) + psi (R8, R72, R168, R720) + psi_oi dlogOI_24h + e_k
     (persistent, follows trailing returns over 8 hâ€¦30 d and OI). Stochastic-EM fit of the censored L.
     Funding feeds back weakly and negatively on the next returns (drift delta * (F - trailing 30-day mean F)).

Innovations: filtered historical simulation. The residuals of layers 1, 3 and 4 for the same real hour are
resampled together (stationary bootstrap, mean block 24 h), so same-bar cross-dependence and empirical tails
are kept; funding residuals are resampled per settlement.

Scenarios (keyword arguments of generate(); every "path" argument is a scalar or an array of n_hours values):
  vol_level     long-run annualised volatility (rescales the variance-targeting constant w)
  start_vol     annualised volatility of the 90-day history the path starts from, or "calm" / "normal" /
                "stressed" (10th / 50th / 90th percentile of real 30-day realised vol)
  vol_mult      multiplier on the conditional volatility per hour (e.g. a stress window); it feeds the cascade,
                so a stressed episode raises the volatility that follows it, as in real data
  drift         annualised log drift (replaces the fitted mean return)
  jumps         {hour offset or timestamp: log return} forced bars (crash / squeeze); volume, OI, funding and
                the later volatility react to them through the model
  funding_shift shift of the long-run latent funding premium per 8 h (e.g. 3e-4 = +0.03 % per settlement)
  funding0      funding rate at the start
  p0, oi0       starting price and open interest
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.common import hour_of_week, hourly_index, seasonal_profile, stationary_bootstrap_indices

COMPONENTS = (1, 4, 24, 168, 720, 2160)
PARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params")
SUBSTEPS = 60  # one-minute steps: coarser bridges give too noisy a range (checked: 12 steps -> corr(range,|r|) 0.77)
T_DF = 6.0
# funding dead band: base rate I and band width c above it. c is not observable directly; 1e-4 was chosen by
# simulation (share pinned at the base: c=5e-5 0.18, 1e-4 0.29, 2e-4 0.41, 5e-4 0.55; real 0.28)
FUND_I, FUND_C = 1e-4, 1e-4
HOURS_YEAR = 8766
# cap on the funding persistence (L_lag + L_mean21 coefficients). Fitted on one calendar year the free estimate can
# exceed 1 (2023: 1.005): funding then trends without bound in a bull path and, through the negative funding ->
# return feedback, turns the drift negative (synthetic 2023 -87 %/y vs real +94 %). At 0.98 the 2023 card still ran
# funding at 3x the real level; 0.95 gives 1.2-1.5e-4 (real 0.7e-4). Whole history: 0.925, below the cap.
FUND_PERS_MAX = 0.95
FUND_FB_H = 720  # trailing window (hours) of the funding level the return feedback is measured against
JUMP_Z = 8.0  # cap on the shock a forced jump passes to volume / OI / intrabar / the volatility cascade
STATE_FILE = "structural_state.npz"
# fitted attributes stored by save_state(): arrays go to the npz, scalars/lists to a JSON header
_STATE_ARRAYS = ("season", "x_hist", "season_lv", "vol_beta", "eta_beta", "eta_hist", "tb_beta", "oi_beta", "J",
                 "fund_beta", "fund_resid", "P_hist", "b", "rv_quantiles")
_STATE_SCALARS = ("mu", "p0", "oi_mean", "oi0", "f_mean", "delta", "w", "w_qml", "g", "persistence", "x_var",
                  "kappa", "theta", "iscale_norm", "intrabar_fit", "fit_info", "data_end", "resid_block")


def _compact(a):
    """Large tables (resampled residuals, histories) as float32 to halve file size; parameters stay float64."""
    a = np.asarray(a, float)
    return a.astype(np.float32) if a.size > 10000 else a


def _as_path(v, n, name):
    a = np.broadcast_to(np.asarray(v, float), (n,)) if np.ndim(v) == 0 else np.asarray(v, float)
    if len(a) != n:
        raise ValueError("%s must be a scalar or have n_hours=%d values (got %d)" % (name, n, len(a)))
    return a


def premium_from_funding(F):
    """A latent premium that maps to funding F (inverse of the dead band; mid-band for the base rate)."""
    if F < FUND_I - 1e-12:
        return F
    if F <= FUND_I + 1e-12:
        return FUND_I + 0.5 * FUND_C
    return F + FUND_C


def rolling_means_lagged(x2, comps):
    """M[t, j] = mean(x2[t-k_j : t]) (previous k_j values), NaN until enough history."""
    c = np.r_[0.0, np.cumsum(x2)]
    T = len(x2)
    M = np.full((T, len(comps)), np.nan)
    for j, k in enumerate(comps):
        t = np.arange(k, T)
        M[t, j] = (c[t] - c[t - k]) / k
    return M


def lagged_mean(y, k):
    """mean of y[t-k : t] (previous k values), NaN-tolerant, NaN until k values exist."""
    s = pd.Series(y)
    return s.shift(1).rolling(k, min_periods=max(1, k // 2)).mean().values


def ols(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res = np.full(len(y), np.nan)
    res[ok] = y[ok] - X[ok] @ beta
    return beta, res


def funding_from_premium(L):
    """Empirical one-sided dead band (fits the real distribution: continuous below the 0.01 % base,
    a point mass at it, and values above only in strong bull phases):
        L < I          -> F = L
        I <= L <= I+c  -> F = I
        L > I + c      -> F = L - c"""
    L = np.asarray(L, float)
    return np.where(L < FUND_I, L, np.where(L <= FUND_I + FUND_C, FUND_I, L - FUND_C))


def intrabar_extremes(r, scale, rng):
    """High/low log excursions (relative to open) of a heavy-tailed bridge from 0 to r with per-bar
    step scale `scale` (intrabar volatility of the whole bar). Extremes always contain open and close."""
    n = len(r)
    e = rng.standard_t(T_DF, size=(n, SUBSTEPS)) / np.sqrt(T_DF / (T_DF - 2))
    W = np.cumsum(e, axis=1) * (scale / np.sqrt(SUBSTEPS))[:, None]
    frac = np.arange(1, SUBSTEPS + 1) / SUBSTEPS
    path = W - frac[None, :] * (W[:, -1] - r)[:, None]
    hi = np.maximum(np.maximum(path.max(axis=1), 0.0), r)
    lo = np.minimum(np.minimum(path.min(axis=1), 0.0), r)
    return hi, lo


def intrabar_scale(sbar, activity, kappa, theta, norm):
    """Intrabar volatility = bar volatility x exp(theta * activity), where activity is the bar's volume in
    excess of what its volatility and hour explain (shock-size term + persistent volume level). Busy bars
    move more inside the bar; because the volume level is persistent, so is the range."""
    return kappa * sbar * np.exp(theta * activity) / norm


class StructuralModel:
    name = "structural"

    def __init__(self, components=COMPONENTS, resid_block=24):
        self.components = tuple(components)
        self.resid_block = resid_block

    # ================================================================== fit
    def fit(self, p):
        r_all = np.log(p["close"]).diff()
        ok = r_all.notna().values
        idx = p.index[ok]
        P = p.loc[idx]
        r = r_all.values[ok]
        how = hour_of_week(idx)
        # ---- 1. returns core
        self.mu = float(np.mean(r))
        self.season = seasonal_profile(np.abs(r - self.mu), how, smooth=1)
        x = (r - self.mu) / self.season[how]
        sigma, z = self._fit_harch(x)  # arrays aligned with idx (NaN during burn-in)
        sbar = self.season[how] * sigma
        self.p0 = float(p["close"].iloc[-1])
        self.x_hist = x
        self.x_var = float(np.mean(x * x))
        self.data_end = str(idx[-1])
        rv30 = pd.Series(r).rolling(720).std().dropna() * np.sqrt(HOURS_YEAR)
        self.rv_quantiles = rv30.quantile([.1, .5, .9]).values  # calm / normal / stressed start presets
        # ---- 2. intrabar
        # ---- 3. volume and taker flow
        lv = np.log(P["volume"].values)
        # the hour-of-week volume profile already contains the intraday activity pattern, so volume is
        # regressed on the *deseasonalised* volatility sigma (not s * sigma) to avoid counting it twice
        self.season_lv = pd.Series(lv).groupby(how).mean().reindex(range(168)).values
        self.season_lv = self.season_lv - self.season_lv.mean()
        y = lv - self.season_lv[how]
        X = np.column_stack([np.ones_like(y), np.log(sigma), np.abs(z), z])
        self.vol_beta, eta = ols(y, X)
        # persistent level: HAR fitted within calendar years (pooling years makes the level shifts between
        # years look like a unit root and gives far too persistent volume inside any one year)
        yr = idx.year.values
        eta_dm = eta - pd.Series(eta).groupby(yr).transform("mean").values
        Xe = np.column_stack([np.ones_like(eta_dm), lagged_mean(eta_dm, 1), lagged_mean(eta_dm, 24), lagged_mean(eta_dm, 168)])
        self.eta_beta, e_v = ols(eta_dm, Xe)
        self.eta_hist = eta_dm
        # ---- 2. intrabar (calibrated after volume: intrabar activity is linked to excess volume)
        activity = self.vol_beta[2] * np.abs(z) + self.vol_beta[3] * z + eta_dm
        logrange = np.log(P["high"] / P["low"]).values
        r_oc = np.log(P["close"] / P["open"]).values
        targets = {"ratio": np.nanmean(logrange ** 2) / np.nanmean(r ** 2),
                   "corr_absr": pd.Series(logrange).corr(pd.Series(np.abs(r_oc))),
                   "corr_lv": pd.Series(np.log(logrange)).corr(pd.Series(lv)),
                   "acf1": pd.Series(np.log(logrange)).autocorr(1)}
        self._calibrate_intrabar(r, sbar, activity, lv, targets)
        target = targets["ratio"]
        tb = np.clip(P["taker_buy_volume"].values / P["volume"].values, 0.01, 0.99)
        lt = np.log(tb / (1 - tb))
        self.tb_beta, e_tb = ols(lt, np.column_stack([np.ones_like(z), z]))
        # ---- 4. open interest
        loi = np.log(P["oi"].values)
        doi = np.r_[loi[1:] - loi[:-1], np.nan]  # change over bar t (oi sampled at bar open)
        self.oi_mean = float(np.nanmean(loi))
        Xo = np.column_stack([np.ones_like(r), r, np.abs(r), np.r_[np.nan, doi[:-1]], loi - self.oi_mean])
        self.oi_beta, e_oi = ols(doi, Xo)
        self.oi0 = float(P["oi"].dropna().iloc[-1])
        # ---- joint residual table (same real hour)
        J = np.column_stack([z, e_v, e_tb, e_oi])
        keep = np.all(np.isfinite(J), axis=1)
        self.J = J[keep]
        self.J[:, 0] = (self.J[:, 0] - self.J[:, 0].mean()) / self.J[:, 0].std()
        # ---- 5. funding, and its (weak, negative) feedback on the next returns
        self._fit_funding(P, r, idx, loi)
        fcur = P["funding"].values
        self.f_mean = float(np.nanmean(fcur))
        # feedback on deviations from the trailing 30-day funding mean (spikes predict weaker returns). Against the
        # fixed sample mean, any path whose funding level runs higher (a bull scenario, funding_shift, a one-year
        # fit) got a permanent negative drift (2023 card: -87 %/y vs real +94 %).
        fbar = pd.Series(fcur).rolling(FUND_FB_H, min_periods=24).mean().shift(1).values
        okf = np.isfinite(fcur) & np.isfinite(fbar)
        self.delta = float(np.polyfit(fcur[okf] - fbar[okf], r[okf], 1)[0])
        self.fit_info.update({"kappa": self.kappa, "theta": self.theta, "parkinson_target": float(target),
                              "intrabar_fit": self.intrabar_fit, "n_joint_resid": int(keep.sum())})
        return self

    def _fit_harch(self, x):
        K = len(self.components)
        x2 = x * x
        M = rolling_means_lagged(x2, self.components)
        neg = np.r_[np.nan, np.minimum(x[:-1], 0) ** 2]
        burn = max(self.components)
        Mb, negb, xb2 = M[burn:], neg[burn:], x2[burn:]
        v = x2.mean()

        def unpack(th):
            e = np.exp(th)
            return e[0], e[1:K + 1], e[K + 1]

        def nll(th):
            w, b, g = unpack(th)
            s2 = w + Mb @ b + g * negb
            return 0.5 * np.mean(np.log(s2) + xb2 / s2)

        th0 = np.log(np.r_[0.05 * v, np.full(K, 0.9 / K), 0.05])
        res = minimize(nll, th0, method="L-BFGS-B")
        w, b, g = unpack(res.x)
        s2 = w + M @ b + g * neg
        sigma = np.sqrt(s2)
        z = x / sigma
        zb = z[burn:]
        zs = (zb - zb.mean()) / zb.std()
        self.b, self.g = b.astype(float), float(g)
        self.persistence = float(b.sum() + g * np.mean(np.minimum(zs, 0) ** 2))
        self.w_qml = float(w)
        self.w = float(v * (1 - self.persistence))  # variance targeting
        self.fit_info = {"nll": float(res.fun), "converged": bool(res.success), "persistence": self.persistence,
                         "w_qml": self.w_qml, "w_targeted": self.w}
        return sigma, z

    def _calibrate_intrabar(self, r, sbar, activity, lv, T):
        """Grid over theta (link of intrabar vol to excess volume); for each, bisection on kappa to hit the
        Parkinson / close-close variance ratio; keep the theta that best matches corr(range, |r|),
        corr(log range, log volume) and the lag-1 ACF of log range."""
        ok = np.isfinite(r) & np.isfinite(sbar) & np.isfinite(activity) & np.isfinite(lv)
        r, sbar, activity, lv = r[ok], sbar[ok], activity[ok], lv[ok]
        # full sample on purpose: calibrating on the last 15,000 h only picked a different theta and failed
        # 3 range facts (the recent period is not representative of the whole)
        best = None
        for theta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            norm = np.mean(np.exp(theta * activity))
            lo, hi = 0.05, 4.0
            for _ in range(22):
                k = 0.5 * (lo + hi)
                u, d = intrabar_extremes(r, intrabar_scale(sbar, activity, k, theta, norm), np.random.default_rng(123))
                if np.mean((u - d) ** 2) / np.mean(r ** 2) < T["ratio"]:
                    lo = k
                else:
                    hi = k
            k = 0.5 * (lo + hi)
            u, d = intrabar_extremes(r, intrabar_scale(sbar, activity, k, theta, norm), np.random.default_rng(123))
            lr = np.log(u - d)
            got = {"corr_absr": np.corrcoef(u - d, np.abs(r))[0, 1], "corr_lv": np.corrcoef(lr, lv)[0, 1],
                   "acf1": np.corrcoef(lr[1:], lr[:-1])[0, 1]}
            loss = sum(abs(got[key] - T[key]) for key in got)
            if best is None or loss < best[0]:
                best = (loss, theta, k, norm, got)
        _, theta, k, norm, got = best
        self.theta, self.kappa, self.iscale_norm = float(theta), float(k), float(norm)
        self.intrabar_fit = {"targets": {a: round(float(b), 3) for a, b in T.items()},
                             "fitted": {a: round(float(b), 3) for a, b in got.items()}}

    def _fit_funding(self, P, r, idx, loi):
        settle = (idx.hour % 8 == 0)
        f = P["funding"].values
        rs = pd.Series(r, index=idx)
        lc = rs.cumsum()
        # drivers known at settlement t: returns up to the open of bar t, OI change over the previous 24 h
        R8 = (lc.shift(1) - lc.shift(9)).values
        R72 = (lc.shift(1) - lc.shift(73)).values
        R168 = (lc.shift(1) - lc.shift(169)).values
        R720 = (lc.shift(1) - lc.shift(721)).values
        dOI = (pd.Series(loi).shift(0) - pd.Series(loi).shift(24)).values
        F = f[settle]
        D = np.column_stack([R8[settle], R72[settle], R168[settle], R720[settle], dOI[settle]])
        cens = np.isclose(F, FUND_I, atol=1e-10)
        Pl = np.where(F > FUND_I, F + FUND_C, F)  # invert the dead band; censored start at I
        lo_c, hi_c = FUND_I, FUND_I + FUND_C
        rng = np.random.default_rng(7)
        pool = None
        for it in range(40):  # stochastic EM: censored premia redrawn from the model, truncated to the band
            X = np.column_stack([np.ones(len(Pl)), np.r_[np.nan, Pl[:-1]], lagged_mean(Pl, 21), D])
            beta, res = ols(Pl, X)
            if beta[1] + beta[2] > FUND_PERS_MAX:
                # impose b1 + b2 = FUND_PERS_MAX: y - c*M = b1 (L - M) + others (substituting b2 = c - b1)
                Xc = X.copy()
                Xc[:, 1] = X[:, 1] - X[:, 2]
                yc = Pl - FUND_PERS_MAX * X[:, 2]
                bc, _ = ols(yc, np.delete(Xc, 2, axis=1))
                beta = np.r_[bc[:2], FUND_PERS_MAX - bc[1], bc[2:]]
                res = Pl - X @ beta
            pred = X @ beta
            unc = np.isfinite(res) & ~cens
            pool = res[unc] - res[unc].mean()
            draw = pred + pool[rng.integers(0, len(pool), len(pred))]
            for _ in range(20):  # rejection sampling into the censoring band
                bad = cens & ((draw < lo_c) | (draw > hi_c))
                if not bad.any():
                    break
                draw[bad] = pred[bad] + pool[rng.integers(0, len(pool), bad.sum())]
            Pl = np.where(cens & np.isfinite(pred), np.clip(draw, lo_c, hi_c), Pl)
        self.fund_beta = beta
        self.fund_resid = pool  # centred: a non-zero mean would shift the level of a persistent process
        self.P_hist = Pl
        self.fit_info["funding_censored_share"] = float(cens.mean())

    # ================================================================== generate
    def annual_vol(self):
        """Unconditional annualised volatility implied by the fitted returns core."""
        return float(np.sqrt(HOURS_YEAR * np.mean(self.season ** 2) * self.w / (1 - self.persistence)))

    def generate(self, n_hours, seed=0, start="2030-01-07", p0=None, oi0=None, vol_level=None, start_vol=None,
                 vol_mult=None, drift=None, jumps=None, funding_shift=None, funding0=None):
        """Hourly synthetic panel (open high low close volume taker_buy_volume oi funding). Scenario arguments:
        see the module docstring. With none of them, the path is a draw from the fitted model."""
        rng = np.random.default_rng(seed)
        idx = hourly_index(start, n_hours)
        how = hour_of_week(idx)
        comps = np.array(self.components)
        H = int(comps.max())
        season_t = self.season[how]
        # ---- scenario inputs
        # long-run level: variance targeting at the requested level (a path makes a slow regime change: the
        # cascade drifts towards the new level with its own long memory)
        lev = _as_path(self.annual_vol() if vol_level is None else vol_level, n_hours, "vol_level") / self.annual_vol()
        w_t = self.w * lev ** 2
        mult = _as_path(1.0 if vol_mult is None else vol_mult, n_hours, "vol_mult")
        mu_t = _as_path(self.mu if drift is None else drift / HOURS_YEAR, n_hours, "drift")
        fshift = _as_path(0.0 if funding_shift is None else funding_shift, n_hours, "funding_shift")
        forced = np.full(n_hours, np.nan)
        for when, ret in (jumps or {}).items():
            if isinstance(when, (int, np.integer)):
                t = int(when)
            else:
                ts = pd.Timestamp(when)
                t = int(idx.get_loc(ts.tz_localize("UTC") if ts.tz is None else ts))
            forced[t] = ret
        # ---- joint innovations
        ji = stationary_bootstrap_indices(n_hours, len(self.J), self.resid_block, rng)
        Z, EV, ETB, EOI = self.J[ji].T.copy()
        # ---- 1 + 4 + 5. returns, open interest and funding stepped together (funding feeds back on returns)
        j0 = rng.integers(0, len(self.x_hist) - H)
        hist = np.empty(H + n_hours)
        hist[:H] = self.x_hist[j0:j0 + H] * lev[0]  # a requested level also applies to the starting history
        if start_vol is not None:  # rescale the starting history so its last 30 days have the requested vol
            if isinstance(start_vol, str):
                start_vol = float(self.rv_quantiles[["calm", "normal", "stressed"].index(start_vol)])
            past_season = self.season[hour_of_week(idx[0] - pd.to_timedelta(np.arange(720, 0, -1), "h"))]
            rv = np.sqrt(HOURS_YEAR * np.mean((past_season * hist[H - 720:H]) ** 2))
            hist[:H] *= start_vol / rv
        past_how = hour_of_week(idx[0] - pd.to_timedelta(np.arange(H, 0, -1), "h"))
        r_pre = self.mu + self.season[past_how] * hist[:H]  # returns before the start (burn-in for other layers)
        x2 = hist * hist
        sums = np.array([x2[H - k:H].sum() for k in comps])
        b, g = self.b, self.g
        a_oi, b1, b2, phi, kap = self.oi_beta
        fb = self.fund_beta
        settle = (idx.hour % 8 == 0)
        sig = np.empty(n_hours)
        r = np.empty(n_hours)
        lcum = np.zeros(n_hours + 1)  # log open of bar t relative to start
        loi = np.empty(n_hours + 1)
        loi[0] = np.log(oi0 or self.oi0)
        prev_doi = 0.0
        Ph = list(self.P_hist[-21:])
        if funding0 is not None:
            Ph = [premium_from_funding(funding0)] * 21
        cur_f = float(funding_from_premium(Ph[-1]))
        fund = np.empty(n_hours)
        fres = self.fund_resid[rng.integers(0, len(self.fund_resid), n_hours // 8 + 2)]
        f_pers = 1.0 - fb[1] - fb[2]  # intercept shift that moves the long-run premium by funding_shift
        fsum = self.f_mean * FUND_FB_H  # running sum of the last FUND_FB_H hourly funding values (feedback reference)
        fbuf = np.full(FUND_FB_H, self.f_mean)
        kf = 0
        for t in range(n_hours):
            if settle[t]:  # settlement at the open of bar t, from price/OI information up to now
                R8 = lcum[t] - lcum[max(t - 8, 0)]
                R72 = lcum[t] - lcum[max(t - 72, 0)]
                R168 = lcum[t] - lcum[max(t - 168, 0)]
                R720 = lcum[t] - lcum[max(t - 720, 0)]
                dOI = loi[t] - loi[max(t - 24, 0)]
                Pk = (fb[0] + f_pers * fshift[t] + fb[1] * Ph[-1] + fb[2] * np.mean(Ph[-21:]) + fb[3] * R8
                      + fb[4] * R72 + fb[5] * R168 + fb[6] * R720 + fb[7] * dOI + fres[kf])
                kf += 1
                Ph.append(Pk)
                cur_f = float(funding_from_premium(Pk))
            fund[t] = cur_f
            fdev = cur_f - fsum / FUND_FB_H  # feedback acts on deviations from the trailing 30-day level
            fsum += cur_f - fbuf[t % FUND_FB_H]
            fbuf[t % FUND_FB_H] = cur_f
            i = H + t
            s0 = np.sqrt(w_t[t] + np.dot(b, sums / comps) + g * min(hist[i - 1], 0.0) ** 2)
            s = mult[t] * s0
            if np.isfinite(forced[t]):
                # forced bar: the price moves exactly as requested, but the shock the other layers and the
                # cascade see is capped at JUMP_Z (real crash bars: |z| ~ 7-8; the linear cascade and volume
                # equations extrapolated to |z| ~ 40 give a second crash and 600x volume, unlike real data)
                zf = (forced[t] - mu_t[t] - self.delta * fdev) / (season_t[t] * s)
                Z[t] = np.clip(zf, -JUMP_Z, JUMP_Z)
                xi = s * zf
            else:
                xi = s * Z[t]
            # the cascade sees the shock at the un-multiplied scale: vol_mult is a transient overlay (feeding
            # it back is explosive, the cascade's persistence is ~1)
            hist[i] = s0 * Z[t]
            x2[i] = hist[i] * hist[i]
            sums += x2[i] - x2[i - comps]
            sig[t] = s
            rt = mu_t[t] + self.delta * fdev + season_t[t] * xi
            r[t] = rt
            lcum[t + 1] = lcum[t] + rt
            dl = a_oi + b1 * rt + b2 * abs(rt) + phi * prev_doi + kap * (loi[t] - self.oi_mean) + EOI[t]
            loi[t + 1] = loi[t] + dl
            prev_doi = dl
        sbar = season_t * sig
        p_start = p0 or self.p0
        close = p_start * np.exp(lcum[1:])
        open_ = np.r_[p_start, close[:-1]]
        oi = np.exp(loi[:-1])
        # ---- 3. volume + taker
        c0, c1, c2, c3 = self.vol_beta
        e0, e1, e24, e168 = self.eta_beta
        eh = self.eta_hist[np.isfinite(self.eta_hist)]
        k0 = rng.integers(0, len(eh) - 168)
        eta = np.r_[eh[k0:k0 + 168], np.zeros(n_hours)]
        csum = np.r_[0.0, np.cumsum(eta)]
        for t in range(n_hours):
            i = 168 + t
            m1 = eta[i - 1]
            m24 = (csum[i] - csum[i - 24]) / 24
            m168 = (csum[i] - csum[i - 168]) / 168
            eta[i] = e0 + e1 * m1 + e24 * m24 + e168 * m168 + EV[t]
            csum[i + 1] = csum[i] + eta[i]
        lv =self.season_lv[how] + c0 + c1 * np.log(sig) + c2 * np.abs(Z) + c3 * Z + eta[168:]
        vol = np.exp(lv)
        tb = 1 / (1 + np.exp(-(self.tb_beta[0] + self.tb_beta[1] * Z + ETB)))
        # ---- 2. intrabar (needs the volume activity)
        activity = c2 * np.abs(Z) + c3 * Z + eta[168:]
        u, d = intrabar_extremes(r, intrabar_scale(sbar, activity, self.kappa, self.theta, self.iscale_norm), rng)
        out = pd.DataFrame(index=idx)
        out["open"] = open_
        out["close"] = close
        # explicit clamp: close comes from a cumulative sum, high/low from the open, so they can differ by
        # one floating-point unit
        out["high"] = np.maximum(open_ * np.exp(u), np.maximum(open_, close))
        out["low"] = np.minimum(open_ * np.exp(d), np.minimum(open_, close))
        out = out[["open", "high", "low", "close"]]
        out["volume"] = vol
        out["taker_buy_volume"] = vol * tb
        out["oi"] = oi
        out["funding"] = fund
        self.last_state = {"sigma": sig, "z": Z, "season": self.season[how], "r_pre": r_pre}
        return out

    # ================================================================== params
    def params(self):
        return {"mu": self.mu, "components_h": list(self.components), "w": self.w, "b": self.b.tolist(), "g": self.g,
                "kappa_intrabar": self.kappa, "theta_intrabar": self.theta,
                "volume_beta[c,log_sigma,|z|,z]": self.vol_beta.tolist(),
                "volume_eta_HAR[c,1h,1d,1w]": self.eta_beta.tolist(), "taker_logit[c,z]": self.tb_beta.tolist(),
                "oi_beta[c,r,|r|,dOI_lag,level_dev]": self.oi_beta.tolist(), "oi_mean_log": self.oi_mean,
                "funding_beta[c,L_lag,L_mean21,R8,R72,R168,R720,dOI24]": self.fund_beta.tolist(),
                "funding_feedback_delta": self.delta, "funding_mean": self.f_mean,
                "fit": self.fit_info, "season_hour_of_week": self.season.round(4).tolist()}

    def save(self, name="structural.json", directory=PARAMS_DIR):
        """Human-readable parameter summary (not enough to generate; see save_state)."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, name), "w") as f:
            json.dump(self.params(), f, indent=1)

    def save_state(self, path=None):
        """Everything generate() needs (parameters + residual tables), so a fitted model can be reloaded
        without refitting. Contains residuals derived from the real data: kept out of git like data/cache."""
        path = path or os.path.join(PARAMS_DIR, STATE_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {k: getattr(self, k) for k in _STATE_SCALARS}
        header["components"] = list(self.components)
        np.savez_compressed(path, header=np.array(json.dumps(header, default=float)),
                            **{k: _compact(getattr(self, k)) for k in _STATE_ARRAYS})
        return path

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(PARAMS_DIR, STATE_FILE)
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            m = cls(components=header.pop("components"), resid_block=header["resid_block"])
            for k in _STATE_ARRAYS:
                setattr(m, k, z[k].astype(float))
        for k, v in header.items():
            setattr(m, k, v)
        return m
