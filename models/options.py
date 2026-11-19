"""Options layer: hourly constant-maturity IV surface (ATM, 25Δ/10Δ risk reversal and butterfly at 7…180 days)
driven by a price path. It needs only the path (close, open, funding), so it attaches to any generator
(or to real history).

Modelled series y: log ATM, RR/ATM and BF/ATM at the 30-day anchor; other tenors as spreads to the anchor
(separate states per tenor let tenors drift apart for weeks). RR and BF scale with the vol level.
Error-correction form, per series:

  y_t = lr_a + beta' F_t + m_t + u_t + n_t
  level     F_t = state of the path at the open of bar t: log realised vol over 1 d / 1 w / 30 d / 90 d, trailing
            returns over 1 w / 30 d, |1 w return|, funding. lr_a + beta'F is the long-run IV given the path (variance
            risk premium, skew follows price, short tenors track recent realised vol...). Static (not an AR
            target): as an AR target the term slope loaded on 24 h RV at 1/4 of the real strength.
  slow      m_{t+1} = phi_h m_t + k a + g1 rb_t + g2 |rb_t| + sqrt(q) eta_t: persistent deviation, with the
            persistent spot-vol / skew response to the bar return rb_t; eta_t = joint shock vector across all 30
            series (resampled real residuals: tenors and quantities co-move).
  fast      u_{t+1} = psi u_t + f1 rb_t + f2 (|rb_t| - E|rb|): the same-hour overshoot. Real 30d RR25 moves 0.80 x the
            bar return within the hour and keeps ~0.4 after 2-4 h; large moves lift short-dated ATM for a few hours.
  noise     n_t: measurement noise of the reconstruction (10-80 % of hourly changes; fitting hourly dynamics
            directly on the noisy surface gives half-lives of hours instead of weeks).

Fit: (0) hourly OLS of y on [1, F]; x = y - level. (1) daily AR on 00:00 snapshots,
x_{d+1} = a + phi_d x_d + g1 sum(rb over day) + g2 sum|rb| + e, with an errors-in-variables correction for the
noise in x_d; converted to hourly (phi_h = phi_d^(1/24), k = (1-phi_h)/(1-phi_d)). (2) hourly impulse response of
x to rb and |rb| over 0-24 h: f = same-hour response minus the persistent part, psi = decay fitted on the curve.
(3) q and the noise variance from the daily residual variance (24 q + 2 s_n^2) and the hourly change variance.
"""
import json
import os

import numpy as np
import pandas as pd

from models.common import stationary_bootstrap_indices
from models.structural import _compact

TENORS = (7, 14, 30, 60, 90, 180)
QUANTS = ("atm", "rr25", "bf25", "rr10", "bf10")
IMPULSE_H = 24
ACF_LAGS = 60  # days
PSI_GRID = np.r_[0.0, np.linspace(0.05, 0.95, 19)]
FEATURES = ("lrv24", "lrv168", "lrv720", "lrv2160", "R168", "R720", "aR168", "funding")
HOURS_YEAR = 8766
PARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params")
STATE_FILE = "options_state.npz"
SERIES = ["%s_%dd" % (q, T) for T in TENORS for q in QUANTS]
ANCHOR = 30  # modelled directly; other tenors as spreads to it


def path_features(r, funding):
    """Features at the open of bar t from returns up to bar t-1 (arrays aligned with r)."""
    s = pd.Series(r)
    out = {}
    for h in (24, 168, 720, 2160):
        rv = (s * s).rolling(h, min_periods=max(12, h // 4)).mean() * HOURS_YEAR
        out["lrv%d" % h] = 0.5 * np.log(rv.clip(lower=1e-8)).shift(1)
    c = s.cumsum()
    for h in (168, 720):
        out["R%d" % h] = (c.shift(1) - c.shift(1 + h)).fillna(c.shift(1))  # shorter history at the start
    out["aR168"] = out["R168"].abs()
    out["funding"] = pd.Series(funding).ffill().bfill()
    return pd.DataFrame(out)[list(FEATURES)].values


def observed_targets(p):
    """Real surface transformed to the modelled series: log ATM, RR/ATM, BF/ATM at the 30-day anchor, and for
    the other tenors their spread to the anchor (term-structure spreads revert faster and load directly on recent
    realised vol; separate slow states per tenor let tenors drift apart for weeks: corr(ATM7, ATM90) 0.71 vs 0.86)."""
    Y = {}
    for T in TENORS:
        atm = p["atm_%dd" % T]
        Y["atm_%dd" % T] = np.log(atm)
        for q in QUANTS[1:]:
            Y["%s_%dd" % (q, T)] = p["%s_%dd" % (q, T)] / atm
    Y = pd.DataFrame(Y)[SERIES]
    for T in TENORS:
        if T != ANCHOR:
            for q in QUANTS:
                Y["%s_%dd" % (q, T)] -= Y["%s_%dd" % (q, ANCHOR)]
    return Y


def surface_from_series(obs, index):
    """Inverse of observed_targets: modelled series (N, S) -> surface columns."""
    o = pd.DataFrame(obs, index=index, columns=SERIES)
    for T in TENORS:
        if T != ANCHOR:
            for q in QUANTS:
                o["%s_%dd" % (q, T)] += o["%s_%dd" % (q, ANCHOR)]
    out = pd.DataFrame(index=index)
    for T in TENORS:
        atm = np.exp(o["atm_%dd" % T])
        out["atm_%dd" % T] = atm
        for q in QUANTS[1:]:
            out["%s_%dd" % (q, T)] = o["%s_%dd" % (q, T)] * atm
    return out


def _path_returns(path):
    """Close-to-close and bar (open-to-close) log returns; missing hours count as flat."""
    r = np.log(path["close"]).diff().values
    r[0] = np.log(path["close"].iloc[0] / path["open"].iloc[0])
    rb = np.log(path["close"] / path["open"]).values
    return np.nan_to_num(r), np.nan_to_num(rb)


def impulse_response(y, rb, horizon=IMPULSE_H):
    """Cumulative response of y (open of bar t -> open of bar t+1+j) to rb_t and |rb_t|, j = 0..horizon."""
    ab = np.abs(rb) - np.nanmean(np.abs(rb))
    d = np.r_[np.diff(y), np.nan]  # change over bar t
    c1, c2 = [], []
    for j in range(horizon + 1):
        dj = np.r_[d[j:], np.full(j, np.nan)]
        ok = np.isfinite(dj) & np.isfinite(rb)
        X = np.column_stack([np.ones(ok.sum()), rb[ok], ab[ok]])
        b = np.linalg.lstsq(X, dj[ok], rcond=None)[0]
        c1.append(b[1])
        c2.append(b[2])
    return np.cumsum(c1), np.cumsum(c2)


def fit_two_speed(lags, ac):
    """ACF(k) = c (w phi1^k + (1-w) phi2^k): slow + medium AR(1) components plus a nugget 1-c (noise and
    intraday transients). Returns daily phi1 >= phi2, w, c."""
    from scipy.optimize import minimize
    ok = np.isfinite(ac)
    lags, ac = lags[ok], ac[ok]

    def loss(th):
        p1, p2, w, c = th
        return np.sum((c * (w * p1 ** lags + (1 - w) * p2 ** lags) - ac) ** 2)

    best = None
    for p1 in (0.97, 0.99, 0.995 - 1e-3):
        for p2 in (0.5, 0.8, 0.9):
            res = minimize(loss, [p1, p2, 0.5, 0.95], method="L-BFGS-B",
                           # slow half-life capped at ~6 months: uncapped, several spreads fit a near-unit root
                           # (structural drift of the real sample) that would let synthetic surfaces wander for years
                           bounds=[(0.9, 0.996), (0.2, 0.97), (0.0, 1.0), (0.3, 1.0)])
            if best is None or res.fun < best.fun:
                best = res
    p1, p2, w, c = best.x
    if p2 > p1:
        p1, p2, w = p2, p1, 1 - w
    return p1, p2, w, c


class OptionsLayer:
    name = "options"

    # ================================================================== fit
    def fit(self, p):
        r, rb = _path_returns(p)
        F = path_features(r, p["funding"].values)
        Y = observed_targets(p).values
        n, S = Y.shape
        t0 = np.flatnonzero((p.index.hour == 0) & (np.arange(n) >= 720) & (np.arange(n) + 24 < n))
        crb = np.r_[0.0, np.cumsum(rb)]
        cab = np.r_[0.0, np.cumsum(np.abs(rb))]
        R24 = crb[t0 + 24] - crb[t0]
        A24 = cab[t0 + 24] - cab[t0]
        dY = np.diff(Y, axis=0)
        n2_lag1 = np.clip(-np.nanmean(dY[1:] * dY[:-1], axis=0), 0, None)  # white-noise part of the observation error
        self.abs_rb_mean = float(np.mean(np.abs(rb)))
        z = np.zeros(S)
        (self.g1, self.g2, self.f1, self.f2, self.psi, self.phi1, self.phi2, self.q1, self.q2,
         self.sn2) = (z.copy() for _ in range(10))
        self.beta = np.zeros((S, len(FEATURES)))
        resid = np.full((len(t0), S), np.nan)
        info = {}
        self.lr_a = np.zeros(S)
        for j, name in enumerate(SERIES):
            # ---- (0) long-run level: static regression on the path state (hourly). Putting the state in the level
            # (instead of only as a slow AR target) lets short tenors track recent realised vol within the day:
            # with an AR-target form the term slope loaded on 24 h RV at 1/4 of the real strength.
            yy = Y[:, j]
            okl = np.isfinite(yy) & np.all(np.isfinite(F), axis=1)
            bl = np.linalg.lstsq(np.column_stack([np.ones(okl.sum()), F[okl]]), yy[okl], rcond=None)[0]
            self.lr_a[j], beta = bl[0], bl[1:]
            x = yy - (bl[0] + F @ beta)  # deviation from the long-run level
            # ---- (1) persistent spot-vol / skew response: daily regression of the deviation on the day's returns
            x1, x0 = x[t0 + 24], x[t0]
            X = np.column_stack([np.ones(len(t0)), x0, R24, A24])
            ok = np.isfinite(x1) & np.all(np.isfinite(X), axis=1)
            Xo, yo = X[ok], x1[ok]
            # errors-in-variables: the lagged observation x0 carries noise of variance n2 -> remove it from X'X
            XtX = Xo.T @ Xo / len(yo)
            XtX[1, 1] -= n2_lag1[j]
            b = np.linalg.solve(XtX, Xo.T @ yo / len(yo))
            resid[ok, j] = yo - Xo @ b
            g1, g2 = b[2], b[3]
            # ---- (2) fast part: same-hour response in excess of the persistent one, decaying at rate psi
            C1, C2 = impulse_response(x, rb)
            f1, f2 = C1[0] - g1, C2[0] - g2
            k = np.arange(IMPULSE_H + 1)
            loss = [np.sum((C1 - g1 - f1 * s ** k) ** 2) / max(f1 * f1, 1e-12)
                    + np.sum((C2 - g2 - f2 * s ** k) ** 2) / max(f2 * f2, 1e-12) for s in PSI_GRID]
            psi = PSI_GRID[int(np.argmin(loss))]
            # ---- (3) two-speed deviation (slow: months, medium: days) fitted on the daily ACF of x up to 60 days;
            # a single AR fitted on daily changes only sees the medium part (real RR/BF deviations persist for
            # months: ACF at 30 d ~0.6, single-AR model ~0)
            xd = pd.Series(x[t0])
            Vx = float(xd.var())
            lags = np.arange(1, ACF_LAGS + 1)
            ac = np.array([xd.autocorr(int(L)) for L in lags])
            phi1, phi2, w, c = fit_two_speed(lags, ac)
            V1, V2 = c * w * Vx, c * (1 - w) * Vx
            ph1, ph2 = phi1 ** (1 / 24), phi2 ** (1 / 24)
            imp_h = np.var(g1 * rb + g2 * np.abs(rb))
            q1 = V1 * (1 - ph1 ** 2)
            q2 = max(V2 * (1 - ph2 ** 2) - imp_h, 0.1 * V2 * (1 - ph2 ** 2))
            # measurement noise from the hourly change variance: Var(dx) = impulses + fast decay + q1 + q2 + 2 sn2
            # (the daily-ACF nugget also holds intraday transients and overstates it several-fold, which drowned
            # the hourly spot-skew correlation)
            imp_all = np.var((g1 + f1) * rb + (g2 + f2) * np.abs(rb))
            imp_fast = np.var(f1 * rb + f2 * np.abs(rb)) * (1 - psi) / (1 + psi)
            # (observed change = level change + deviation changes + noise; the level moves hourly with the RV features)
            dlev = np.diff(F @ beta)
            sn2 = max((np.nanvar(np.diff(yy)) - np.nanvar(dlev) - imp_all - imp_fast - q1 - q2) / 2, 0.0)
            (self.beta[j], self.g1[j], self.g2[j], self.f1[j], self.f2[j], self.psi[j], self.phi1[j], self.phi2[j],
             self.q1[j], self.q2[j], self.sn2[j]) = beta, g1, g2, f1, f2, psi, ph1, ph2, q1, q2, sn2
            hl = lambda ph: round(float(np.log(.5) / np.log(ph)), 1)  # noqa: E731
            info[name] = {"halflife_slow_d": hl(phi1), "halflife_medium_d": hl(phi2), "slow_share": round(float(w), 2),
                          "noise_share_daily": round(float(1 - c), 2),
                          "g1": round(float(g1), 3), "f1": round(float(f1), 3), "g2": round(float(g2), 3),
                          "f2": round(float(f2), 3), "psi": round(float(psi), 2), "n_days": int(ok.sum())}

        # joint shocks, re-centred and re-scaled on rows complete across series (a non-zero mean added every hour to
        # a persistent process would shift the IV level)
        def standardise_complete(A):
            A = A[np.all(np.isfinite(A), axis=1)]
            return (A - A.mean(axis=0)) / A.std(axis=0)
        # slow component: daily residual rows; medium component and measurement noise: hourly change rows (they
        # carry the same-hour cross-correlation, e.g. dATM vs dBF -0.5)
        self.eta = standardise_complete(resid)
        self.noise = standardise_complete(dY)
        self.fit_info = {"series": info, "n_eta_rows": int(len(self.eta)), "n_noise_rows": int(len(self.noise))}
        return self

    # ================================================================== generate
    def generate(self, path, seed=0, r_pre=None, atm_shift=0.0, rr_shift=0.0):
        """Surface for a price path (DataFrame with open, close, funding). r_pre: returns before the path (e.g.
        StructuralModel.last_state["r_pre"]) used as burn-in for the realised-vol features and the IV state;
        without it the first weeks use shorter windows. atm_shift: shift of the long-run log ATM (0.1 = +10 %);
        rr_shift: shift of the long-run RR/ATM (e.g. -0.1 = put skew deeper by 10 % of ATM)."""
        rng = np.random.default_rng(seed)
        r, rb = _path_returns(path)
        fund = path["funding"].values
        if r_pre is not None and len(r_pre):
            r = np.r_[r_pre, r]
            rb = np.r_[r_pre, rb]
            fund = np.r_[np.full(len(r_pre), fund[0]), fund]
        n0 = len(r) - len(path)
        F = path_features(r, fund)
        F = pd.DataFrame(F).bfill().fillna(0.0).values  # very first hours: first available feature values
        N, S = len(r), len(SERIES)
        lr_a = self.lr_a.copy()
        is_anchor = np.array([s.endswith("_%dd" % ANCHOR) for s in SERIES])  # other tenors are spreads to it
        lr_a[is_anchor & np.array([s.startswith("atm") for s in SERIES])] += atm_shift
        lr_a[is_anchor & np.array([s.startswith("rr") for s in SERIES])] += rr_shift
        level = lr_a[None, :] + F @ self.beta.T  # long-run level given the path state (N, S)
        e1 = self.eta[rng.integers(0, len(self.eta), N)] * np.sqrt(self.q1)
        # medium shocks and measurement noise share the same real hour: ATM's hourly move is mostly its medium shock,
        # BF's mostly noise; separate draws would lose their same-hour correlation (real dATM vs dBF -0.5)
        hrow = self.noise[rng.integers(0, len(self.noise), N)]
        e2 = hrow * np.sqrt(self.q2)
        ab = np.abs(rb) - self.abs_rb_mean
        s = np.zeros((N + 1, S))  # slow deviation (months)
        v = np.zeros((N + 1, S))  # medium deviation (days) with the persistent return response
        u = np.zeros((N + 1, S))  # same-hour overshoot
        # the slow part starts from its stationary distribution (the pre-history is shorter than its memory)
        s[0] = self.eta[rng.integers(0, len(self.eta))] * np.sqrt(self.q1 / (1 - self.phi1 ** 2))
        for t in range(N):
            s[t + 1] = self.phi1 * s[t] + e1[t]
            v[t + 1] = self.phi2 * v[t] + self.g1 * rb[t] + self.g2 * ab[t] + e2[t]
            u[t + 1] = self.psi * u[t] + self.f1 * rb[t] + self.f2 * ab[t]
        # shock e2[t] enters the state at t+1, so the noise of observation t+1 must come from the same row
        noise = np.vstack([hrow[-1:], hrow[:-1]]) * np.sqrt(self.sn2)
        obs = (level + s[:N] + v[:N] + u[:N] + noise)[n0:]
        self.last_latent = (s + v)[n0:N]
        return surface_from_series(obs, path.index)

    def attach(self, path, seed=0, r_pre=None, **kw):
        """path + surface columns."""
        return pd.concat([path, self.generate(path, seed=seed, r_pre=r_pre, **kw)], axis=1)

    # ================================================================== persistence
    _ARRAYS = ("lr_a", "beta", "g1", "g2", "f1", "f2", "psi", "phi1", "phi2", "q1", "q2", "sn2", "eta", "noise")

    def save_state(self, path=None):
        path = path or os.path.join(PARAMS_DIR, STATE_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, header=np.array(json.dumps({"fit_info": self.fit_info, "abs_rb_mean": self.abs_rb_mean})),
                            **{k: _compact(getattr(self, k)) for k in self._ARRAYS})
        with open(os.path.join(os.path.dirname(path), "options.json"), "w") as f:
            json.dump({"series": SERIES, "features": list(FEATURES), "fit": self.fit_info,
                       "beta": {s: dict(zip(FEATURES, np.round(self.beta[j], 5).tolist())) for j, s in enumerate(SERIES)}},
                      f, indent=1)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(PARAMS_DIR, STATE_FILE)
        m = cls()
        with np.load(path, allow_pickle=False) as z:
            for k in cls._ARRAYS:
                setattr(m, k, z[k].astype(float))
            h = json.loads(str(z["header"]))
        m.fit_info, m.abs_rb_mean = h["fit_info"], h["abs_rb_mean"]
        return m
