"""Characterise the real hourly data with the stylised facts in facts.py.

    python validation/characterize_real.py

Writes validation/reports/real/:
    fingerprint.csv     every scalar fact: pooled value + value per calendar year + min/max across years
    curves/<group>__<name>.csv
    figures/*.png       ACFs, leverage, Zumbach, aggregation, seasonality, lead-lag, options
    REPORT.md           human-readable summary
The per-year spread is the yardstick for synthetic data: a synthetic fact inside the real inter-year range passes.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facts  # noqa: E402
import panel  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "real")


def save_curves(results, d):
    os.makedirs(d, exist_ok=True)
    for g, res in results.items():
        for name, obj in res["curves"].items():
            obj.to_csv(os.path.join(d, "%s__%s.csv" % (g, name)))


def figures(R, d):
    os.makedirs(d, exist_ok=True)

    def fig(name, draw, title):
        f, ax = plt.subplots(figsize=(7, 4))
        draw(ax)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        f.tight_layout()
        f.savefig(os.path.join(d, name + ".png"), dpi=110)
        plt.close(f)

    dep = R["dependence"]["curves"]
    fig("acf_abs_returns", lambda ax: (ax.plot(dep["acf_abs"].index[1:], dep["acf_abs"].values[1:], label="|r|"),
                                        ax.plot(dep["acf_absds"].index[1:], dep["acf_absds"].values[1:], label="|r| deseasonalised"),
                                        ax.plot(dep["acf_sq"].index[1:], dep["acf_sq"].values[1:], label="r²", alpha=.7),
                                        ax.plot(dep["acf_r"].index[1:], dep["acf_r"].values[1:], label="r"),
                                        ax.set_xscale("log"), ax.set_xlabel("lag (h)"), ax.legend()),
        "Autocorrelation: returns vs absolute / squared returns")
    fig("pacf_abs_returns", lambda ax: (ax.bar(dep["pacf_abs"].index[1:], dep["pacf_abs"].values[1:]), ax.set_xlabel("lag (h)")),
        "PACF of |r|")
    L = R["leverage"]["curves"]["leverage"]
    fig("leverage", lambda ax: (ax.bar(L.index, L.values), ax.axvline(0, color="k", lw=.5),
                                ax.set_xlabel("tau (h): corr(r_t, r²_{t+tau})")), "Leverage effect")
    Z = R["zumbach"]["curves"]["zumbach"]
    fig("zumbach", lambda ax: (ax.plot(Z.index, Z.values), ax.axvline(0, color="k", lw=.5),
                               ax.set_xlabel("tau (h)")), "Zumbach: corr(coarse vol_t, fine vol_{t+tau})")
    D = R["distribution"]["curves"]["by_horizon"]
    fig("aggregational_gaussianity", lambda ax: (ax.plot(D.index, D["exkurt"], "o-", label="excess kurtosis"),
                                                 ax.plot(D.index, 10 * D["ks_normal"], "s-", label="10 × KS distance to N(0,1)"),
                                                 ax.plot(D.index, 10 * (D["qtail"] - 1), "^-", label="10 × (quantile tail weight − 1)"),
                                                 ax.set_xscale("log"), ax.set_xlabel("horizon (h)"), ax.legend()),
        "Aggregational Gaussianity")
    fig("vol_scaling", lambda ax: (ax.plot(D.index, D["scaling"], "o-"), ax.axhline(1, color="k", lw=.5),
                                   ax.set_xscale("log"), ax.set_xlabel("horizon (h)"), ax.set_ylabel("std(h) / (std(1h)·√h)")),
        "Volatility scaling")
    S = R["seasonality"]["curves"]
    fig("seasonality_hour", lambda ax: (S["by_hour"].plot(ax=ax), ax.set_xlabel("UTC hour")), "Intraday profile (relative)")
    fig("seasonality_weekday", lambda ax: (S["by_weekday"].plot(ax=ax, marker="o"), ax.set_xlabel("weekday (0 = Mon)")), "Weekly profile (relative)")
    xc = {}
    if R["volume"]["curves"]:
        xc["|r| → log volume"] = R["volume"]["curves"]["xcorr_absr_logvol"]
        if "xcorr_imbalance_r" in R["volume"]["curves"]:
            xc["taker imbalance → r"] = R["volume"]["curves"]["xcorr_imbalance_r"]
    if R["open_interest"]["curves"]:
        xc["r → ΔOI"] = R["open_interest"]["curves"]["xcorr_r_doi"]
        xc["|r| → ΔOI"] = R["open_interest"]["curves"]["xcorr_absr_doi"]
    if R["options"]["curves"]:
        xc["r → Δ ATM30"] = R["options"]["curves"]["xcorr_r_datm30"]
    if xc:
        fig("lead_lag", lambda ax: ([ax.plot(s.index, s.values, label=k) for k, s in xc.items()],
                                    ax.axvline(0, color="k", lw=.5), ax.set_xlabel("k (h): corr(x_t, y_{t+k})"), ax.legend(fontsize=8)),
            "Lead–lag cross-correlations")


def main():
    p = panel.load_real()
    print("panel: %d hourly bars %s .. %s; columns: %s" % (len(p), p.index[0], p.index[-1], ", ".join(p.columns[:12]) + (" ..." if len(p.columns) > 12 else "")))
    R = facts.characterise(p)
    fp = pd.Series(facts.fingerprint(R), name="pooled")
    years = {}
    for y, g in p.groupby(p.index.year):
        if len(g) < 24 * 60:
            continue
        years[str(y)] = pd.Series(facts.fingerprint(facts.characterise(g)))
        print("  year %s done" % y)
    tab = pd.concat([fp] + [s.rename(k) for k, s in years.items()], axis=1)
    yc = list(years)
    num = tab[yc].apply(pd.to_numeric, errors="coerce")
    tab["year_min"], tab["year_max"] = num.min(axis=1), num.max(axis=1)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "fingerprint.csv"))
    save_curves(R, os.path.join(OUT, "curves"))
    figures(R, os.path.join(OUT, "figures"))
    report(tab, p, os.path.join(OUT, "REPORT.md"))
    print("written to", OUT)


def report(tab, p, path):
    def v(key, fmt="%.3f"):
        if key not in tab.index or pd.isna(tab.at[key, "pooled"]):
            return "n/a"
        lo, hi = tab.at[key, "year_min"], tab.at[key, "year_max"]
        s = fmt % tab.at[key, "pooled"]
        if pd.notna(lo):
            s += " (years: " + fmt % lo + " … " + fmt % hi + ")"
        return s
    L = ["# Real-data stylised facts — BTCUSDT perp (Binance) + Deribit IV", "",
         "Sample: %s → %s, %d hourly bars. Values: pooled (range across calendar years)." % (p.index[0].date(), p.index[-1].date(), len(p)),
         "", "## Returns distribution",
         "- 1h: annualised vol %s, skew %s, excess kurtosis %s" % (v("distribution.r1h_ann_vol"), v("distribution.r1h_skew"), v("distribution.r1h_exkurt", "%.1f")),
         "- robust: quantile skew %s, quantile tail weight vs normal %s, Hill tail index left %s / right %s" % (v("distribution.r1h_qskew"), v("distribution.r1h_qtail"), v("distribution.r1h_hill_left", "%.2f"), v("distribution.r1h_hill_right", "%.2f")),
         "- aggregational Gaussianity: excess kurtosis 1d %s, 1w %s; KS distance to normal 1d %s, 1w %s" % (v("distribution.exkurt_24h", "%.2f"), v("distribution.exkurt_168h", "%.2f"), v("distribution.ks_normal_24h"), v("distribution.ks_normal_168h")),
         "- volatility scaling std(h)/(std1·√h): 1d %s, 1w %s (1 = random walk)" % (v("distribution.scaling_24h"), v("distribution.scaling_168h")),
         "", "## Dependence",
         "- ACF of r at lag 1 %s, max |ACF| lags 1–24 %s, Ljung–Box(24) p-value %s" % (v("dependence.acf_r_1"), v("dependence.acf_r_max_abs_1_24"), v("dependence.ljung_box_24_p")),
         "- ACF of |r|: lag 1 %s, 24 %s, 168 %s, 720 %s" % (v("dependence.acf_abs_1"), v("dependence.acf_abs_24"), v("dependence.acf_abs_168"), v("dependence.acf_abs_720")),
         "- ACF of deseasonalised |r|: lag 1 %s, 24 %s, 168 %s, 720 %s; power-law slope (24–720 h) %s; seasonal bump at 24 h %s" % (v("dependence.acf_absds_1"), v("dependence.acf_absds_24"), v("dependence.acf_absds_168"), v("dependence.acf_absds_720"), v("dependence.acf_absds_powerlaw_slope"), v("dependence.acf_abs_seasonal_bump_24")),
         "- leverage: mean corr(r_t, r²_{t+τ}) τ=1…24 %s vs τ=−24…−1 %s" % (v("leverage.leverage_mean_1_24", "%.4f"), v("leverage.leverage_mean_m24_m1", "%.4f")),
         "- Zumbach time-reversal asymmetry %s" % v("zumbach.zumbach_asymmetry", "%.4f"),
         "- extremes (|r| > q99): inter-arrival CV %s (1 = Poisson), share within 24 h %s vs %s if independent" % (v("extremes.exceed_interarrival_cv", "%.2f"), v("extremes.exceed_within_24h_share", "%.2f"), v("extremes.exceed_within_24h_if_iid", "%.2f")),
         "- max drawdown in 7-day windows: median %s, q90 %s (log)" % (v("extremes.maxdd_7d_median"), v("extremes.maxdd_7d_q90")),
         "", "## Seasonality",
         "- intraday |r| max/min hour ratio %s; weekend |r| relative %s; funding hours (00/08/16) |r| relative %s" % (v("seasonality.hour_absr_max_over_min", "%.2f"), v("seasonality.weekend_absr_rel", "%.2f"), v("seasonality.funding_hours_absr_rel", "%.2f")),
         "", "## Bars and volume",
         "- wicks: upper %s, lower %s, body %s of range; corr(log range, |r|) %s; Parkinson / close-close variance %s" % (v("bars.upper_wick_mean"), v("bars.lower_wick_mean"), v("bars.body_mean"), v("bars.corr_range_absr"), v("bars.parkinson_over_cc_var")),
         "- log volume ACF lag 1 %s, 168 %s; corr(log vol, |r|) %s; corr(taker imbalance, r) %s, with next bar %s" % (v("volume.acf_logvol_1"), v("volume.acf_logvol_168"), v("volume.corr_logvol_absr"), v("volume.corr_imbalance_r"), v("volume.corr_imbalance_next_r", "%.4f")),
         "", "## Open interest",
         "- corr(bar return, ΔlogOI) %s; corr(|r|, ΔlogOI) %s; mean ΔlogOI in worst 1%% bars %s / best 1%% bars %s / all %s" % (v("open_interest.corr_r_doi"), v("open_interest.corr_absr_doi"), v("open_interest.doi_mean_when_r_below_q01", "%.4f"), v("open_interest.doi_mean_when_r_above_q99", "%.4f"), v("open_interest.doi_mean_all", "%.5f")),
         "", "## Funding",
         "- mean %s per 8h (annualised %s), share negative %s; ACF per settlement lag 1 %s, 21 (1 week) %s" % (v("funding.funding_mean", "%.6f"), v("funding.funding_ann_mean"), v("funding.funding_share_negative", "%.2f"), v("funding.acf_funding_1settle"), v("funding.acf_funding_21settle")),
         "- corr with trailing 24h / 168h return %s / %s; with next 24h return %s" % (v("funding.corr_funding_trailing_r_24h"), v("funding.corr_funding_trailing_r_168h"), v("funding.corr_funding_next_r_24h")),
         "", "## Options (Deribit constant-maturity IV)",
         "- 30d ATM mean %s; VRP (ATM30 − next-30d RV) mean %s, positive %s of the time" % (v("options.atm30_mean"), v("options.vrp30_mean"), v("options.vrp30_share_positive", "%.2f")),
         "- spot–vol: corr(r, ΔATM30) %s, beta %s; corr(r, ΔRR25_30d) %s; corr(r, ΔBF25_30d) %s" % (v("options.corr_atm_30d"), v("options.beta_atm_30d"), v("options.corr_rr25_30d"), v("options.corr_bf25_30d")),
         "- RR25_30d mean %s, positive %s of time, corr with trailing 7d return %s" % (v("options.rr25_30d_mean"), v("options.rr25_30d_share_positive", "%.2f"), v("options.corr_rr25_30d_trailing_r_7d")),
         "- term slope ATM7 − ATM90 mean %s, inverted %s of time, corr with trailing 24h RV %s; ATM30 half-life %s days" % (v("options.term_slope_7_90_mean"), v("options.term_slope_share_inverted", "%.2f"), v("options.corr_term_slope_trailing_rv_24h"), v("options.atm30_halflife_days", "%.1f")),
         "", "Full table: `fingerprint.csv`; curves: `curves/`; plots: `figures/`."]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
