"""Score a synthetic generator against the real stylised-facts fingerprint.

    python validation/compare.py --model bootstrap
    python validation/compare.py --model structural --paths 3 --groups distribution,dependence,leverage,zumbach,seasonality,extremes

Method
  * The model is fitted on the real panel and generates `--paths` paths as long as the real sample.
  * Each synthetic path is cut into 1-year chunks; every chunk is characterised exactly like a real calendar year.
  * For every scalar fact: PASS if the median over synthetic chunks lies inside the real [min, max] across
    years, NEAR if within half a range-width outside, FAIL otherwise. Facts measured only on the pooled real
    sample are compared pooled vs pooled and reported as INFO.
  * Key curves (|r| ACF, leverage, Zumbach, intraday profile, aggregation) are overlaid in figures.
Writes validation/reports/<model>/: score.csv, SUMMARY.md, figures/.
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import facts  # noqa: E402
import panel as panel_mod  # noqa: E402
import models  # noqa: E402

CHUNK = 8760
NO_OPTIONS = [g for g in facts.ALL if g != "options"]


def real_fingerprint():
    p = os.path.join(HERE, "reports", "real", "fingerprint.csv")
    if not os.path.exists(p):
        sys.exit("run validation/characterize_real.py first")
    return pd.read_csv(p, index_col=0)


def characterise_paths(paths, groups):
    chunk_fp, pooled_fp, pooled_res = [], [], []
    for k, sp in enumerate(paths):
        R = facts.characterise(sp, groups)
        pooled_res.append(R)
        pooled_fp.append(pd.Series(facts.fingerprint(R)))
        for i in range(0, len(sp) - CHUNK // 2, CHUNK):
            c = sp.iloc[i:i + CHUNK]
            if len(c) >= 24 * 60:
                chunk_fp.append(pd.Series(facts.fingerprint(facts.characterise(c, groups))))
    return pd.DataFrame(chunk_fp), pd.DataFrame(pooled_fp), pooled_res


def score(real, chunks, pooled):
    rows = []
    for key in real.index:
        if key not in chunks.columns and key not in pooled.columns:
            continue
        rp = pd.to_numeric(real.at[key, "pooled"], errors="coerce")
        lo = pd.to_numeric(real.at[key, "year_min"], errors="coerce")
        hi = pd.to_numeric(real.at[key, "year_max"], errors="coerce")
        sc = pd.to_numeric(chunks.get(key), errors="coerce") if key in chunks.columns else pd.Series(dtype=float)
        spool = pd.to_numeric(pooled.get(key), errors="coerce").median() if key in pooled.columns else np.nan
        med = sc.median() if len(sc) else np.nan
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(med):
            w = max(hi - lo, 1e-12)
            if lo <= med <= hi:
                status = "PASS"
            elif lo - 0.5 * w <= med <= hi + 0.5 * w:
                status = "NEAR"
            else:
                status = "FAIL"
            inside = float(((sc >= lo) & (sc <= hi)).mean())
        else:
            status, inside = ("INFO" if np.isfinite(spool) else "N/A"), np.nan
        rows.append({"fact": key, "group": key.split(".")[0], "status": status, "real_pooled": rp,
                     "real_year_min": lo, "real_year_max": hi, "syn_chunk_median": med,
                     "syn_chunk_inside_share": inside, "syn_pooled": spool})
    return pd.DataFrame(rows).set_index("fact")


def overlay_figures(real_dir, syn_res, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def real_curve(name):
        p = os.path.join(real_dir, "curves", name + ".csv")
        return pd.read_csv(p, index_col=0) if os.path.exists(p) else None

    def syn_curves(group, name):
        cs = [R[group]["curves"].get(name) for R in syn_res if group in R and name in R[group]["curves"]]
        return [c for c in cs if c is not None]

    specs = [("dependence", "acf_absds", "ACF of deseasonalised |r|", True),
             ("dependence", "acf_r", "ACF of r", False),
             ("leverage", "leverage", "Leverage corr(r_t, r²_{t+τ})", False),
             ("zumbach", "zumbach", "Zumbach corr(coarse_t, fine_{t+τ})", False),
             ("volume", "acf_logvol", "ACF of log volume", True),
             ("open_interest", "xcorr_r_doi", "corr(r_t, ΔlogOI_{t+k})", False),
             ("funding", "acf_funding", "ACF of funding (per settlement)", False)]
    for group, name, title, logx in specs:
        rc = real_curve("%s__%s" % (group, name))
        sc = syn_curves(group, name)
        if rc is None or not sc:
            continue
        f, ax = plt.subplots(figsize=(7, 4))
        x = rc.index.values
        start = 1 if name.startswith("acf") else 0
        ax.plot(x[start:], rc.iloc[start:, 0].values, "k-", lw=2, label="real")
        for i, s in enumerate(sc):
            s = s if isinstance(s, pd.Series) else s.iloc[:, 0]
            ax.plot(s.index.values[start:], s.values[start:], alpha=.6, label="synthetic" if i == 0 else None)
        if logx:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(alpha=.3)
        ax.legend()
        f.tight_layout()
        f.savefig(os.path.join(out_dir, "%s__%s.png" % (group, name)), dpi=110)
        plt.close(f)
    # aggregation + intraday profile
    rd = real_curve("distribution__by_horizon")
    sd = syn_curves("distribution", "by_horizon")
    if rd is not None and sd:
        f, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, t in ((axs[0], "exkurt", "excess kurtosis vs horizon"), (axs[1], "qtail", "quantile tail weight vs horizon")):
            ax.plot(rd.index, rd[col], "k-o", lw=2, label="real")
            for i, s in enumerate(sd):
                ax.plot(s.index, s[col], "-o", alpha=.6, label="synthetic" if i == 0 else None)
            ax.set_xscale("log"); ax.set_title(t); ax.grid(alpha=.3); ax.legend()
        f.tight_layout(); f.savefig(os.path.join(out_dir, "distribution__aggregation.png"), dpi=110); plt.close(f)
    rh = real_curve("seasonality__by_hour")
    sh = syn_curves("seasonality", "by_hour")
    if rh is not None and sh:
        f, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rh.index, rh["absr"], "k-", lw=2, label="real |r|")
        for i, s in enumerate(sh):
            ax.plot(s.index, s["absr"], alpha=.6, label="synthetic |r|" if i == 0 else None)
        ax.set_title("intraday |r| profile"); ax.grid(alpha=.3); ax.legend()
        f.tight_layout(); f.savefig(os.path.join(out_dir, "seasonality__by_hour.png"), dpi=110); plt.close(f)


def write_summary(tab, model, n_paths, n_hours, path, secs):
    L = ["# %s vs real fingerprint" % model, "",
         "%d synthetic paths × %d hours, cut into 1-year chunks; status = synthetic chunk median vs real per-year [min, max]." % (n_paths, n_hours),
         "Fit + generation + characterisation: %.0f s." % secs, ""]
    graded = tab[tab["status"].isin(["PASS", "NEAR", "FAIL"])]
    L.append("**Overall: %d PASS, %d NEAR, %d FAIL out of %d graded facts.**" % (
        (graded.status == "PASS").sum(), (graded.status == "NEAR").sum(), (graded.status == "FAIL").sum(), len(graded)))
    L += ["", "| group | PASS | NEAR | FAIL |", "|---|---|---|---|"]
    for g, d in graded.groupby("group"):
        L.append("| %s | %d | %d | %d |" % (g, (d.status == "PASS").sum(), (d.status == "NEAR").sum(), (d.status == "FAIL").sum()))
    bad = graded[graded.status != "PASS"]
    if len(bad):
        L += ["", "## Not passing", "", "| fact | status | real years [min, max] | synthetic chunk median |", "|---|---|---|---|"]
        for k, r in bad.iterrows():
            L.append("| %s | %s | [%.4g, %.4g] | %.4g |" % (k, r.status, r.real_year_min, r.real_year_max, r.syn_chunk_median))
    L += ["", "Full table: `score.csv`; overlays: `figures/`."]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def run(model_name, n_paths=3, groups=None, seed0=0, refit=False):
    t0 = time.time()
    real_p = panel_mod.load_real()
    m = models.load(model_name, refit=refit, panel=real_p)
    n = len(real_p)
    paths = [m.generate(n, seed=seed0 + k) for k in range(n_paths)]
    groups = groups or NO_OPTIONS
    chunks, pooled, res = characterise_paths(paths, groups)
    real = real_fingerprint()
    real = real[real.index.map(lambda k: k.split(".")[0] in groups)]
    tab = score(real, chunks, pooled)
    out = os.path.join(HERE, "reports", model_name)
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, "score.csv"))
    overlay_figures(os.path.join(HERE, "reports", "real"), res, os.path.join(out, "figures"))
    write_summary(tab, model_name, n_paths, n, os.path.join(out, "SUMMARY.md"), time.time() - t0)
    return tab, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--paths", type=int, default=3)
    ap.add_argument("--groups", help="comma-separated fact groups (default: all except options)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true", help="refit instead of loading the saved model state")
    a = ap.parse_args()
    tab, _ = run(a.model, a.paths, a.groups.split(",") if a.groups else None, a.seed, a.refit)
    g = tab[tab.status.isin(["PASS", "NEAR", "FAIL"])]
    print(g.groupby("group").status.value_counts().unstack(fill_value=0).to_string())
    print("written to", os.path.join(HERE, "reports", a.model))
