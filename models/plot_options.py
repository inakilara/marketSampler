"""Real vs synthetic IV surface paths, plus sanity rates (negative butterflies, calendar-variance violations).

    python models/plot_options.py [--seed 1]
Writes validation/reports/structural+options/figures/surface_paths.png and prints the sanity table.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
import models  # noqa: E402
import panel  # noqa: E402
from models.options import TENORS  # noqa: E402


def sanity(p):
    out = {}
    for q in ("bf25", "bf10"):
        v = p[["%s_%dd" % (q, T) for T in TENORS]].values
        out["%s < 0" % q] = np.nanmean(v[np.isfinite(v)] < 0)
    # calendar: total ATM variance w = atm^2 T must not decrease with T
    W = np.column_stack([p["atm_%dd" % T] ** 2 * T for T in TENORS])
    ok = np.all(np.isfinite(W), axis=1)
    out["calendar violation (any tenor pair)"] = np.mean(np.any(np.diff(W[ok], axis=1) < 0, axis=1))
    # 10-delta wings should sit outside the 25-delta ones: BF10 >= BF25
    d = np.column_stack([p["bf10_%dd" % T] - p["bf25_%dd" % T] for T in TENORS])
    out["bf10 < bf25"] = np.nanmean(d[np.isfinite(d)] < 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    real = panel.load_real()
    m = models.load("structural+options")
    syn = m.generate(len(real), seed=a.seed, start=str(real.index[0].date()))
    print(pd.DataFrame({"real": sanity(real), "synthetic": sanity(syn)}).round(4).to_string())
    rows = [("ATM 7d / 30d / 90d", ["atm_7d", "atm_30d", "atm_90d"]),
            ("RR25 7d / 30d / 90d", ["rr25_7d", "rr25_30d", "rr25_90d"]),
            ("BF25 30d, BF10 30d", ["bf25_30d", "bf10_30d"]),
            ("term slope ATM7 - ATM90", None)]
    f, axs = plt.subplots(len(rows) + 1, 2, figsize=(13, 2.4 * (len(rows) + 1)), sharex="col")
    for j, (name, p) in enumerate((("real", real), ("synthetic seed %d" % a.seed, syn))):
        axs[0, j].plot(p.index, p["close"], "k", lw=.6)
        axs[0, j].set_yscale("log")
        axs[0, j].set_title(name)
        axs[0, j].set_ylabel("price", fontsize=8)
        for i, (label, cols) in enumerate(rows, start=1):
            ax = axs[i, j]
            d = p[cols].resample("D").mean() if cols else (p["atm_7d"] - p["atm_90d"]).resample("D").mean().to_frame("slope")
            for k, c in enumerate(d.columns):
                ax.plot(d.index, 100 * d[c], lw=.7, label=c, color="C%d" % k)
            if cols is None:
                ax.axhline(0, color="k", lw=.5)
            if j == 0:
                ax.set_ylabel(label + " (vol pts)", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=7)
    f.tight_layout()
    out = os.path.join(ROOT, "validation", "reports", "structural+options", "figures", "surface_paths.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f.savefig(out, dpi=105)
    print(out)


if __name__ == "__main__":
    main()
