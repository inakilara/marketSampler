"""Plot the scenario controls of the structural model against a baseline with the same seed.

    python models/plot_scenarios.py [--seed 1] [--days 180]
Writes validation/reports/structural/figures/scenarios.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402


def scenarios(n):
    stress = np.ones(n)
    stress[n // 3:n // 3 + 336] = 2.5
    regime = np.r_[np.full(n // 3, 0.35), np.full(n - n // 3, 0.9)]
    return [("baseline", {}),
            ("calm start (start_vol='calm')", {"start_vol": "calm"}),
            ("stressed start (start_vol='stressed')", {"start_vol": "stressed"}),
            ("regime 35% -> 90% (vol_level path)", {"vol_level": regime}),
            ("2-week stress x2.5 (vol_mult)", {"vol_mult": stress}),
            ("-15% crash at day 60 (jumps)", {"jumps": {60 * 24: -0.15}}),
            ("bull drift +80%/y", {"drift": 0.8}),
            ("funding stress +0.03%/8h", {"funding_shift": 3e-4})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)  # seed 3 starts from an average-vol history
    ap.add_argument("--days", type=int, default=180)
    a = ap.parse_args()
    m = models.load("structural")
    n = 24 * a.days
    rows = [("price", lambda p: p["close"], True),
            ("realised vol 7d (ann.)", lambda p: np.log(p["close"]).diff().rolling(168).std() * np.sqrt(8766), False),
            ("volume 1d mean", lambda p: p["volume"].rolling(24).mean(), True),
            ("open interest", lambda p: p["oi"], False),
            ("funding %/8h", lambda p: 100 * p["funding"], False)]
    sc = scenarios(n)
    f, axs = plt.subplots(len(rows), len(sc), figsize=(3.2 * len(sc), 2.1 * len(rows)), sharex=True)
    base = m.generate(n, seed=a.seed, start="2030-01-07")
    for j, (title, kw) in enumerate(sc):
        p = m.generate(n, seed=a.seed, start="2030-01-07", **kw)
        for i, (label, fn, logy) in enumerate(rows):
            ax = axs[i, j]
            if j:
                ax.plot(base.index, fn(base), lw=.6, color="0.7")
            ax.plot(p.index, fn(p), lw=.7, color="k" if j == 0 else "C%d" % (j % 10))
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=.3)
    for ax in axs[-1]:
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
    f.suptitle("structural model scenarios (grey: baseline, same seed)", fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(ROOT, "validation", "reports", "structural", "figures", "scenarios.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
