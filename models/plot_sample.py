"""Plot the real history next to synthetic paths of the same length.

    python models/plot_sample.py --model structural --seeds 1 2
Writes validation/reports/<model>/figures/sample_paths.png
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
sys.path.insert(0, os.path.join(ROOT, "validation"))
import models  # noqa: E402
import panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="structural")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    real = panel.load_real()
    m = models.get(a.model).fit(real)
    paths = [("real", real)] + [("synthetic seed %d" % s, m.generate(len(real), seed=s, start=str(real.index[0].date())))
                                for s in a.seeds]
    rows = [("close", "price (log)", lambda p: p["close"], True),
            ("vol", "volume (BTC, 1-day mean)", lambda p: p["volume"].rolling(24).mean(), True),
            ("oi", "open interest (BTC)", lambda p: p["oi"], False),
            ("funding", "funding per 8 h (%)", lambda p: 100 * p["funding"], False),
            ("rv", "realised vol, 7-day (annualised)", lambda p: np.log(p["close"]).diff().rolling(168).std() * np.sqrt(8766), False)]
    f, axs = plt.subplots(len(rows), len(paths), figsize=(5 * len(paths), 2.4 * len(rows)), sharex="col")
    for j, (name, p) in enumerate(paths):
        for i, (_, label, fn, logy) in enumerate(rows):
            ax = axs[i, j]
            ax.plot(p.index, fn(p), lw=.6, color="k" if j == 0 else "C%d" % j)
            if logy:
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            if i == 0:
                ax.set_title(name)
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=7)
    f.tight_layout()
    out = os.path.join(ROOT, "validation", "reports", a.model, "figures", "sample_paths.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
