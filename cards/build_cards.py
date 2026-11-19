"""Build pre-fitted model cards: the structural generator + options layer fitted on the whole history and on each
calendar year separately.

    python cards/build_cards.py                 # all cards
    python cards/build_cards.py --only 2022 all

Each card is a folder cards/<name>/:
    structural_state.npz, options_state.npz   fitted states (load with models.load_card(name))
    structural.json, options.json             readable parameters
    card.json                                 period, data window, fit summary, real vs synthetic check
cards/README.md is rewritten with the index table.

Needs the data in data/cache (data/fetch_binance.py, data/fetch_deribit.py, data/build_iv_surface.py); using a
card does not.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
import panel as panel_mod  # noqa: E402
import models  # noqa: E402
from models.options import OptionsLayer  # noqa: E402
from models.structural import StructuralModel  # noqa: E402

warnings.filterwarnings("ignore")
HOURS_YEAR = 8766
N_CHECK_PATHS = 3


def summary(p):
    """A few headline statistics of a panel (real or synthetic) for the card's sanity table."""
    r = np.log(p["close"]).diff().dropna()
    f8 = p["funding"][p.index.hour % 8 == 0].dropna()
    out = {"ann_vol": r.std() * np.sqrt(HOURS_YEAR),
           "hourly_exkurt": r.kurt(),
           "acf_absr_1h": r.abs().autocorr(1),
           "log_return_per_year": r.mean() * HOURS_YEAR,
           "volume_median": p["volume"].median(),
           "oi_change_per_year": float(np.log(p["oi"].dropna().iloc[-1] / p["oi"].dropna().iloc[0]) * HOURS_YEAR / len(p)),
           "funding_mean": f8.mean(),
           "funding_share_at_base": np.isclose(f8, 1e-4, atol=1e-9).mean()}
    if "atm_30d" in p and p["atm_30d"].notna().any():
        rv_fwd = np.sqrt((r ** 2).rolling(720).sum().shift(-720) * HOURS_YEAR / 720)
        out.update({"atm30_mean": p["atm_30d"].mean(),
                    "vrp30_mean": (p["atm_30d"] - rv_fwd.reindex(p.index)).mean(),
                    "rr25_30d_mean": p["rr25_30d"].mean(),
                    "bf25_30d_mean": p["bf25_30d"].mean(),
                    "term_slope_7_90_mean": (p["atm_7d"] - p["atm_90d"]).mean()})
    return {k: float(v) for k, v in out.items()}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def build(name, p, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    st = StructuralModel().fit(p)
    st.save_state(os.path.join(out_dir, "structural_state.npz"))
    st.save(directory=out_dir)
    ol = OptionsLayer().fit(p)
    ol.save_state(os.path.join(out_dir, "options_state.npz"))
    fit_secs = time.time() - t0
    # check: synthetic paths of the same length from the saved card vs the real period
    m = models.WithOptions(StructuralModel.load(os.path.join(out_dir, "structural_state.npz")),
                           OptionsLayer.load(os.path.join(out_dir, "options_state.npz")))
    syn = pd.DataFrame([summary(m.generate(len(p), seed=s, start=str(p.index[0].date()))) for s in range(N_CHECK_PATHS)])
    real = summary(p)
    card = {"name": name,
            "description": ("fitted on the whole history" if name == "all" else "fitted on calendar year %s only" % name),
            "period": {"start": str(p.index[0]), "end": str(p.index[-1]), "n_hours": int(len(p))},
            "built": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "code_commit": git_commit(),
            "fit_seconds": round(fit_secs, 1),
            "structural": {"annual_vol_unconditional": round(st.annual_vol(), 4), "persistence": round(st.persistence, 5),
                           "start_vol_presets_calm_normal_stressed": [round(float(v), 4) for v in st.rv_quantiles],
                           "start_price": st.p0, "start_oi": st.oi0,
                           "funding_censored_share": round(st.fit_info["funding_censored_share"], 4)},
            "check": {"real": real, "synthetic_median_%d_paths" % N_CHECK_PATHS: syn.median().to_dict()},
            "usage": "models.load_card(%r).generate(n_hours, seed=..., start=...)" % name}
    with open(os.path.join(out_dir, "card.json"), "w") as f:
        json.dump(card, f, indent=1)
    return card


def write_index(cards):
    rows = ["| card | period | hours | real ann. vol | synth. ann. vol | real ATM30 | synth. ATM30 | real funding | synth. funding |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in cards:
        rl, sy = c["check"]["real"], c["check"]["synthetic_median_%d_paths" % N_CHECK_PATHS]
        rows.append("| `%s` | %s → %s | %d | %.2f | %.2f | %.2f | %.2f | %.4f %% | %.4f %% |" % (
            c["name"], c["period"]["start"][:10], c["period"]["end"][:10], c["period"]["n_hours"],
            rl["ann_vol"], sy["ann_vol"], rl.get("atm30_mean", np.nan), sy.get("atm30_mean", np.nan),
            100 * rl["funding_mean"], 100 * sy["funding_mean"]))
    text = """# cards/ — pre-built models

Fitted states of the structural generator (OHLCV + funding + OI) with the options layer (hourly IV surface),
ready to generate without downloading data or fitting:

```python
import models
models.list_cards()                      # ['2022', '2023', ..., 'all']
m = models.load_card("2022")             # a bear-market regime; "all" = whole history
path = m.generate(24 * 365, seed=1, start="2027-01-01")      # same scenario keywords as models.load(...)
m.card                                   # metadata: period, fit summary, real vs synthetic check
models.load_card("all", options=False)   # price / funding / OI only
```

A year card reproduces that year's regime (vol level, drift of funding, skew sign...). Note the year cards are
fitted on ~8,760 hours (the 90-day volatility component uses the first 2,160 as burn-in), so they are noisier
than `all`; the `all` card is the one validated with `validation/compare.py` (125–129 PASS of 151 facts over
seeds; see models/README.md). Starting price / OI / funding default to the end of the card's period.

Check table (statistics of the real period vs the median of %d synthetic paths of the same length from the card):

%s

Each folder: `structural_state.npz`, `options_state.npz` (fitted states; residual tables stored as float32),
`structural.json`, `options.json` (readable parameters), `card.json` (metadata + full check table).
Rebuild after refreshing the data: `python cards/build_cards.py` (~2 min total), using the project venv.
""" % (N_CHECK_PATHS, "\n".join(rows))
    open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="card names to build (all, 2022, 2023, ...)")
    a = ap.parse_args()
    p = panel_mod.load_real()
    periods = {"all": p}
    for y in sorted(set(p.index.year)):
        periods[str(y)] = p.loc[str(y)]
    for name, sub in periods.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        c = build(name, sub, os.path.join(HERE, name))
        print("%-5s %6d h  real vol %.2f synth %.2f  (%.0f s)" % (
            name, len(sub), c["check"]["real"]["ann_vol"], c["check"]["synthetic_median_%d_paths" % N_CHECK_PATHS]["ann_vol"],
            time.time() - t0), flush=True)
    write_index([json.load(open(os.path.join(HERE, n, "card.json"))) for n in models.list_cards()])


if __name__ == "__main__":
    main()
