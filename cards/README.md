# cards/ — pre-built models

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

Check table (statistics of the real period vs the median of 3 synthetic paths of the same length from the card):

| card | period | hours | real ann. vol | synth. ann. vol | real ATM30 | synth. ATM30 | real funding | synth. funding |
|---|---|---|---|---|---|---|---|---|
| `2022` | 2022-01-01 → 2022-12-31 | 8760 | 0.64 | 0.45 | 0.65 | 0.52 | 0.0038 % | 0.0037 % |
| `2023` | 2023-01-01 → 2023-12-31 | 8760 | 0.43 | 0.40 | 0.46 | 0.45 | 0.0072 % | 0.0134 % |
| `2024` | 2024-01-01 → 2024-12-31 | 8784 | 0.53 | 0.51 | 0.56 | 0.54 | 0.0109 % | 0.0080 % |
| `2025` | 2025-01-01 → 2025-12-31 | 8760 | 0.44 | 0.43 | 0.44 | 0.43 | 0.0047 % | 0.0036 % |
| `2026` | 2026-01-01 → 2026-11-18 | 7723 | 0.43 | 0.40 | 0.40 | 0.39 | 0.0031 % | 0.0025 % |
| `all` | 2022-01-01 → 2026-11-18 | 42787 | 0.50 | 0.44 | 0.50 | 0.42 | 0.0060 % | 0.0056 % |

Each folder: `structural_state.npz`, `options_state.npz` (fitted states; residual tables stored as float32),
`structural.json`, `options.json` (readable parameters), `card.json` (metadata + full check table).
Rebuild after refreshing the data: `python cards/build_cards.py` (~2 min total), using the project venv.
