# marketSampler

Synthetic BTC market paths: hourly OHLCV, funding, open interest and an implied-vol surface.
See `models/README.md` (generators), `validation/` (stylised facts and scoring), `cards/` (pre-built models),
`BACKLOG.md`.

## Setup

The project has its own virtual environment (Python 3.12, versions pinned in `requirements.txt`):

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe validation\compare.py --model structural+options
```

In VS Code, select `.venv\Scripts\python.exe` as the interpreter.

## Data

`data/cache/` (not committed) is re-created by the scripts in `data/`. The IV surface is built from the
shared, read-only option chain in `C:\ClaudeSandbox\shared-data\option-chain\` (`data/build_iv_surface_chain.py`),
which is read in place and never modified.
