"""Synthetic BTC perp path generators. Every model has fit(panel) -> self and generate(n_hours, seed, start) -> panel
(columns as in validation/panel.py).

    import models
    m = models.load("structural")              # saved fitted state; fits (~2.5 min) and saves it if missing
    m = models.load("structural", refit=True)  # refit on the current data/cache and overwrite the state
"""
import os


def get(name):
    if name == "bootstrap":
        from models.bootstrap import BlockBootstrap
        return BlockBootstrap()
    if name == "structural":
        from models.structural import StructuralModel
        return StructuralModel()
    raise ValueError("unknown model %r" % name)


def _real_panel():
    import sys
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation")
    if here not in sys.path:
        sys.path.insert(0, here)
    import panel
    return panel.load_real()


def load(name, refit=False, panel=None):
    """Fitted model. Models with a saved state (structural) are loaded from models/params/ unless refit=True
    or no state exists yet, in which case they are fitted on the real panel and the state is saved."""
    if name == "structural":
        from models.structural import StructuralModel, PARAMS_DIR, STATE_FILE
        path = os.path.join(PARAMS_DIR, STATE_FILE)
        if os.path.exists(path) and not refit:
            return StructuralModel.load(path)
        m = StructuralModel().fit(panel if panel is not None else _real_panel())
        m.save_state(path)
        m.save()
        return m
    return get(name).fit(panel if panel is not None else _real_panel())
