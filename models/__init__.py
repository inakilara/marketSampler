"""Synthetic BTC perp path generators. Every model has fit(panel) -> self and generate(n_hours, seed, start) -> panel
(columns as in validation/panel.py)."""


def get(name):
    if name == "bootstrap":
        from models.bootstrap import BlockBootstrap
        return BlockBootstrap()
    if name == "structural":
        from models.structural import StructuralModel
        return StructuralModel()
    raise ValueError("unknown model %r" % name)
