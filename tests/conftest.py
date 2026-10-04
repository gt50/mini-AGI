"""
A model small enough to build, train and reload on a CPU in seconds.

Every test that touches a model gets its own config file and weights
directory under tmp_path, selected through MINAGI_CONFIG exactly as
`--config` selects one for train.py and serve.py.
"""

import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["model"].update(d_model=32, n_head=2, d_ff=64, max_steps=2,
                      bptt_window=2, train_steps_mean=0, context_start=128,
                      context_end=128)
    c["pool"].update(experts=6, width=32, top_k=2, resident=3, ram_cache=2)
    c["training"].update(chunk=64, precision="fp32")
    w = str(tmp_path / "weights")
    c["data"].update(weights=w)
    c["plots"]["enabled"] = False
    p = tmp_path / "tiny.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f)
    monkeypatch.setenv("MINAGI_CONFIG", str(p))
    from minagi import device
    monkeypatch.setattr(device, "_HW", None)
    return {"config": str(p), "weights": w, "cfg": c}
