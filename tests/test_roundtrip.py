"""
Create, train, grow, save and reload a paged model - the life cycle every
run goes through - and check the reloaded model computes the same thing.
"""

import os
from dataclasses import asdict

import numpy as np
import torch


def _text(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(32, 127, (1, n), generator=g)


def _train(model, opt, steps=3):
    model.train()
    x = _text(65)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x[:, :-1], targets=x[:, 1:])
        loss.backward()
        opt.step()
    return float(loss)


def test_create_train_grow_save_reload(tiny):
    from minagi import store
    from minagi.build import build_adamw, build_paged
    from minagi.create import create

    w = tiny["weights"]
    create(w, verbose=False)
    torch.manual_seed(0)
    model, cfg, pool, _ = build_paged(w, "cpu")
    assert pool.tiers.ram_capacity == 2          # from the tiny config
    opt, trunk, pool_ps = build_adamw(model, 1e-3, 0.1, 0.0)
    assert {g["name"] for g in opt.param_groups} == {"trunk", "pool"}
    loss = _train(model, opt)
    assert np.isfinite(loss)

    n0 = pool.n_experts()
    pool.add_experts(1, step=3)
    assert pool.n_experts() == n0 + 1
    store.save(model, w, step=3, val=None, opt=opt, cfg=asdict(cfg))
    assert len(os.listdir(os.path.join(w, "experts"))) == n0 + 1

    x = _text(64, seed=1)
    model.eval()
    with torch.no_grad():
        a = model(x)[0]
    m2, _, p2, man = build_paged(w, "cpu")
    m2.eval()
    assert p2.n_experts() == n0 + 1
    assert man["step"] == 3
    with torch.no_grad():
        b = m2(x)[0]
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_unlimited_tier_keeps_everything(tiny):
    from minagi.create import create
    from minagi.paged import Tiers

    w = tiny["weights"]
    create(w, verbose=False)
    t = Tiers(os.path.join(w, "experts"), 32, 32, ram_capacity=0,
              read_only=True)
    for i in range(6):
        t.fetch(i)
    assert len(t.ram) == 6 and t.evictions == 0


def test_expert_file_replaceable_after_read(tiny):
    """Windows cannot replace a file that is still open."""
    from minagi.create import create
    from minagi.paged import Tiers

    w = tiny["weights"]
    create(w, verbose=False)
    t = Tiers(os.path.join(w, "experts"), 32, 32, ram_capacity=0)
    ent = t.fetch(0)
    t.put(0, ent, dirty=True)
    t.flush()                                    # os.replace over e00000.npz
    assert t.writebacks == 1
