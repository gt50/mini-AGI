"""
Running only the chosen experts must compute what running all of them did:
the same outputs, and the same gradients on every slot.
"""

import torch


def _sites(model):
    from minagi.pool import PooledMLP
    return [m for m in model.modules() if isinstance(m, PooledMLP)]


def _run(model, x, subset):
    for s in _sites(model):
        s.subset_dispatch = subset
    model.zero_grad(set_to_none=True)
    logits, loss = model(x[:, :-1], targets=x[:, 1:])
    loss.backward()
    grads = [p.grad.detach().clone() if p.grad is not None else None
             for p in (model.pool.w1, model.pool.w3, model.pool.w2)]
    return logits.detach(), grads


def test_subset_dispatch_matches_full(tiny):
    from minagi.build import build_paged
    from minagi.create import create

    create(tiny["weights"], verbose=False)
    # 6 slots and top_k 2: a single character reaches 2 of them, under half,
    # which is when the subset path runs
    model, _, pool, _ = build_paged(tiny["weights"], "cpu", resident=6)
    model.train()
    for s in _sites(model):
        s.grad_checkpoint = False

    g = torch.Generator().manual_seed(3)
    for n in (2, 3):                    # one and two characters per forward
        x = torch.randint(32, 127, (1, n), generator=g)
        torch.manual_seed(0)
        a, ga = _run(model, x, subset=True)
        torch.manual_seed(0)
        b, gb = _run(model, x, subset=False)
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
        for u, v in zip(ga, gb):
            assert (u is None) == (v is None)
            if u is not None:
                assert torch.allclose(u, v, atol=1e-6), (u - v).abs().max()
