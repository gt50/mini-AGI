"""
The grouped dispatch must compute what one padded rectangle did: the same
outputs, and the same gradients on every slot - for a character being written
and for a chunk whose experts carry very different loads.
"""

import torch


def _sites(model):
    from minagi.pool import PooledMLP
    return [m for m in model.modules() if isinstance(m, PooledMLP)]


def _run(model, x, grouped):
    for s in _sites(model):
        s.grouped_dispatch = grouped
    model.zero_grad(set_to_none=True)
    logits, loss = model(x[:, :-1], targets=x[:, 1:])
    loss.backward()
    grads = [p.grad.detach().clone() if p.grad is not None else None
             for p in (model.pool.w1, model.pool.w3, model.pool.w2)]
    return logits.detach(), grads


def _model(tiny, resident=6):
    from minagi.build import build_paged
    from minagi.create import create
    create(tiny["weights"], verbose=False)
    model, _, _, _ = build_paged(tiny["weights"], "cpu", resident=resident)
    model.train()
    for s in _sites(model):
        s.grad_checkpoint = False
    return model


def _same(a, ga, b, gb):
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    for u, v in zip(ga, gb):
        assert (u is None) == (v is None)
        if u is not None:
            assert torch.allclose(u, v, atol=1e-6), (u - v).abs().max()


def test_bucket_sizes():
    from minagi.pool import CAP_STEP, _bucket
    assert [_bucket(c) for c in (0, 1, 2, 3, 5, 64, 65, 128)] == \
        [0, 1, 2, 4, 8, 64, 128, 128]
    assert _bucket(CAP_STEP + 1) == 2 * CAP_STEP
    assert _bucket(1000) == 1024 and _bucket(1025) == 1152
    # few sizes in play across every capacity a 2,048-character chunk can hit
    assert len({_bucket(c) for c in range(1, 2049)}) <= 24


def test_padding_changes_nothing(tiny, monkeypatch):
    """A rounded-up rectangle computes what an exact one did."""
    import minagi.pool as pool_mod
    model = _model(tiny)
    x = torch.randint(32, 127, (1, 40), generator=torch.Generator().manual_seed(5))
    torch.manual_seed(0)
    a, ga = _run(model, x, grouped=False)
    monkeypatch.setattr(pool_mod, "_bucket", lambda c: c)
    torch.manual_seed(0)
    b, gb = _run(model, x, grouped=False)
    _same(a, ga, b, gb)


def test_grouped_matches_rectangle_when_writing(tiny):
    """One character: top_k of the card, one group."""
    model = _model(tiny)
    g = torch.Generator().manual_seed(3)
    for n in (2, 3):                    # one and two characters per forward
        x = torch.randint(32, 127, (1, n), generator=g)
        torch.manual_seed(0)
        a, ga = _run(model, x, grouped=True)
        torch.manual_seed(0)
        b, gb = _run(model, x, grouped=False)
        _same(a, ga, b, gb)


def test_grouped_matches_rectangle_with_uneven_load(tiny, monkeypatch):
    """A chunk whose experts carry different loads splits into groups."""
    import minagi.pool as pool_mod
    model = _model(tiny)
    seen = []
    real = pool_mod._bucket

    def spy(c):
        seen.append(c)
        return real(c)
    monkeypatch.setattr(pool_mod, "_bucket", spy)

    x = torch.randint(32, 127, (1, 97), generator=torch.Generator().manual_seed(9))
    torch.manual_seed(0)
    a, ga = _run(model, x, grouped=True)
    grouped_sizes = set(real(c) for c in seen)
    seen.clear()
    torch.manual_seed(0)
    b, gb = _run(model, x, grouped=False)
    _same(a, ga, b, gb)
    # the grouped path really did split the experts by load
    assert len(grouped_sizes) >= 2, grouped_sizes
