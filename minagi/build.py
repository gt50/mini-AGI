"""
Building a model from a weights directory.

These lived in train.py, which made the library import its own entry point to
load a paged directory (recur.load_recur did `import train`). They are the
library's: train.py, serve.py and the replication probe all reach them here.
"""

import json
import os

import numpy as np
import torch

from minagi.pool import PooledMLP
from minagi.recur import RecurConfig, RecurCoder


def _split_trunk_pool(model):
    """
    Which parameters are the pool's, and which are the shared body.

    A resident pool keeps its experts as modules; a paged pool keeps three
    stacked slot tensors instead, so the set has to be found by asking the
    pool rather than by walking a ModuleList that is not there.
    """
    p = model.pool
    ids = {id(p.gate)}
    if hasattr(p, "experts") and p.experts is not None:
        ids |= {id(q) for e in p.experts for q in e.parameters()}
    for nm in ("w1", "w3", "w2"):
        obj = getattr(p, nm, None)
        if obj is None:
            continue
        ids |= ({id(obj)} if torch.is_tensor(obj)
                else {id(q) for q in obj.parameters()})
    for s in model.modules():
        if isinstance(s, PooledMLP):
            ids.add(id(s.router.weight))
            ids.add(id(s.depth_emb))
    trunk = [q for q in model.parameters() if id(q) not in ids]
    pool = [q for q in model.parameters() if id(q) in ids]
    return trunk, pool


def build_paged(wdir, device, resident=None, ram_capacity=None, ceiling=None,
                read_only=False):
    """
    Load the model with its pool on disk rather than in VRAM.

    The three tiers the project describes: every expert is a file, RAM keeps
    the recently wanted ones, and the card holds the experts the current
    forward admitted. While a forward has room on the card, every character ranks the
    whole pool; a forward may use at most `resident` experts, the ones its
    characters ask for most - see PagedPool.admit.

    read_only makes the pool incapable of writing to `wdir`. Pass it from any
    tool that inspects a directory a training run may own - paging an expert in
    marks it dirty whether or not anything touched it, so a plain read would
    otherwise write expert files back under the run.

    ram_capacity None takes pool.ram_cache from the config; 0 keeps every
    expert the pool has. Where the RAM tier itself lives is
    minagi.device.ram_tier_on_gpu's call - on a unified-memory APU it sits on
    the device, so paging onto the card is a device-to-device copy.
    """
    from minagi.paged import PagedPool
    from minagi.device import ram_tier_on_gpu
    from minagi.config import load as _load_cfg, get as _get_cfg
    if ram_capacity is None:
        ram_capacity = int(_get_cfg(_load_cfg(), "pool.ram_cache", 256))
    with open(os.path.join(wdir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    cfgd = dict(man["cfg"])
    resident = resident or cfgd.get("pool_resident") or cfgd["pool_top_k"]
    cfg = RecurConfig(**{k: v for k, v in cfgd.items()
                         if k in RecurConfig.__dataclass_fields__})
    # The capacity bound comes from config.yaml rather than from the
    # checkpoint, because it is a property of the machine the model is running
    # on - how much VRAM the dispatch may use - not of the model. A directory
    # written before this existed carries no value for it and would otherwise
    # get the dataclass default, which is right but silent; taking it from the
    # config every load means the number in the file is the number in effect.
    try:
        from minagi.config import load as _load_cfg, get as _get_cfg
        _c = _load_cfg()
        cfg.pool_capacity_factor = float(
            _get_cfg(_c, "pool.capacity_factor", cfg.pool_capacity_factor))
    except Exception:
        pass
    # the per-token router keeps one row per EXPERT, not per VRAM slot, so it
    # is sized by the pool and grows with it - see PooledMLP.__init__
    # EXACTLY what the pool holds. pool_max sizes the routers, which keep one
    # row per expert, so a checkpoint's router weights only fit a model built
    # at the same count - one row too many and load_state_dict refuses the
    # whole model.
    #
    # It was briefly max(cfg.pool_max, n_experts) to stop `stream` latching
    # its growth ceiling to the pool's current size. That conflated two jobs
    # in one number: sizing the routers, and capping growth. The manifest's
    # pool_max ratchets up and never comes down, so after a prune took the
    # pool from 158 to 157 the max() still read 158 and the directory would
    # not load at all. The ceiling now lives in `stream` where it belongs.
    cfg.pool_max = int(man["n_experts"])
    if ceiling and ceiling > cfg.block:
        # RoPE tables are built to cfg.block and carry no learned parameters,
        # so raising the ceiling on an existing model costs a bigger table and
        # nothing else. The window the reader actually uses is separate, and
        # grows a character at a time.
        cfg.block = int(ceiling)
    model = RecurCoder(cfg).to(device)
    pool = PagedPool(os.path.join(wdir, "experts"), cfg.d_model, cfg.pool_d_ff,
                     int(man["n_experts"]), resident=resident,
                     ram_capacity=ram_capacity, device=device,
                     read_only=read_only,
                     tier_on_device=ram_tier_on_gpu(device)).to(device)
    for m in model.modules():
        if isinstance(m, PooledMLP):
            m._pool[0] = pool
    model.pool = pool
    pool.attach_sites(model)
    pool.load_telemetry(man.get("telemetry"))
    # The balance term's weight and the temperature of expert selection - see
    # PagedPool.note_balance and PagedPool._draw. Read from config.yaml at
    # every load, like the capacity bound: they are choices about how the model
    # trains and chooses, not properties of its weights.
    try:
        from minagi.config import load as _load_cfg, get as _get_cfg
        _c = _load_cfg()
        pool.balance = float(_get_cfg(_c, "pool.balance", 0.0) or 0.0)
        pool.select_temperature = float(
            _get_cfg(_c, "pool.select_temperature", 0.0) or 0.0)
    except Exception:
        pass
    ever = cfgd.get("pool_ever")
    if ever:
        n = min(len(ever), pool.ever.numel())
        pool.ever[:n] = torch.tensor(ever[:n], dtype=torch.bool,
                                     device=pool.ever.device)
    # the trunk still comes from the bundles; the experts come from their files
    #
    # Both are opened in `with` so the handles close here: on Windows a file
    # still open cannot be replaced, and the next checkpoint replaces these.
    with np.load(os.path.join(wdir, "core.npz")) as core,             np.load(os.path.join(wdir, "routers.npz")) as rout:
        sd = {k: torch.from_numpy(core[k]) for k in core.files}
        # Router rows belong to EXPERTS, not to VRAM slots, so they are loaded
        # whole. Slicing them to the resident count - which this did, left
        # over from when rows were slots - silently discarded every expert
        # past the card and made any resume after growth fail to load.
        for k in rout.files:
            # The segment router chose working sets before selection moved
            # into the router itself; directories written then still carry
            # its rows.
            if k.startswith("pool.segment_router."):
                continue
            sd[k] = torch.from_numpy(rout[k])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in unexpected if "router" in k or "gate" in k]
    if bad:
        raise RuntimeError(f"router/gate tensors did not load: {bad}")
    return model, cfg, pool, man
