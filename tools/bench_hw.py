#!/usr/bin/env python3
"""
What this machine costs the model, measured.

    python tools/bench_hw.py                                  # config.yaml
    python tools/bench_hw.py --config configs/strix_halo_large.yaml
    python tools/bench_hw.py --quick                          # a minute

No corpus and no weights needed: it builds a throwaway model from the config
in a temporary directory and times the four things that decide how fast it
reads.

  attention   F.scaled_dot_product_attention at the config's shapes, with no
              cache (is_causal) and behind one (the boolean mask every
              cached forward uses), and which SDPA kernels this build will
              run. On ROCm without AOTriton the masked path is the math
              kernel, which builds the whole T x T score matrix.
  experts     the expert dispatch's batched matmuls, forward and backward,
              at [resident, capacity, d_model] x [resident, d_model, width].
  step        one training step as the reader takes it: `chunk` characters
              behind a cache of the rest of the window, forward, backward,
              AdamW. Characters per second and peak memory.
  paging      one expert from disk into the RAM tier, and from the tier onto
              a VRAM slot - with the tier in host memory and, on a GPU, on
              the device.

Run it once per config you are considering, on the machine you will train
on, before committing weeks to a run. Results go to runs/bench_hw-<arch>.json.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from minagi.config import take_config_flag  # noqa: E402
from minagi.device import set_env_before_torch  # noqa: E402
take_config_flag()
set_env_before_torch()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev, reps, warm=2):
    """Median milliseconds of fn() over reps, after warm calls."""
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def sdpa_kernels(dev, dt, H, hd):
    """Which SDPA backends this build will run at these shapes."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return {}
    q = torch.randn(1, H, 256, hd, device=dev, dtype=dt)
    out = {}
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        be = getattr(SDPBackend, name, None)
        if be is None:
            continue
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            out[name.lower()] = True
        except RuntimeError:
            out[name.lower()] = False
    return out


def bench_attention(dev, dt, H, hd, lengths, chunk, reps):
    rows = []
    for T in lengths:
        q = torch.randn(1, H, T, hd, device=dev, dtype=dt)
        try:
            causal = timed(lambda: F.scaled_dot_product_attention(
                q, q, q, is_causal=True), dev, reps)
        except RuntimeError as e:          # out of memory, most likely
            causal = f"failed: {str(e).splitlines()[0][:80]}"
        # the cached path: `chunk` queries against T keys, an offset triangle
        Q = min(chunk, T)
        qc = torch.randn(1, H, Q, hd, device=dev, dtype=dt)
        P = T - Q
        mask = (torch.arange(T, device=dev).unsqueeze(0)
                <= torch.arange(Q, device=dev).unsqueeze(1) + P)
        try:
            masked = timed(lambda: F.scaled_dot_product_attention(
                qc, q, q, attn_mask=mask), dev, reps)
        except RuntimeError as e:
            masked = f"failed: {str(e).splitlines()[0][:80]}"
        rows.append({"T": T, "causal_ms": causal, "cached_ms": masked,
                     "cached_queries": Q})
        del q, qc, mask
    return rows


def bench_experts(dev, dt, resident, d_model, width, chunk, top_k, cf, reps):
    cap = max(1, int(chunk * top_k / resident * cf))
    x = torch.randn(resident, cap, d_model, device=dev, dtype=dt,
                    requires_grad=True)
    w1 = torch.randn(resident, d_model, width, device=dev, dtype=dt,
                     requires_grad=True)
    w3 = torch.randn_like(w1, requires_grad=True)
    w2 = torch.randn(resident, width, d_model, device=dev, dtype=dt,
                     requires_grad=True)

    def fwd():
        return torch.bmm(F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3), w2)

    def fwd_bwd():
        fwd().float().sum().backward()

    with torch.no_grad():
        f = timed(fwd, dev, reps)
    fb = timed(fwd_bwd, dev, reps)
    flops = 2 * resident * cap * d_model * width * 3
    return {"resident": resident, "capacity": cap, "fwd_ms": f,
            "fwd_bwd_ms": fb, "fwd_tflops": flops / (f * 1e-3) / 1e12,
            "fwd_bwd_tflops": 3 * flops / (fb * 1e-3) / 1e12}


def bench_step(dev, wdir, chunk, context, reps, lr=3e-4):
    from minagi.build import build_paged, _split_trunk_pool
    from minagi.precision import amp
    from minagi.stream import detach_caches, trim_caches

    model, cfg, pool, _ = build_paged(wdir, dev, ceiling=context)
    model.train()
    trunk, pool_ps = _split_trunk_pool(model)
    opt = torch.optim.AdamW(
        [{"params": trunk, "lr": lr * 0.1}, {"params": pool_ps, "lr": lr}],
        lr=lr, betas=(0.9, 0.95), fused=(dev.type == "cuda"))
    pool.attach_optimiser(opt)
    g = torch.Generator().manual_seed(0)

    def text(n):
        return torch.randint(32, 127, (1, n), generator=g).to(dev)

    caches = model.empty_caches()
    P = max(0, context - chunk)
    if P:
        with torch.no_grad(), amp(dev):
            for i in range(0, P, chunk):
                n = min(chunk, P - i)
                model(text(n), caches=caches, pos_offset=i)

    def step():
        trim_caches(caches, context - chunk)
        pos = caches[0]["k"].shape[-2] if caches[0].get("k") is not None else 0
        x = text(chunk + 1)
        opt.zero_grad(set_to_none=True)
        with amp(dev):
            _, loss = model(x[:, :-1], targets=x[:, 1:], caches=caches,
                            pos_offset=pos)
        loss.backward()
        detach_caches(caches)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ms = timed(step, dev, reps, warm=1)
    peak = (torch.cuda.max_memory_allocated() / (1 << 30)
            if dev.type == "cuda" else None)
    rep = pool.report() if hasattr(pool, "report") else {}
    return {"chunk": chunk, "context": context, "step_ms": ms,
            "chars_per_s": chunk / (ms * 1e-3), "peak_gb": peak,
            "resident": pool.resident, "experts": pool.n_experts(),
            "loads": getattr(pool, "loads", None),
            "tier_hit_rate": rep.get("hit_rate")}


def bench_paging(dev, wdir, reps):
    from minagi.paged import Tiers
    import json as _j
    with open(os.path.join(wdir, "manifest.json"), encoding="utf-8") as f:
        man = _j.load(f)
    cfg = man["cfg"]
    d_model, width = cfg["d_model"], cfg["pool_d_ff"]
    exp = os.path.join(wdir, "experts")
    ids = sorted(int(n[1:6]) for n in os.listdir(exp) if n.endswith(".npz"))
    slot = torch.zeros(width, d_model, device=dev)
    out = {}
    places = [("host", False)] + ([("device", True)] if dev.type == "cuda" else [])
    for name, on_dev in places:
        t = Tiers(exp, d_model, width, ram_capacity=0, device=dev,
                  read_only=True, tier_on_device=on_dev)
        n = min(reps, len(ids))
        t0 = time.perf_counter()
        for i in ids[:n]:
            t.fetch(i)
        sync(dev)
        disk_ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)

        def to_slot():
            for i in ids[:n]:
                slot.copy_(t.fetch(i)["w1"])
                slot.copy_(t.fetch(i)["w3"])
        tier_ms = timed(to_slot, dev, 3, warm=1) / max(n, 1)
        out[name] = {"disk_to_tier_ms": disk_ms,
                     "tier_to_slot_ms_2_of_3_tensors": tier_ms}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true",
                    help="fewer repetitions and shorter windows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", default="",
                    help="comma list of attention,experts,step,paging")
    a = ap.parse_args()

    from minagi import device as hwdev
    from minagi.config import get, load, path as cfg_path
    from minagi.precision import compute_dtype, set_compute_dtype

    c = load()
    dev = torch.device(a.device)
    set_compute_dtype(get(c, "training.precision", "bf16"))
    dt = compute_dtype() if dev.type == "cuda" else torch.float32
    skip = set(s for s in a.skip.split(",") if s)

    d_model = int(get(c, "model.d_model", 512))
    H = int(get(c, "model.n_head", 8))
    width = int(get(c, "pool.width", 2048))
    resident = int(get(c, "pool.resident", 32))
    top_k = int(get(c, "pool.top_k", 8))
    cf = float(get(c, "pool.capacity_factor", 1.5) or 1.5)
    chunk = int(get(c, "training.chunk", 2048))
    ctx = int(get(c, "model.context_end", 4096))
    reps = 3 if a.quick else 10
    if a.quick:
        ctx = min(ctx, 4096)

    res = {"config": cfg_path(), "banner": hwdev.banner(dev),
           "backend": hwdev.backend(), "arch": hwdev.arch(),
           "torch": torch.__version__, "dtype": str(dt)}
    print(res["banner"])
    print(f"config {cfg_path()}: d_model {d_model}, heads {H}, width {width}, "
          f"resident {resident}, top_k {top_k}, chunk {chunk}, "
          f"context {ctx}")

    if "attention" not in skip:
        res["sdpa_kernels"] = sdpa_kernels(dev, dt, H, d_model // H)
        print(f"\nSDPA kernels available: {res['sdpa_kernels']}")
        lengths = sorted({min(2048, ctx), ctx} | ({2 * ctx} if not a.quick
                                                  else set()))
        res["attention"] = bench_attention(dev, dt, H, d_model // H, lengths,
                                           chunk, reps)
        for r in res["attention"]:
            f = (lambda v: f"{v:8.2f} ms" if isinstance(v, float) else v)
            print(f"  T {r['T']:>6}: causal {f(r['causal_ms'])}   "
                  f"cached ({r['cached_queries']} queries) {f(r['cached_ms'])}")

    if "experts" not in skip:
        res["experts"] = bench_experts(dev, dt, resident, d_model, width,
                                       chunk, top_k, cf, reps)
        e = res["experts"]
        print(f"\nexperts: {resident} x [{e['capacity']}, {d_model}] x "
              f"[{d_model}, {width}]  fwd {e['fwd_ms']:.2f} ms "
              f"({e['fwd_tflops']:.1f} TFLOP/s)  fwd+bwd "
              f"{e['fwd_bwd_ms']:.2f} ms ({e['fwd_bwd_tflops']:.1f} TFLOP/s)")

    if not ({"step", "paging"} <= skip):
        from minagi.create import create
        tmp = tempfile.mkdtemp(prefix="minagi-bench-")
        wdir = os.path.join(tmp, "weights")
        try:
            create(wdir, verbose=False)
            if "paging" not in skip:
                res["paging"] = bench_paging(dev, wdir, 8 if a.quick else 32)
                print("\npaging, per expert:")
                for k, v in res["paging"].items():
                    print(f"  tier in {k:6}: disk -> tier "
                          f"{v['disk_to_tier_ms']:.1f} ms, tier -> slot "
                          f"{v['tier_to_slot_ms_2_of_3_tensors']:.2f} ms")
            if "step" not in skip:
                res["step"] = bench_step(dev, wdir, chunk, ctx,
                                         2 if a.quick else 5)
                s = res["step"]
                pk = (f", peak {s['peak_gb']:.1f} GB"
                      if s["peak_gb"] is not None else "")
                print(f"\ntraining step: {s['chunk']} characters behind "
                      f"{s['context']}: {s['step_ms']:.0f} ms, "
                      f"{s['chars_per_s']:,.0f} char/s{pk}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    out = a.out or os.path.join(HERE, "runs", f"bench_hw-{res['arch']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
