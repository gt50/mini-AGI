"""
The hardware this process runs on, in one place.

The code was written against one NVIDIA laptop card, and ROCm builds of
PyTorch answer to the same `torch.cuda` namespace, so almost nothing needs to
know which vendor it is on. What does need to know:

  - the memory budget. Growth refuses to add an expert once the peak this
    process allocated passes a share of the card. A discrete card reports its
    size honestly; on an APU such as Strix Halo (gfx1151) the size depends on
    how much system memory the driver lends the GPU, and what torch reports
    need not match what can really be allocated. So the budget can be stated,
    and a stated budget wins.
  - whether CPU and GPU share memory. When they do, the RAM tier of the
    expert pool can sit on the device itself, and paging an expert onto the
    card becomes a device-to-device copy instead of a blocking host transfer.
  - environment variables torch reads once at import - the allocator, ROCm's
    experimental attention kernels, TunableOp - which have to be set before
    torch loads. set_env_before_torch() is the only function here that may be
    called before `import torch`; everything else imports it lazily.

The `hardware:` block of the config file drives all of it; every key is
optional and the defaults reproduce the behaviour on a discrete NVIDIA card.
"""

import os

# Integrated RDNA 3.5: Strix Point (gfx1150), Strix Halo (gfx1151), Krackan
# Point (gfx1152). Memory is the system's LPDDR5X, shared with the CPU.
UNIFIED_ARCHES = ("gfx1150", "gfx1151", "gfx1152")

_HW = None


def hw():
    """The `hardware:` block of the active config, {} when there is none."""
    global _HW
    if _HW is None:
        from minagi.config import load, get
        _HW = get(load(), "hardware", {}) or {}
    return _HW


def set_env_before_torch():
    """
    Environment torch reads once, at import or at first device init.

    Every variable is setdefault: anything exported in the shell wins.
    """
    h = hw()
    if h.get("expandable_segments", True):
        # PyTorch's caching allocator otherwise strands memory between the
        # checkpointed expert dispatch's large short-lived tensors - see the
        # note in train.py. ROCm builds read the HIP name; recent CUDA builds
        # the CUDA one. A build that does not support it warns and carries on.
        for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
            os.environ.setdefault(k, "expandable_segments:True")
    if h.get("rocm_aotriton"):
        # Without it SDPA on RDNA 3/3.5 can fall back to the math kernel,
        # which materialises the whole T x T score matrix at every row.
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    if h.get("tunableop"):
        # Tunes each GEMM shape once and remembers the winner. The expert
        # dispatch's batched matmuls are the shapes that matter here.
        os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
        os.environ.setdefault("PYTORCH_TUNABLEOP_FILENAME",
                              h.get("tunableop_file", "runs/tunableop.csv"))


def _props():
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device())


def backend():
    """'rocm', 'cuda' or 'cpu'."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def arch():
    """gfx1151, sm_86, ... or 'cpu'."""
    p = _props()
    if p is None:
        return "cpu"
    name = getattr(p, "gcnArchName", "") or ""
    if name:
        return name.split(":")[0]
    return f"sm_{p.major}{p.minor}"


def is_unified():
    """True when the GPU's memory is the system's memory."""
    if "unified" in hw():
        return bool(hw()["unified"])
    p = _props()
    if p is None:
        return False
    return arch() in UNIFIED_ARCHES or bool(getattr(p, "is_integrated", 0))


def memory_budget_bytes():
    """
    What this process may plan to allocate on the GPU, in bytes.

    hardware.gpu_mem_gb, then MINAGI_GPU_MEM_GB, then what the device reports.
    0 without a GPU.
    """
    gb = hw().get("gpu_mem_gb") or os.environ.get("MINAGI_GPU_MEM_GB")
    if gb:
        return int(float(gb) * (1 << 30))
    p = _props()
    return int(p.total_memory) if p is not None else 0


def budget_source():
    if hw().get("gpu_mem_gb"):
        return "config"
    if os.environ.get("MINAGI_GPU_MEM_GB"):
        return "MINAGI_GPU_MEM_GB"
    return "device"


def peak_allocated():
    """Peak bytes this process has allocated on the GPU, None without one."""
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated()


def mem_frac():
    """Peak allocated as a share of the budget. 0.0 without a GPU."""
    peak, budget = peak_allocated(), memory_budget_bytes()
    if peak is None or budget <= 0:
        return 0.0
    return peak / budget


def ram_tier_on_gpu(device):
    """
    Whether the expert pool's RAM tier should live on `device`.

    hardware.ram_tier_on_gpu when it is set, otherwise yes exactly when the
    device is a GPU that shares memory with the CPU.
    """
    import torch
    if torch.device(device).type != "cuda":
        return False
    if "ram_tier_on_gpu" in hw():
        return bool(hw()["ram_tier_on_gpu"])
    return is_unified()


def banner(device):
    """One line naming the hardware, printed at startup."""
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return f"hardware: cpu, torch {torch.__version__}"
    p = _props()
    b = backend()
    ver = (f"HIP {torch.version.hip}" if b == "rocm"
           else f"CUDA {torch.version.cuda}")
    extras = []
    if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1":
        extras.append("aotriton")
    if os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1":
        extras.append("tunableop")
    if ram_tier_on_gpu(device):
        extras.append("ram tier on gpu")
    return (f"hardware: {p.name} ({arch()}), {b}, torch {torch.__version__} "
            f"{ver}, {'unified' if is_unified() else 'discrete'} memory, "
            f"budget {memory_budget_bytes() / (1 << 30):.1f} GB "
            f"({budget_source()})"
            + (f", {', '.join(extras)}" if extras else ""))
