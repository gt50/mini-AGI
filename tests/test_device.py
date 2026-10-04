import os
from types import SimpleNamespace

from minagi import device


def _props(name="", total=8 << 30, integrated=0):
    return SimpleNamespace(gcnArchName=name, total_memory=total, major=8,
                           minor=6, is_integrated=integrated, name="test")


def test_budget_precedence(monkeypatch):
    monkeypatch.setattr(device, "_props", lambda: _props(total=2 << 30))
    monkeypatch.delenv("MINAGI_GPU_MEM_GB", raising=False)

    monkeypatch.setattr(device, "_HW", {})
    assert device.memory_budget_bytes() == 2 << 30
    assert device.budget_source() == "device"

    monkeypatch.setenv("MINAGI_GPU_MEM_GB", "16")
    assert device.memory_budget_bytes() == 16 << 30
    assert device.budget_source() == "MINAGI_GPU_MEM_GB"

    monkeypatch.setattr(device, "_HW", {"gpu_mem_gb": 96})
    assert device.memory_budget_bytes() == 96 << 30
    assert device.budget_source() == "config"


def test_no_gpu_budget_is_zero(monkeypatch):
    monkeypatch.setattr(device, "_props", lambda: None)
    monkeypatch.setattr(device, "_HW", {})
    monkeypatch.delenv("MINAGI_GPU_MEM_GB", raising=False)
    assert device.memory_budget_bytes() == 0


def test_unified_detection(monkeypatch):
    monkeypatch.setattr(device, "_HW", {})
    monkeypatch.setattr(device, "_props", lambda: _props("gfx1151:xnack-"))
    assert device.arch() == "gfx1151"
    assert device.is_unified()

    monkeypatch.setattr(device, "_props", lambda: _props("gfx1100"))
    assert not device.is_unified()

    monkeypatch.setattr(device, "_props", lambda: _props(""))
    assert device.arch() == "sm_86"
    assert not device.is_unified()

    monkeypatch.setattr(device, "_HW", {"unified": True})
    assert device.is_unified()


def test_ram_tier_never_on_cpu(monkeypatch):
    monkeypatch.setattr(device, "_HW", {"ram_tier_on_gpu": True})
    assert device.ram_tier_on_gpu("cpu") is False


def test_env_is_setdefault(monkeypatch):
    monkeypatch.setattr(device, "_HW", {"rocm_aotriton": True,
                                        "tunableop": True})
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.delenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
                       raising=False)
    device.set_env_before_torch()
    assert os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
    assert os.environ["PYTORCH_TUNABLEOP_ENABLED"] == "0"   # the shell wins
