"""
Reading config.yaml.

One file holds the settings worth changing, and both the tool that creates a
model and the one that trains it read it, so a model cannot be built with one
shape and trained with another. Command-line flags still win where they are
given - the file is the default, not a cage.

A different file can stand in for config.yaml - a hardware profile such as
configs/strix_halo.yaml - through MINAGI_CONFIG or `--config` on train.py and
serve.py. Every reader in the process then reads that one file, so the shape a
model is built with and the shape it is trained with still cannot disagree.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "config.yaml")
ENV = "MINAGI_CONFIG"


def path():
    """The config file this process reads."""
    return os.environ.get(ENV) or DEFAULT


def take_config_flag(argv=None):
    """
    Pull `--config FILE` out of argv and point MINAGI_CONFIG at it.

    Has to run before argparse, because the parsers take their defaults from
    the config while they are being built, and before torch, because the
    hardware block sets environment variables torch reads once at import.
    """
    argv = sys.argv if argv is None else argv
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            p = argv[i + 1]
            del argv[i:i + 2]
        elif a.startswith("--config="):
            p = a.split("=", 1)[1]
            del argv[i]
        else:
            continue
        if not os.path.exists(p):
            raise SystemExit(f"--config {p}: no such file")
        os.environ[ENV] = os.path.abspath(p)
        return os.environ[ENV]
    return None


def load(path_=None):
    import yaml
    p = path_ or path()
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg, dotted, fallback=None):
    """get(cfg, 'pool.d_ff') - missing sections return the fallback."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur
