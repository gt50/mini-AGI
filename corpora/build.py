"""
Build the whole training corpus in one command.

    python3 -m corpora all

The model reads eight subjects in round-robin, and each takes an EQUAL share
of the reading however much of it exists. A lane ten times larger than another
is therefore not read ten times as much - it is read ten times less
thoroughly. That is why the defaults here sample rather than take everything:
there is little point pulling all of Wikipedia before the model has been
through any of it. `--full` takes the lot.

Four lanes come down from Hugging Face. Four are made here from things already
on the machine - Python packages for code, a PGN for chess, and pure synthesis
for arithmetic and the self-knowledge turns.

The generated chat lane - primal chat and the self-knowledge turns - and the
OpenHermes download share data/train/chat: the download sits in its hermes/
sub-folder, the generated files beside it. They are held out separately, as
data/val/chat and data/val/chat_hermes, because rebuilding the generated lane
clears its own files from data/{train,val}/chat - which, when OpenHermes held
out into the same folder, silently deleted OpenHermes' held-out set.

A lane that already has files is left alone, so an interrupted build can
simply be run again; `--force` rebuilds it anyway. A lane that fails does not
stop the others - what failed is named at the end so it can be retried on its
own with `--only`.
"""

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (where it lands, what makes it)
# Sampled sizes first, then what --full uses instead (0 meaning everything).
LANES = ["wikipedia", "stories", "chat", "reasoning",
         "arithmetic", "code", "chess", "self-knowledge"]

SAMPLED = {"wikipedia": 120_000, "stories": 400_000,
           "chat": 200_000, "reasoning": 20_000}


def _files(where):
    """A lane's files: everything under a directory, or what a glob matches.

    A glob, because the generated chat lane writes chat-*.txt into a folder
    that also holds the OpenHermes download, so "the folder has files" is
    true of it before the lane has been built at all."""
    if any(c in where for c in "*?["):
        return [f for f in glob.glob(where) if os.path.isfile(f)]
    return [os.path.join(r, f) for r, _, fs in os.walk(where) for f in fs]


def _has_files(where):
    return bool(_files(where))


def _sub(*args):
    """
    One lane, in its own process.

    Not an in-process call. The streaming reader leaves a worker thread alive
    and CPython aborts finalising it, so corpora/fetch.py ends with os._exit
    once its files are written - which in-process would take the whole build
    down with it, after the first lane. A subprocess also means a lane that
    segfaults costs that lane and nothing else.
    """
    cmd = [sys.executable, "-m", "corpora"] + [str(x) for x in args]
    return subprocess.run(cmd, cwd=ROOT).returncode


def _hold(limit, default=400):
    """
    How many records to keep back, when `limit` may be smaller than the hold.

    A trial run of 40 records against a fixed hold of 400 puts every one of
    them in the held-out set and none in training, which looks like the
    download failed. Never take more than a tenth.
    """
    return default if not limit else max(1, min(default, limit // 10))


def build_wikipedia(limit):
    return _sub("fetch", "--dataset", "wikimedia/wikipedia",
                "--config", "20231101.en", "--kind", "text", "--streaming",
                "--limit", limit, "--hold", _hold(limit),
                "--out", "data/train/wikipedia",
                "--held-out", "data/val/wikipedia")


def build_stories(limit):
    return _sub("fetch", "--dataset", "roneneldan/TinyStories",
                "--kind", "text", "--limit", limit, "--hold", _hold(limit),
                "--out", "data/train/stories",
                "--held-out", "data/val/stories")


def build_chat(limit):
    return _sub("fetch", "--dataset", "teknium/OpenHermes-2.5",
                "--kind", "chat", "--limit", limit, "--hold", _hold(limit),
                "--out", "data/train/chat/hermes",
                "--held-out", "data/val/chat_hermes")


def build_reasoning(limit):
    return _sub("reasoning", "--limit", limit, "--hold", _hold(limit))


def _generated(target, expand_as, *extra):
    """A lane written as .bin by a generator, then expanded to text."""
    rc = _sub(target, *extra)
    return rc or _sub("expand", "--only", expand_as)


def build_arithmetic(limit):
    n = limit or 4_000_000
    return _generated("arithmetic", "arithmetic", "--n", n,
                      "--val", max(200, n // 200))


def build_code(_):
    return _generated("code", "code")


def build_chess(_):
    return _generated("chess", "chess")


def build_self_knowledge(_):
    return _generated("chat", "chat")


BUILDERS = {
    "wikipedia":      (build_wikipedia,      "data/train/wikipedia"),
    "stories":        (build_stories,        "data/train/stories"),
    "chat":           (build_chat,           "data/train/chat/hermes"),
    "reasoning":      (build_reasoning,      "data/train/reasoning"),
    "arithmetic":     (build_arithmetic,     "data/train/arithmetic"),
    "code":           (build_code,           "data/train/code"),
    "chess":          (build_chess,          "data/train/chess"),
    # the generated lane lands in data/train/chat, next to hermes/ - see the
    # module docstring
    "self-knowledge": (build_self_knowledge, "data/train/chat/chat-*.txt"),
}


def main():
    ap = argparse.ArgumentParser(
        prog="python3 -m corpora all",
        description="build every lane of the training corpus")
    ap.add_argument("--only", nargs="+", metavar="LANE", default=None,
                    choices=LANES, help="build just these")
    ap.add_argument("--full", action="store_true",
                    help="take entire datasets instead of a sample; tens of "
                         "gigabytes and many hours")
    ap.add_argument("--force", action="store_true",
                    help="rebuild lanes that already have files")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="records per downloaded lane, overriding the "
                         "sampled defaults. Useful for a quick trial run "
                         "before committing to the real thing")
    a = ap.parse_args()

    os.chdir(ROOT)
    try:
        import datasets                                   # noqa: F401
    except ImportError:
        print("the downloaded lanes need the datasets package:\n"
              "    pip install datasets", file=sys.stderr)
        if not a.only or set(a.only) & {"wikipedia", "stories", "chat",
                                        "reasoning"}:
            return 1

    wanted = a.only or LANES
    failed, skipped = [], []
    for name in wanted:
        fn, where = BUILDERS[name]
        if not a.force and _has_files(where):
            print(f"== {name}: already in {where} - skipping "
                  f"(--force to rebuild)")
            skipped.append(name)
            continue
        print(f"== {name} -> {where}", flush=True)
        if a.limit is not None:
            limit = a.limit
        else:
            limit = 0 if a.full else SAMPLED.get(name, 0)
        fn(limit)
        if not _has_files(where):
            print(f"!! {name} produced nothing - carrying on", file=sys.stderr)
            failed.append(name)

    print("\ncorpus:")
    total = 0
    for name in LANES:
        where = BUILDERS[name][1]
        files = _files(where)
        if not files:
            continue
        n, mb = len(files), sum(os.path.getsize(f) for f in files)
        total += mb
        print(f"  {name:<16} {n:>7,} files  {mb / 1e6:>8,.0f} MB")
    print(f"  {'TOTAL':<16} {'':>7}         {total / 1e6:>8,.0f} MB")

    if failed:
        print(f"\nlanes that produced nothing: {' '.join(failed)}")
        print(f"retry with: python3 -m corpora all --only {' '.join(failed)}")

    print("\nnext:\n    python3 train.py read data/train --save "
          "--held-out data/val")
    return 1 if failed else 0
