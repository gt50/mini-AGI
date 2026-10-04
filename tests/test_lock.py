import json
import os
import socket

import pytest

from minagi import lock


def _write(wdir, pid, host=None):
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, lock.NAME), "w", encoding="utf-8") as f:
        json.dump({"pid": pid, "host": host or socket.gethostname(),
                   "role": "test"}, f)


def test_acquire_release(tmp_path):
    w = str(tmp_path / "w")
    lock.acquire(w, "test")
    assert lock.owner(w)["pid"] == os.getpid()
    lock.acquire(w, "test")                 # already ours: fine
    lock.release(w)
    assert lock.owner(w) is None


def test_live_owner_refuses(tmp_path):
    w = str(tmp_path / "w")
    _write(w, os.getppid())                 # the test runner's parent: alive
    with pytest.raises(lock.Busy):
        lock.acquire(w, "test")


def test_dead_owner_is_taken_over(tmp_path):
    w = str(tmp_path / "w")
    _write(w, 2 ** 22 + 12345)              # no such process
    lock.acquire(w, "test")
    assert lock.owner(w)["pid"] == os.getpid()
    lock.release(w)


def test_other_host_is_never_stale(tmp_path):
    w = str(tmp_path / "w")
    _write(w, 1, host="some-other-machine")
    with pytest.raises(lock.Busy):
        lock.acquire(w, "test")
