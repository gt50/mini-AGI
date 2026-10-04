"""
One writer per weights directory.

A training read and a learning server both page experts in and out of the
same directory, and each writes back what it trained. Nothing used to stop
them running at once, and then the last writer won: a manifest from one with
the expert count of the other, an expert file pruned by one while the other
was about to read it. On Windows a save could also fail outright, because a
file the other process holds open cannot be replaced.

So the directory carries a lock file naming its owner. It is created with
O_EXCL, which makes taking it atomic, and holds the pid, host and role of the
process that took it. A lock whose process is gone - a run that was killed -
is stale and is taken over; one held by a live process is refused, and the
caller decides what to do instead (serve.py serves read-only).
"""

import atexit
import json
import os
import socket
import sys
import time

NAME = ".lock"


class Busy(RuntimeError):
    """The directory is held by another live process."""

    def __init__(self, path, owner):
        self.path, self.owner = path, owner
        super().__init__(
            f"{os.path.dirname(path)} is in use by {owner.get('role', '?')} "
            f"(pid {owner.get('pid')} on {owner.get('host')}, since "
            f"{owner.get('since', '?')}). Stop it first, or point this at "
            f"another directory. Delete {path} only if you are sure it is "
            f"not running.")


def _alive(pid):
    """Whether a process with this pid exists on this machine."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not h:
            # access denied means it exists; anything else, that it does not
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == 259                  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner(wdir):
    """The lock's contents, or None when the directory is free."""
    try:
        with open(os.path.join(wdir, NAME), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _stale(info):
    if not info:
        return True
    if info.get("host") != socket.gethostname():
        return False          # cannot see another machine's processes
    return not _alive(int(info.get("pid", -1)))


def acquire(wdir, role):
    """
    Take the directory for this process, or raise Busy.

    Released at exit. Taking a lock this process already holds is a no-op.
    """
    os.makedirs(wdir, exist_ok=True)
    path = os.path.join(wdir, NAME)
    me = {"pid": os.getpid(), "host": socket.gethostname(), "role": role,
          "since": time.strftime("%Y-%m-%d %H:%M:%S")}
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = owner(wdir)
            if info and info.get("pid") == me["pid"] \
                    and info.get("host") == me["host"]:
                return path
            if not _stale(info):
                raise Busy(path, info)
            try:
                os.remove(path)          # a dead owner's; take it over
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(me, f)
        atexit.register(release, wdir)
        return path
    raise Busy(path, owner(wdir) or {})


def release(wdir):
    """Give the directory back, if this process holds it."""
    info = owner(wdir)
    if info and info.get("pid") == os.getpid() \
            and info.get("host") == socket.gethostname():
        try:
            os.remove(os.path.join(wdir, NAME))
        except OSError:
            pass
