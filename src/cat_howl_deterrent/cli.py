"""Console entry point: `cat-howl-deterrent` or `python -m cat_howl_deterrent`."""

from __future__ import annotations

import os
import subprocess
import sys

from cat_howl_deterrent.config import RuntimeConfig
from cat_howl_deterrent.detector import run


def _start_caffeinate() -> subprocess.Popen | None:
    """Prevent macOS idle sleep while the detector runs.

    `-w <pid>` ties caffeinate's lifetime to ours, so it exits even if we are
    SIGKILLed and never get a chance to clean up.
    """
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-s", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"caffeinate: holding idle+system-sleep assertion (pid {proc.pid})")
        return proc
    except FileNotFoundError:
        print("caffeinate: binary not found, skipping sleep assertion")
        return None


def main() -> int:
    cfg = RuntimeConfig.from_env()
    caffeinate = _start_caffeinate()
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
