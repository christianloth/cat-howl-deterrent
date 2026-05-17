"""Console entry point: `cat-howl-deterrent` or `python -m cat_howl_deterrent`."""

from __future__ import annotations

import sys

from cat_howl_deterrent.config import RuntimeConfig
from cat_howl_deterrent.detector import run


def main() -> int:
    cfg = RuntimeConfig.from_env()
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
