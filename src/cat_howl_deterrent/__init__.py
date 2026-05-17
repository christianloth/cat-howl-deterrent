"""Cat-howl detector + ultrasonic deterrent.

Top-level package. Most consumers should use the CLI entry point
(`cat-howl-deterrent` or `python -m cat_howl_deterrent`) rather than
importing internals directly.

Side-effect on import: loads a project-root `.env` file into `os.environ`
(if present) so that configuration knobs documented in `.env.example` are
picked up automatically. Existing environment variables take precedence,
so inline `FOO=bar cat-howl-deterrent` overrides still work.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.2.0"

# Force line-buffered stdout/stderr so detector progress shows up live in
# log files (the default is block-buffered when stdout is not a TTY, which
# means nohup-redirected logs would only flush every ~4 KB — making the
# detector look stuck during boot). Equivalent to running with `python -u`.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Project root contains .env (one dir up from src/cat_howl_deterrent/).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)
