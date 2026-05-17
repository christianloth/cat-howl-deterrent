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

from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.2.0"

# Project root contains .env (one dir up from src/cat_howl_deterrent/).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)
