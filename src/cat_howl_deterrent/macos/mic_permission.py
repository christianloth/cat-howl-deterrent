"""macOS microphone-permission helpers (via pyobjc/AVFoundation).

The detector hangs silently if python doesn't have TCC permission for the
microphone. We surface that explicitly here: trigger the macOS dialog from
python itself so the permission is tied to the python.framework binary.

Importing this module requires pyobjc — the package's `[macos]` extra.
"""

from __future__ import annotations

import sys
import threading

try:
    from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "pyobjc-framework-AVFoundation is required. "
        "Install with: pip install pyobjc-framework-AVFoundation pyobjc-core"
    ) from e


_STATUS_NAMES = {
    0: "notDetermined",
    1: "restricted",
    2: "denied",
    3: "authorized",
}


def status() -> str:
    """Return the current microphone authorization status name."""
    return _STATUS_NAMES.get(
        int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)),
        "unknown",
    )


def request(timeout_seconds: float = 180.0) -> bool:
    """Trigger the macOS mic permission dialog if needed.

    Returns True if authorized after the request, False otherwise.
    Must be called from a UI-attached process (e.g. via Terminal.app)
    for the dialog to actually appear.
    """
    s = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio))
    if s == 3:
        return True
    if s in (1, 2):
        return False

    done = threading.Event()
    granted_ref = [False]

    def _cb(granted):
        granted_ref[0] = bool(granted)
        done.set()

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeAudio, _cb
    )
    if not done.wait(timeout=timeout_seconds):
        return False
    return granted_ref[0]


def cli() -> int:
    """Console entry: print status, request access, exit 0 if authorized."""
    print(f"[mic] initial status: {status()}", flush=True)
    if status() == "authorized":
        return 0
    if status() in ("restricted", "denied"):
        print(
            "[mic] DENIED — enable in System Settings → Privacy & Security → Microphone.",
            flush=True,
        )
        return 2
    print("[mic] requesting access (dialog should appear — click Allow)", flush=True)
    ok = request()
    print(f"[mic] final status: {status()} granted={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(cli())
