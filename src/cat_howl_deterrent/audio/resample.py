"""Sample-rate conversion helpers.

USB mics (Yeti, etc.) often hang when PortAudio is asked to resample on
the fly. We open at native rate and resample in Python instead.
"""

from __future__ import annotations

import numpy as np
import scipy.signal


def make_resampler(native_sr: int, target_sr: int):
    """Return a callable that resamples 1D float32 arrays to target_sr.

    Identity when rates already match.
    """
    if native_sr == target_sr:
        def identity(x: np.ndarray) -> np.ndarray:
            return x.astype(np.float32, copy=False)
        return identity

    def resample(x: np.ndarray) -> np.ndarray:
        return scipy.signal.resample_poly(x, target_sr, native_sr).astype(np.float32)

    return resample
