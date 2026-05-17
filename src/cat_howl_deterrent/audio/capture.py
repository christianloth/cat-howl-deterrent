"""Microphone capture backends.

Two implementations, selected at runtime:

  * SoundDeviceMic — sounddevice/PortAudio (default). Reliable on a
    healthy macOS audio stack and on Linux. Hangs silently on macOS 26
    when coreaudiod is wedged (e.g. by Loopback's AudioServerPlugin).

  * FFmpegMic — spawns ffmpeg with AVFoundation. Useful as a fallback
    when PortAudio misbehaves. Requires ffmpeg with mic TCC permission.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd


def resolve_device_index(name_or_index: str | None) -> int | None:
    """Resolve a MIC_DEVICE env value (int or substring) to a PortAudio index."""
    if not name_or_index:
        return None
    try:
        return int(name_or_index)
    except ValueError:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_or_index.lower() in d["name"].lower():
            return i
    raise SystemExit(f"MIC_DEVICE={name_or_index!r} not found among input devices")


class _MicBase:
    """Common state for any mic backend: native rate, hop, callback counter."""

    def __init__(self, native_sr: int, hop_native: int):
        self.native_sr = native_sr
        self.hop_native = hop_native
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.callbacks_received = 0
        self.started_at = time.time()

    def start(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SoundDeviceMic(_MicBase):
    """sounddevice/PortAudio capture.

    On stereo devices (Yeti), opening channels=1 sometimes silently fails on
    macOS HAL. We open with the device's max input channels (capped at 2)
    and downmix to mono in the callback.
    """

    def __init__(self, device_index: int | None, native_sr: int, hop_native: int):
        super().__init__(native_sr, hop_native)
        self.device_index = device_index
        if device_index is not None:
            info = sd.query_devices(device_index)
            self.channels = min(2, int(info["max_input_channels"]))
        else:
            self.channels = 1
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        self.callbacks_received += 1
        if self.callbacks_received == 1:
            print(
                f"  · first mic callback: shape={indata.shape} dtype={indata.dtype}"
            )
        if status:
            print(f"  ! mic status: {status}")
        mono = indata.mean(axis=1) if indata.ndim == 2 else indata
        self.queue.put(mono.astype(np.float32, copy=False).copy())

    def start(self) -> None:
        print(
            f"  opening sd.InputStream channels={self.channels} "
            f"sr={self.native_sr} blocksize={self.hop_native}"
        )
        self._stream = sd.InputStream(
            samplerate=self.native_sr,
            channels=self.channels,
            blocksize=self.hop_native,
            callback=self._callback,
            device=self.device_index,
            dtype="float32",
        )
        self._stream.start()
        print("  sd.InputStream started")

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None


class FFmpegMic(_MicBase):
    """ffmpeg/AVFoundation capture (macOS fallback).

    Spawns an `ffmpeg -f avfoundation` process and reads raw float32 PCM
    from its stdout in a background thread. Requires ffmpeg to have mic
    TCC permission — homebrew's adhoc-signed ffmpeg may need manual grant.
    """

    def __init__(self, device_substring: str | None, native_sr: int, hop_native: int):
        super().__init__(native_sr, hop_native)
        self.device_substring = device_substring
        self._proc: subprocess.Popen | None = None

    def _resolve_avf_device(self) -> int:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        )
        in_audio = False
        first_idx: int | None = None
        for ln in probe.stderr.decode("utf-8", "ignore").splitlines():
            if "AVFoundation audio devices:" in ln:
                in_audio = True
                continue
            if not in_audio:
                continue
            m = re.search(r"\[(\d+)\]\s+(.*)", ln)
            if not m:
                continue
            idx, name = int(m.group(1)), m.group(2)
            if self.device_substring and self.device_substring.lower() in name.lower():
                return idx
            if first_idx is None:
                first_idx = idx
        if first_idx is None:
            raise SystemExit("ffmpeg: no avfoundation audio device found")
        return first_idx

    def start(self) -> None:
        av_idx = self._resolve_avf_device()
        print(f"  ffmpeg avfoundation audio device: [{av_idx}]")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "avfoundation",
            "-ar", str(self.native_sr), "-ac", "1",
            "-i", f":{av_idx}",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(self.native_sr),
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._stderr_pump, daemon=True).start()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout is not None
        bytes_per_chunk = self.hop_native * 4
        try:
            while True:
                buf = self._proc.stdout.read(bytes_per_chunk)
                if not buf or len(buf) < bytes_per_chunk:
                    print(f"  ! ffmpeg stream ended ({len(buf) if buf else 0} bytes)")
                    return
                self.callbacks_received += 1
                self.queue.put(np.frombuffer(buf, dtype=np.float32).copy())
        except Exception as e:
            print(f"  ! ffmpeg reader: {e}")

    def _stderr_pump(self) -> None:
        assert self._proc and self._proc.stderr is not None
        for ln in self._proc.stderr:
            s = ln.decode("utf-8", "ignore").rstrip()
            if s:
                print(f"  [ffmpeg] {s}")

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            self._proc = None


def make_mic_capture(backend: str, mic_device: str | None) -> _MicBase:
    """Factory: build the right mic backend and probe its native rate."""
    from cat_howl_deterrent.config import HOP_SECONDS, SAMPLE_RATE

    if backend == "sd":
        device_index = resolve_device_index(mic_device)
        if device_index is not None:
            info = sd.query_devices(device_index)
            native_sr = int(info["default_samplerate"])
            print(
                f"Mic: [{device_index}] {info['name']} "
                f"(in_ch={info['max_input_channels']}, sr={native_sr})"
            )
        else:
            default_in = (
                sd.default.device[0]
                if isinstance(sd.default.device, (list, tuple))
                else sd.default.device
            )
            try:
                info = sd.query_devices(default_in)
                native_sr = int(info["default_samplerate"])
                print(f"Mic: [{default_in}] {info['name']} (default)")
            except Exception:
                native_sr = SAMPLE_RATE
                print("Mic: system default")
        hop_native = int(round(native_sr * HOP_SECONDS))
        return SoundDeviceMic(device_index, native_sr, hop_native)

    if backend == "ffmpeg":
        native_sr = int(os.environ.get("MIC_NATIVE_SR", "48000"))
        hop_native = int(round(native_sr * HOP_SECONDS))
        print(f"Mic: ffmpeg/AVFoundation, native_sr={native_sr}")
        return FFmpegMic(mic_device, native_sr, hop_native)

    raise SystemExit(f"unknown MIC_BACKEND={backend!r}")
