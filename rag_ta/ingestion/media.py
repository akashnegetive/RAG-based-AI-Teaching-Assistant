"""ffmpeg helpers: extract audio from video, split long audio for the Whisper API."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class MediaError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise MediaError("ffmpeg not found on PATH. Install it (apt install ffmpeg / brew install ffmpeg).")


def _run(cmd: list[str]) -> None:
    log.debug("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(proc.stderr[-2000:])


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Video -> mono 16 kHz MP3 (small, ideal for Whisper)."""
    ensure_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video_path.stem}.mp3"
    _run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", "-b:a", "48k", str(out)])
    return out


def audio_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError as e:
        raise MediaError(f"Could not read duration of {path}: {proc.stderr}") from e


def split_audio(path: Path, segment_seconds: int, work_dir: Path) -> list[tuple[Path, float]]:
    """Split into fixed-length pieces. Returns [(piece_path, offset_seconds), ...].

    The Whisper API rejects files over 25 MB; 10 minutes of 48 kbps mono is ~3.6 MB, so
    600 s segments leave plenty of headroom.
    """
    ensure_ffmpeg()
    work_dir.mkdir(parents=True, exist_ok=True)
    pattern = work_dir / f"{path.stem}_part%03d.mp3"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "48k",
            str(pattern),
        ]
    )
    parts = sorted(work_dir.glob(f"{path.stem}_part*.mp3"))
    if not parts:
        raise MediaError("ffmpeg produced no segments")
    return [(p, i * float(segment_seconds)) for i, p in enumerate(parts)]
