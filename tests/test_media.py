import shutil
import subprocess

import pytest

from rag_ta.ingestion.media import audio_duration, split_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_split_audio_offsets(tmp_path):
    src = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=25", "-q:a", "9", str(src)],
        check=True,
        capture_output=True,
    )
    assert 24 < audio_duration(src) < 26
    parts = split_audio(src, segment_seconds=10, work_dir=tmp_path / "parts")
    assert len(parts) == 3
    assert [o for _, o in parts] == [0.0, 10.0, 20.0]
