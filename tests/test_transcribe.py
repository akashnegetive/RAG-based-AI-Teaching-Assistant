"""The segment parser must handle both SDK objects and plain dicts, with the correct offset."""
from dataclasses import dataclass

import pytest

from rag_ta.config import Settings
from rag_ta.ingestion import transcribe as T


@dataclass
class FakeSegment:  # mirrors openai TranscriptionSegment: attributes, not subscriptable
    start: float
    end: float
    text: str


class FakeResponse:
    def __init__(self, segments):
        self.segments = segments


@pytest.fixture
def patched(monkeypatch, tmp_path):
    holder = {}

    class FakeAudio:
        class transcriptions:
            @staticmethod
            def create(**kwargs):
                holder["kwargs"] = kwargs
                return holder["response"]

    class FakeClient:
        audio = FakeAudio()

    monkeypatch.setattr(T, "get_openai", lambda cfg: FakeClient())
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    return holder, f


def test_parses_sdk_objects(patched):
    holder, f = patched
    holder["response"] = FakeResponse([FakeSegment(0.0, 2.5, " hello "), FakeSegment(2.5, 5.0, "world")])
    segs = T._transcribe_file(f, 0.0, Settings())
    assert [(s.start, s.end, s.text) for s in segs] == [(0.0, 2.5, "hello"), (2.5, 5.0, "world")]


def test_parses_dicts(patched):
    holder, f = patched
    holder["response"] = FakeResponse([{"start": 1.0, "end": 2.0, "text": "hi"}])
    assert T._transcribe_file(f, 0.0, Settings())[0].text == "hi"


def test_offset_applied_for_split_audio(patched):
    holder, f = patched
    holder["response"] = FakeResponse([FakeSegment(0.0, 3.0, "part two")])
    seg = T._transcribe_file(f, 600.0, Settings())[0]
    assert (seg.start, seg.end) == (600.0, 603.0)


def test_missing_timestamps_are_skipped(patched):
    holder, f = patched
    holder["response"] = FakeResponse([FakeSegment(None, 1.0, "bad"), FakeSegment(0.0, 1.0, "good")])
    segs = T._transcribe_file(f, 0.0, Settings())
    assert len(segs) == 1 and segs[0].text == "good"


def test_empty_response(patched):
    holder, f = patched
    holder["response"] = FakeResponse([])
    assert T._transcribe_file(f, 0.0, Settings()) == []


def test_language_passed_only_when_set(patched):
    holder, f = patched
    holder["response"] = FakeResponse([])
    T._transcribe_file(f, 0.0, Settings())
    assert "language" not in holder["kwargs"]
    T._transcribe_file(f, 0.0, Settings(whisper_language="hi"))
    assert holder["kwargs"]["language"] == "hi"
