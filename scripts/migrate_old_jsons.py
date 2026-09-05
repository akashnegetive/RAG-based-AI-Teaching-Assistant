"""One-off: re-index transcripts produced by the old pipeline (jsons/*.json) with the new chunker.

python scripts_migrate_old_jsons.py path/to/old/jsons
"""

import sys
from pathlib import Path

from rag_ta.config import settings
from rag_ta.ingestion.indexer import index_segments
from rag_ta.ingestion.transcribe import load_transcript, save_transcript

src = Path(sys.argv[1] if len(sys.argv) > 1 else "jsons")
for p in sorted(src.glob("*.json")):
    title, segs = load_transcript(p)
    title = title if title != p.stem else p.stem
    save_transcript(title, segs, settings.jsons_dir)
    n = index_segments(title, segs, settings)
    print(f"{title}: {len(segs)} segments -> {n} chunks")
