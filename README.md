# 🎓 RAG-based AI Teaching Assistant (v2)

Ask questions about lecture recordings and get grounded answers with exact timestamps.
Upload video/audio or paste a YouTube link → Whisper transcription → timestamp-aware chunking →
hybrid retrieval (dense + BM25) → reranking → GPT-5 answer with inline citations → jump-to-timestamp playback.

## What changed in v2

| Area | v1 | v2 |
|---|---|---|
| Chunking | raw Whisper segments (2–8 s) | merged ~45 s windows with 10 s overlap |
| Retrieval | dense top-5 | dense + BM25 → reciprocal-rank fusion → reranker → top-5 |
| Unanswerable questions | always answered | relevance threshold; says "not in the lectures" |
| Citations | top-1 timestamp only | every source shown with timestamp, score, and player |
| Transcription | `translations` (forced English), 25 MB limit | `transcriptions`, auto-split of long audio |
| Summaries | full transcript in one prompt, regenerated every click | map-reduce for long lectures, cached on disk |
| Code | one 1,300-line `app.py` | `rag_ta/` package, thin UI, typed models, config via env |
| Quality | — | 23 unit tests, RAGAS eval harness, ruff, GitHub Actions, Docker |

## Quick start

```bash
git clone <repo> && cd <repo>
pip install -r requirements.txt          # needs ffmpeg on PATH
cp .env.example .env                     # add OPENAI_API_KEY
streamlit run app.py
```

Docker:

```bash
cp .env.example .env && docker compose up --build   # http://localhost:8501
```

## Architecture

```
rag_ta/
├── config.py            # all tunables, overridable via RAG_* env vars
├── models.py            # Segment, Chunk, RetrievedChunk
├── store.py             # Chroma persistent client
├── ingestion/
│   ├── media.py         # ffmpeg: extract audio, split long files
│   ├── transcribe.py    # Whisper with timestamp offsets across splits
│   ├── chunking.py      # timestamp-aware merge + overlap
│   └── indexer.py       # ingest_video / ingest_audio / reindex / delete
├── retrieval/
│   ├── embeddings.py    # OpenAI embeddings, batched
│   ├── sparse.py        # BM25 index over the same chunks
│   ├── fusion.py        # reciprocal rank fusion
│   ├── rerank.py        # LLM / FlashRank / none
│   └── retriever.py     # HybridRetriever orchestration
└── llm/
    ├── client.py        # OpenAI client with timeout + retries
    ├── prompts.py       # all prompts, versioned in one place
    ├── answer.py        # grounded answer with [S#] citations
    └── summarize.py     # map-reduce summaries + cache
app.py                   # Streamlit UI
tests/                   # pytest, no API key needed
eval/run_eval.py         # RAGAS: faithfulness, answer relevancy, context precision/recall
```

### Retrieval pipeline

```
query ──► embed ──► Chroma top-20 ─┐
                                   ├─► RRF (k=60) ─► top-15 ─► reranker ─► top-5 ─► threshold ─► LLM
query ──► BM25  ──► top-20 ────────┘
```

* **Why hybrid?** Embeddings handle paraphrase ("penalty that zeroes weights" → lasso); BM25 handles exact
  tokens ("Kadane", "L2", variable names). RRF merges them without score normalisation.
* **Why rerank?** A cross-encoder / LLM judge reads query and passage *together*, which is far more accurate
  than cosine similarity but too slow for the full corpus — so it runs on ~15 candidates only.
* **Threshold:** if the best reranked score is below `RAG_MIN_RELEVANCE`, the assistant says the answer isn't
  in the indexed lectures instead of hallucinating.

Reranker backends (`RAG_RERANKER`): `llm` (default, no extra deps), `flashrank` (local ONNX cross-encoder,
`pip install -r requirements-optional.txt`), `none`.

## Configuration

Every knob is an env var — see `.env.example`. Common ones:

| Var | Default | Notes |
|---|---|---|
| `RAG_CHAT_MODEL` | `gpt-5` | |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-large` | changing this requires re-indexing |
| `RAG_WHISPER_LANGUAGE` | *(auto)* | `hi`, `en`, … |
| `RAG_CHUNK_SECONDS` / `RAG_CHUNK_OVERLAP` | `45` / `10` | re-index after changing |
| `RAG_FINAL_TOP_K` | `5` | |
| `RAG_MIN_RELEVANCE` | `0.15` | 0–1 |
| `RAG_RERANKER` | `llm` | `llm` \| `flashrank` \| `none` |

## Evaluation

```bash
pip install -r requirements-eval.txt
cp eval/dataset.example.jsonl eval/dataset.jsonl   # add your own Q/A pairs
python -m eval.run_eval --dataset eval/dataset.jsonl
```

Prints faithfulness, answer relevancy, context precision, context recall and latency; saves a JSON under
`eval/results/` with the config used. Run before and after any retrieval or prompt change.

## Development

```bash
pip install -r requirements-dev.txt
make test    # pytest (fakes for OpenAI; real Chroma in a temp dir)
make lint    # ruff
```

CI runs lint + tests on Python 3.11/3.12 and builds the Docker image on every push and PR.

## Migrating from v1

Old `jsons/*.json` transcripts are still readable. Re-index them with the new chunker:

```bash
python scripts/migrate_old_jsons.py path/to/old/jsons
```

Media files, the Chroma store, and transcripts now live under `RAG_DATA_DIR` (default `~/rag_data`) and are
git-ignored — don't commit them.
