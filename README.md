<div align="center">

# 🎓 RAG-based AI Teaching Assistant

**Ask a question about any lecture video and get an answer with the exact timestamp where it's explained.**

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://rag-based-ai-teaching-akash.streamlit.app/)
[![CI](https://github.com/akashnegetive/RAG-based-AI-Teaching-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/akashnegetive/RAG-based-AI-Teaching-Assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### 👉 **[Try the live app](https://rag-based-ai-teaching-akash.streamlit.app/)**

</div>

---

## The problem

You watched a 90-minute lecture last week. Today you need the one part where the professor explained
the bias–variance tradeoff. Scrubbing through the video to find it wastes ten minutes — and YouTube's
search only matches the title, not what was actually said.

## What this does

Upload a lecture (video, audio, or a YouTube link). The app transcribes it, indexes what was said
along with when it was said, and then answers your questions in plain language — citing the exact
minute and second, with a player that jumps straight there.

**In one sentence:** it turns lecture recordings into a searchable, question-answerable knowledge base.

| Ask this | Get this |
|---|---|
| *"At what timestamp is the bias–variance tradeoff explained?"* | An explanation plus `12:04–12:49`, with the video cued to 12:04 |
| *"What's the difference between ridge and lasso?"* | A grounded comparison, citing the two passages it came from |
| *"Summarise this lecture for tomorrow's exam"* | A quick summary and full study notes, downloadable as PDF |
| *"What is quantum chromodynamics?"* | *"I couldn't find this in the indexed lectures"* — it doesn't make things up |

---

## Screenshots

> Replace each placeholder below with your own screenshot. See [How to add screenshots](#how-to-add-screenshots).

### 1. Home — ask a question across all lectures

![Home page](docs/screenshots/01-home.png)

The main screen. Pick a scope in the sidebar (one lecture or all of them), type a question in plain
English, and hit **Search**. Query history is kept so you can re-run earlier questions with one click.

---

### 2. Ingesting a lecture

![Upload and ingest](docs/screenshots/02-upload.png)

Three ways in: paste a **YouTube URL**, upload an **MP4**, or upload an **MP3/M4A/WAV**. The status
panel shows each stage as it runs — audio extraction → Whisper transcription → chunking → embedding —
so a long lecture never looks frozen. Duplicate titles are rejected before any API cost is incurred.

---

### 3. Answer with timestamp grounding

![Answer with citations](docs/screenshots/03-answer.png)

The answer cites its sources inline as `[S1]`, `[S2]`. Below it, every retrieved passage is listed with
its lecture title, timestamp range, relevance score, and which retriever found it (`dense`, `sparse`, or
both) — so you can verify the answer rather than trusting it.

---

### 4. Jump-to-timestamp playback

![Video playback at timestamp](docs/screenshots/04-playback.png)

Expand any source and the video (or audio) player is already cued to that moment. No scrubbing.

---

### 5. Lecture summaries and study notes

![Summary panel](docs/screenshots/05-summary.png)

Select a single lecture and generate two summaries at once: a **quick summary** (120–180 words, for the
night before an exam) and **detailed notes** (overview, key concepts, topic flow, definitions, examples,
10-line revision list). Both download as PDF. Summaries are cached, so re-opening one is instant and free.

---

### 6. Managing the library

![Library management](docs/screenshots/06-library.png)

Every indexed lecture is listed in the sidebar. **Re-index** re-chunks a lecture from its saved
transcript (useful after changing chunk settings) without paying for transcription again. **Delete**
removes its vectors and media, behind a confirmation step.

---

## How it works

### The user's journey

```mermaid
flowchart LR
    A["📥 Add a lecture<br/>YouTube, MP4 or MP3"] --> B["⚙️ App indexes it<br/>one-time, ~1 min per 10 min"]
    B --> C["❓ Ask a question<br/>in plain English"]
    C --> D["✅ Get an answer<br/>+ exact timestamp"]
    D --> E["▶️ Jump to that moment<br/>in the player"]
```

### Under the hood

```mermaid
flowchart TB
    subgraph ingest ["Ingestion — runs once per lecture"]
        V["Video / audio / YouTube"] --> F["ffmpeg<br/><i>extract mono 16 kHz audio</i>"]
        F --> W["Whisper<br/><i>transcribe with timestamps</i>"]
        W --> CH["Chunker<br/><i>~45 s windows, 10 s overlap</i>"]
        CH --> EM["OpenAI embeddings"]
        EM --> DB[("ChromaDB<br/><i>vectors + timestamps</i>")]
        CH --> BM[("BM25 index<br/><i>keywords</i>")]
    end

    subgraph query ["Query — runs per question"]
        Q["User question"] --> QD["Dense search<br/><i>top 20</i>"]
        Q --> QS["Keyword search<br/><i>top 20</i>"]
        DB -.-> QD
        BM -.-> QS
        QD --> RRF["Reciprocal rank fusion<br/><i>merge both rankings → top 15</i>"]
        QS --> RRF
        RRF --> RR["Reranker<br/><i>reads query + passage together → top 5</i>"]
        RR --> TH{"Relevance<br/>above threshold?"}
        TH -->|no| NA["“Not in the indexed lectures”"]
        TH -->|yes| GEN["GPT-5<br/><i>grounded answer with [S#] citations</i>"]
        GEN --> OUT["Answer + sources + cued player"]
    end
```

**Why two searches instead of one?** Embeddings are good at meaning — they match *"the penalty that
zeroes out weights"* to a passage about lasso even with no shared words. But they're weak on exact
tokens: names like *Kadane*, symbols like *L2*, specific variable names. BM25 keyword search covers
exactly that gap. Reciprocal rank fusion merges the two rankings by position, so BM25 scores and cosine
distances never have to be made comparable.

**Why rerank?** A reranker reads the question and a passage *together* and judges relevance directly,
which is far more accurate than comparing two independently-computed vectors. It's too slow to run over
a whole corpus — so the cheap searches narrow things to ~15 candidates first, and the reranker picks the
best 5 from those.

**Why a threshold?** Without one, a question the lectures never cover still returns five passages and
the model dutifully writes a confident answer from them. With it, the app says it doesn't know. For a
study tool, that matters more than coverage.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Fast to build, deploys free, good enough for a single-user study tool |
| Transcription | OpenAI Whisper (`whisper-1`) | Segment-level timestamps out of the box; auto-splits files over the API's 25 MB limit |
| Embeddings | `text-embedding-3-large` | Strong retrieval quality; batched to cut request count |
| Vector store | ChromaDB | Embedded, no server to run, metadata filtering for per-lecture scoping |
| Keyword search | `rank-bm25` | Pure Python, no infrastructure, covers the exact-token gap |
| Reranking | LLM judge (default) or FlashRank | LLM needs no extra deps; FlashRank runs locally and free on CPU |
| Answers | GPT-5 | Grounded generation with inline citations |
| Quality | pytest, ruff, RAGAS, GitHub Actions, Docker | Retrieval changes are measured, not guessed at |

---

## Quick start

```bash
git clone https://github.com/akashnegetive/RAG-based-AI-Teaching-Assistant.git
cd RAG-based-AI-Teaching-Assistant

pip install -r requirements.txt     # needs ffmpeg on PATH
cp .env.example .env                # add your OPENAI_API_KEY
streamlit run app.py                # → http://localhost:8501
```

With Docker (includes ffmpeg, persists data in a volume):

```bash
cp .env.example .env
docker compose up --build           # → http://localhost:8501
```

---

## Project structure

```
rag_ta/
├── config.py            # every tunable, overridable via RAG_* env vars
├── models.py            # Segment, Chunk, RetrievedChunk
├── store.py             # Chroma persistent client
├── ingestion/
│   ├── media.py         # ffmpeg: extract audio, split long files
│   ├── transcribe.py    # Whisper with timestamp offsets across splits
│   ├── chunking.py      # timestamp-aware merge + overlap
│   └── indexer.py       # ingest / reindex / delete
├── retrieval/
│   ├── embeddings.py    # batched OpenAI embeddings
│   ├── sparse.py        # BM25 index
│   ├── fusion.py        # reciprocal rank fusion
│   ├── rerank.py        # LLM / FlashRank / none
│   └── retriever.py     # HybridRetriever orchestration
└── llm/
    ├── client.py        # OpenAI client with timeouts + retries
    ├── prompts.py       # all prompts, versioned in one place
    ├── answer.py        # grounded answer with [S#] citations
    └── summarize.py     # map-reduce summaries + disk cache
app.py                   # Streamlit UI (thin — no business logic)
tests/                   # pytest; runs without an API key
eval/run_eval.py         # RAGAS evaluation harness
```

---

## Configuration

Everything is an environment variable — see `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `RAG_CHAT_MODEL` | `gpt-5` | |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-large` | changing this requires re-indexing |
| `RAG_WHISPER_LANGUAGE` | *(auto-detect)* | pin with `hi`, `en`, … |
| `RAG_CHUNK_SECONDS` / `RAG_CHUNK_OVERLAP` | `45` / `10` | re-index after changing |
| `RAG_FINAL_TOP_K` | `5` | passages sent to the model |
| `RAG_MIN_RELEVANCE` | `0.15` | 0–1; raise it to make the app say "I don't know" more often |
| `RAG_RERANKER` | `llm` | `llm` \| `flashrank` \| `none` |

---

# Evaluation

This directory contains the evaluation benchmark and runner for measuring the RAG system.

## What is being evaluated?

The evaluation measures retrieval and answer quality using RAGAS.

The benchmark currently contains **25 transcript-grounded questions** covering five lecture transcripts, with five questions per lecture.

The evaluation reports:

- **Faithfulness** — whether the generated answer is supported by the retrieved context.
- **Answer relevancy** — whether the answer addresses the question.
- **Context precision** — how much of the retrieved context is relevant to the question.
- **Context recall** — whether the retrieved context contains the information needed to answer the question.
- **Average latency** — average time taken to process a question.

RAGAS scores are comparative evaluation signals for this benchmark, not general claims of model accuracy.

## Files

```text
eval/
├── dataset.example.jsonl
├── dataset.jsonl
├── run_eval.py
├── generate_dataset.py
├── README.md
└── results/
```

- `dataset.example.jsonl` — example dataset showing the expected format.
- `dataset.jsonl` — current 25-question benchmark.
- `run_eval.py` — runs the RAG pipeline and evaluates the generated answers with RAGAS.
- `generate_dataset.py` — generates transcript-grounded benchmark questions.
- `results/` — JSON reports produced by evaluation runs.

## Dataset format

Each line in the dataset is a JSON object:

```json
{
  "question": "What are the four approaches described in the lecture?",
  "ground_truth": "Recursive backtracking, top-down DP with memoization, bottom-up DP with tabulation, and bottom-up no-memory DP.",
  "lecture": "4_Steps_to_Solve_Any_Dynamic_Programming_DP_Problem_144P"
}
```

The current benchmark covers:

1. Ridge and Lasso Regression
2. Bias and Variance
3. Maximum Subarray / Kadane's Algorithm
4. Dynamic Programming
5. Post-Pruning and Pre-Pruning

## Running the evaluation

Install the evaluation dependencies:

```bash
pip install -r requirements-eval.txt
```

Make sure the lecture corpus has already been indexed locally and that the required API keys are available in `.env`.

Then run:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl
```

For lecture-level retrieval scoping:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --scope-by-lecture
```

Each run saves a JSON report under `eval/results/`.

## Baseline vs V2

The same 25-question benchmark was run twice to measure the effect of the shipped LLM reranker.

Baseline:

```powershell
$env:RAG_RERANKER = "none"
python -m eval.run_eval --dataset eval/dataset.jsonl
```

V2:

```powershell
$env:RAG_RERANKER = "llm"
python -m eval.run_eval --dataset eval/dataset.jsonl
```

Both runs use the same corpus, embeddings, chat model, final top-k setting, and benchmark questions.

### Results

| Metric | Baseline | V2 | Change |
|---|---:|---:|---:|
| Faithfulness | 0.834 | 0.652 | -0.182 |
| Answer relevancy | 0.253 | **0.613** | **+0.360** |
| Context precision | 0.231 | **0.543** | **+0.312** |
| Context recall | 0.497 | **0.660** | **+0.163** |
| Avg. latency | **9.49s** | 17.05s | +7.56s |

On this benchmark, V2 substantially improves answer relevancy, context precision, and context recall. The trade-off is lower measured faithfulness and higher average latency.

These results should be interpreted as a **retrieval-quality trade-off**, not as evidence that V2 improves every evaluation metric.

## Reproducibility

Each evaluation result records the configuration used for the run.

For a fair comparison:

- use the same benchmark questions;
- use the same lecture corpus;
- keep the embedding model unchanged;
- keep the chat model unchanged;
- keep `RAG_FINAL_TOP_K` unchanged;
- change only the retrieval component being evaluated.

Do not compare scores from different datasets as if they were a controlled before/after experiment.

## Limitations

This benchmark has several limitations:

- It contains only 25 questions.
- The questions were generated from lecture transcripts rather than manually validated by an expert.
- The benchmark covers five lectures and may not represent the full corpus.
- RAGAS metrics can vary across runs because language models are involved in generation and evaluation.
- Latency depends on model/API response time and the local environment.
- The results should not be interpreted as a universal measure of RAG quality.

The purpose of this evaluation is to provide a reproducible, quantitative way to compare retrieval configurations during development.


---

## Development

```bash
pip install -r requirements-dev.txt
make test     # pytest — uses fakes for OpenAI, a temp Chroma dir for the store
make lint     # ruff
```

CI runs lint and tests on Python 3.11 and 3.12 and builds the Docker image on every push and PR.

---

## What changed in v2

| Area | v1 | v2 |
|---|---|---|
| Chunking | raw Whisper segments (2–8 s) | merged ~45 s windows with 10 s overlap |
| Retrieval | dense top-5 only | dense + BM25 → RRF → reranker → top-5 |
| Unanswerable questions | always answered something | relevance threshold; says it doesn't know |
| Citations | top-1 timestamp only | every source with timestamp, score, and player |
| Transcription | `translations` (forced English), 25 MB cap | `transcriptions`, auto-splits long audio |
| Summaries | whole transcript in one prompt, regenerated each click | map-reduce for long lectures, cached |
| Code | one 1,300-line `app.py` | `rag_ta/` package, thin UI, typed models, env config |
| Quality | — | 29 unit tests, RAGAS harness, ruff, CI, Docker |

### Migrating from v1

Old `jsons/*.json` transcripts still load. Re-index them with the new chunker — no re-transcription cost:

```bash
python scripts/migrate_old_jsons.py path/to/old/jsons
```

---

## Known limitations

- **YouTube ingestion can fail** with a 403 on hosted environments — YouTube blocks datacenter IPs and
  recent `yt-dlp` needs a JavaScript runtime. Download the video and upload it instead.
- **Streamlit Cloud storage is ephemeral**: indexed lectures are lost on redeploy. Run the Docker image
  on a VM with a persistent volume if you need them to stick around.
- **Transcription quality sets the ceiling.** Heavy accents, poor audio, or dense notation (formulas read
  aloud) degrade retrieval, because the app can only find what Whisper heard correctly.

---

## How to add screenshots

1. Run the app and capture each of the six views listed above.
2. Create the folder and drop them in with these exact names:

```bash
mkdir -p docs/screenshots
# 01-home.png  02-upload.png  03-answer.png
# 04-playback.png  05-summary.png  06-library.png
```

3. Commit and push:

```bash
git add docs/screenshots README.md
git commit -m "docs: add screenshots and architecture walkthrough"
git push
```

**Tips for good screenshots:** capture the browser content area only (not the whole desktop), use a
window around 1400px wide, keep light mode for readability on GitHub, and make sure a real answer with
real timestamps is on screen — an empty state teaches the reader nothing. For screenshot 3, expand one
source card so the citation, score, and retriever tags are all visible.

---

## License

MIT
