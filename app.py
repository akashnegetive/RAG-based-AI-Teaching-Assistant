"""Streamlit UI — a thin layer over the `rag_ta` package."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_ta.config import settings
from rag_ta.ingestion.indexer import delete_lecture, ingest_audio, ingest_video, lecture_exists, reindex_from_transcript
from rag_ta.llm.answer import generate_answer
from rag_ta.llm.summarize import summarize
from rag_ta.logging_config import setup_logging
from rag_ta.pdf import markdown_to_pdf
from rag_ta.retrieval.retriever import HybridRetriever
from rag_ta.store import get_collection, list_titles

log = setup_logging()
settings.ensure_dirs()
st.set_page_config(page_title="RAG Teaching Assistant", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(settings)


def invalidate_indexes() -> None:
    get_retriever.clear()


def download_youtube(url: str) -> Path:
    import yt_dlp

    opts = {
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "outtmpl": str(settings.videos_dir / "%(title).80s.%(ext)s"),
        "merge_output_format": "mp4",
        "restrictfilenames": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


def run_ingest(fn, path: Path) -> None:
    with st.status("Processing lecture…", expanded=True) as status:
        try:
            n = fn(path, settings, progress=status.write)
        except Exception as e:  # noqa: BLE001
            log.exception("Ingestion failed for %s", path)
            status.update(label="Ingestion failed", state="error")
            st.error(str(e))
            return
        status.update(label=f"Indexed {n} chunks from {path.stem}", state="complete")
    invalidate_indexes()
    st.session_state["flash"] = f"✅ **{path.stem}** added to the knowledge base ({n} chunks)."
    st.rerun()


# --------------------------------------------------------------------------- header
st.markdown(
    """<div style="padding:22px;border-radius:14px;background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;margin-bottom:18px;"><h2 style="margin:0">🎓 RAG-based AI Teaching Assistant</h2>
    <p style="color:#cfd8dc;margin:6px 0 0">Hybrid retrieval (dense + BM25) → reranking → grounded answers with lecture timestamps.</p></div>""",
    unsafe_allow_html=True,
)
if msg := st.session_state.pop("flash", None):
    st.success(msg)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📥 Add a lecture")

    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
    if st.button("Download & index", use_container_width=True, disabled=not yt_url.strip()):
        try:
            with st.status("Downloading from YouTube…"):
                video_path = download_youtube(yt_url.strip())
        except Exception as e:  # noqa: BLE001
            st.error(f"Download failed: {e}")
        else:
            run_ingest(ingest_video, video_path)

    up = st.file_uploader("Or upload a file", type=["mp4", "mp3", "m4a", "wav"])
    if up is not None:
        title = Path(up.name).stem
        if lecture_exists(title, settings):
            st.warning(f"'{title}' is already indexed. Delete it first to re-upload.")
        elif st.button("Index uploaded file", use_container_width=True):
            is_video = up.name.lower().endswith(".mp4")
            dest = (settings.videos_dir if is_video else settings.audios_dir) / up.name
            dest.write_bytes(up.getbuffer())
            run_ingest(ingest_video if is_video else ingest_audio, dest)

    st.divider()
    st.header("📚 Lecture library")
    titles = list_titles(settings)
    scope = st.selectbox("Search scope", ["All lectures"] + titles)
    selected = None if scope == "All lectures" else scope

    if selected:
        c1, c2 = st.columns(2)
        if c1.button("🔄 Re-index", use_container_width=True, help="Re-chunk & re-embed with current settings"):
            with st.spinner("Re-indexing…"):
                n = reindex_from_transcript(selected, settings)
            invalidate_indexes()
            st.session_state["flash"] = f"Re-indexed **{selected}** ({n} chunks)."
            st.rerun()
        if c2.button("🗑 Delete", use_container_width=True, type="primary"):
            st.session_state["confirm_delete"] = selected
        if st.session_state.get("confirm_delete") == selected:
            st.warning(f"Permanently delete **{selected}** and its media?")
            if st.button("Yes, delete it", use_container_width=True):
                delete_lecture(selected, settings)
                invalidate_indexes()
                st.session_state.pop("confirm_delete", None)
                st.session_state["flash"] = f"Deleted **{selected}**."
                st.rerun()

    with st.expander("⚙️ Retrieval settings", expanded=False):
        st.caption(
            f"Chat: `{settings.chat_model}` · Embeddings: `{settings.embedding_model}` · "
            f"Reranker: `{settings.reranker}`\n\n"
            f"Chunks ≈{settings.chunk_target_seconds:.0f}s with {settings.chunk_overlap_seconds:.0f}s overlap · "
            f"top-{settings.final_top_k} of {settings.rerank_top_k} candidates · "
            f"min relevance {settings.min_relevance}"
        )
        st.caption(f"Indexed chunks: {get_collection(settings).count()}")

# --------------------------------------------------------------------------- summary panel
if selected:
    with st.expander(f"📘 Lecture summary — {selected}", expanded=False):
        force = st.checkbox("Regenerate (ignore cache)", value=False)
        if st.button("Generate summary"):
            with st.spinner("Summarising…"):
                st.session_state["summary"] = (selected, summarize(selected, settings, force=force))
        cached = st.session_state.get("summary")
        if cached and cached[0] == selected:
            _, summ = cached
            tab_q, tab_d = st.tabs(["⚡ Quick summary", "📚 Detailed notes"])
            with tab_q:
                st.markdown(summ.quick)
                st.download_button(
                    "Download PDF",
                    markdown_to_pdf(f"{selected} – Quick summary", summ.quick),
                    file_name=f"{selected}_quick_summary.pdf",
                    mime="application/pdf",
                )
            with tab_d:
                st.markdown(summ.detailed)
                st.download_button(
                    "Download PDF",
                    markdown_to_pdf(f"{selected} – Detailed notes", summ.detailed),
                    file_name=f"{selected}_detailed_notes.pdf",
                    mime="application/pdf",
                    key="dl_detailed",
                )

# --------------------------------------------------------------------------- Q&A
st.subheader("🔎 Ask a question")
st.caption(
    f"Searching {'**' + selected + '**' if selected else 'across **all lectures**'}. "
    "Example: *At what timestamp is the bias–variance trade-off explained?*"
)

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Question", placeholder="Type your question…", label_visibility="collapsed")
if st.button("Search", type="primary") and query.strip():
    if get_collection(settings).count() == 0:
        st.info("No lectures indexed yet. Add one from the sidebar.")
        st.stop()
    if query not in st.session_state.history:
        st.session_state.history.insert(0, query)

    with st.status("Retrieving…", expanded=False) as status:
        result = get_retriever().retrieve(query, title=selected)
        status.write(
            f"Considered {result.candidates_considered} candidates from dense + keyword search, "
            f"reranked to top {len(result.chunks)}."
        )
        status.update(label="Generating answer…")
        answer = generate_answer(query, result.chunks, result.answerable, settings)
        status.update(label="Done", state="complete")

    st.markdown("### 🤖 Answer")
    (st.info if answer.grounded else st.warning)(answer.text)

    if result.chunks:
        st.markdown("### 📍 Sources")
        for i, c in enumerate(result.chunks, start=1):
            with st.expander(
                f"[S{i}] {c.title} @ {c.timestamp}  ·  relevance {c.score:.2f}  ·  {'+'.join(c.sources)}",
                expanded=(i == 1),
            ):
                st.write(c.text)
                media_v = settings.videos_dir / f"{c.title}.mp4"
                media_a = settings.audios_dir / f"{c.title}.mp3"
                if media_v.exists():
                    st.video(str(media_v), start_time=int(c.start))
                elif media_a.exists():
                    st.audio(str(media_a), start_time=int(c.start))

if st.session_state.history:
    with st.expander("🕘 Recent questions"):
        for q in st.session_state.history[:10]:
            st.write("•", q)
