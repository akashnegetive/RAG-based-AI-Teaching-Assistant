# ============================================================
# IMPORTS & CONFIGURATION
# ============================================================
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
import os
import subprocess
import json
import yt_dlp
import io
from openai import OpenAI
from dotenv import load_dotenv
from chroma_client import get_chroma
from preprocess_json_uploaded import embed_json_file
from fpdf import FPDF
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet



load_dotenv()

BASE_DATA_DIR = os.path.join(os.path.expanduser("~"), "rag_data")

VIDEOS_DIR = os.path.join(BASE_DATA_DIR, "videos")
AUDIOS_DIR = os.path.join(BASE_DATA_DIR, "audios")
JSONS_DIR  = os.path.join(BASE_DATA_DIR, "jsons")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(AUDIOS_DIR, exist_ok=True)
os.makedirs(JSONS_DIR, exist_ok=True)


client = OpenAI(api_key=os.getenv("api_key"))

st.set_page_config(page_title="RAG Video Assistant", layout="wide")




if "last_ingested" in st.session_state:
    st.success(st.session_state["last_ingested"])
    del st.session_state["last_ingested"]
    
if "lecture_summary_quick" not in st.session_state:
    st.session_state["lecture_summary_quick"] = None
    
if "lecture_summary_full" not in st.session_state:
    st.session_state["lecture_summary_full"] = None
    
# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>
/* Download PDF button */
a[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #020617, #1e293b);
    border: 1px solid #475569;
    color: #e5e7eb;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    transition: all 0.2s ease;
}

a[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-color: #60a5fa;
    box-shadow: 0 8px 24px rgba(96,165,250,0.35);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stToggle"] label {
    font-weight: 500;
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


    
  
# ============================================================
# PDF GENERATION HELPERS
# ============================================================




def generate_pdf_bytes(title, content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    for line in content.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer



# ============================================================
# SUMMARIZATION HELPERS
# ============================================================

def summarize_lecture_both(title):
    collection = st.session_state["collection"]

    # Fetch all chunks of this lecture
    results = collection.get(
        where={"title": title},
        include=["documents", "metadatas"]
    )

    # Sort by chunk number (correct lecture order)
    chunks = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda x: x[1]["number"]
    )

    full_text = "\n".join([c[0] for c in chunks])

    # -------- Quick Summary Prompt --------
    quick_prompt = f"""
    You are a senior university professor preparing executive revision notes.

    Task:
    Create a **1–2 minute executive summary** of the lecture for fast revision.

    Output Requirements:
    - 120–180 words (strict)
    - Make sure of this : Bullet points only (no paragraphs)
    - Capture only the most important concepts, principles, and conclusions
    - No derivations, no examples, no storytelling
    - Use precise academic terminology
    - Each bullet must be a complete, standalone idea
    - No repetition
    - No information not present in the transcript

    Style:
    - Concise
    - Exam-focused
    - Clear hierarchy of ideas
    - Professional academic tone

    Lecture Transcript:
    {full_text}
    """

    # -------- Detailed Notes Prompt --------
    full_prompt = f"""
    You are a senior AI Teaching Assistant preparing complete, exam-ready lecture notes.

    Task:
    Transform the following lecture transcript into **fully structured study material** suitable for university revision.

    Required Structure (use Markdown headings):

    1. **Lecture Title**
    2. **Executive Overview**  
    - 5–8 bullet points summarizing the full lecture
    3. **Key Concepts Explained**  
    - Each major concept with concise technical explanation
    4. **Step-by-Step Topic Flow**  
    - Ordered progression of ideas as taught in the lecture
    5. **Important Definitions**  
    - Clear, formal definitions of all core terms
    6. **Illustrative Examples** (only if present in transcript)  
    7. **Final 10-Line Revision Notes**  
    - Ultra-condensed exam-oriented takeaways

    Strict Rules:
    - Use only information present in the transcript
    - No hallucinations, no external knowledge
    - Academic, precise, and technical tone
    - Markdown formatting with clear section headers
    - Bullet points and numbered lists where appropriate
    - No verbosity, no storytelling, no filler
    - Each section must be logically coherent and complete

    Lecture Transcript:
    {full_text}
    """


    with st.status("🧠 Generating executive and detailed summaries..."):
        quick_summary = inference(quick_prompt)
        full_summary = inference(full_prompt)

    return quick_summary, full_summary


    
    
# ============================================================
# DELETE & RE-INDEX LECTURE HELPERS
# ============================================================

def delete_lecture(title):
    # 1. Delete from ChromaDB
    collection.delete(where={"title": title})

    # 2. Delete media files
    video_path = os.path.join(VIDEOS_DIR, title + ".mp4")
    audio_path = os.path.join(AUDIOS_DIR, title + ".mp3")
    json_path  = os.path.join(JSONS_DIR, title + ".json")


    for path in [video_path, audio_path, json_path]:
        if os.path.exists(path):
            os.remove(path)

    # Store message for next run
    st.session_state["delete_msg"] = f"🗑 Lecture '{title}' deleted successfully."
    st.session_state["reset_topic"] = True
    st.rerun()

def reindex_lecture(title):
    json_path = os.path.join(JSONS_DIR, title + ".json")

    # First delete old vectors
    collection.delete(where={"title": title})

    # Re-embed
    count = embed_json_file(json_path)
    st.success(f"🔄 Re-indexed {title} ({count} chunks)")


    
# ============================================================
# Chroma Embedding DB Client
# ============================================================


if "collection" not in st.session_state:
    _, st.session_state["collection"] = get_chroma()

collection = st.session_state["collection"]





# ============================================================
# EMBEDDING & LLM INFERENCE
# ============================================================

def create_embedding(text_lists):
    """Create vector embeddings for text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text_lists
    )
    return [item.embedding for item in response.data]


def inference(prompt):
    """Generate answer from LLM using retrieved context."""
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    return response.output_text

# ============================================================
# DOWNLOADING YOUTUBE VIDEO
# ============================================================

def download_youtube_video(url):
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(VIDEOS_DIR, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        return os.path.join(VIDEOS_DIR, f"{video_id}.mp4")



# ============================================================
# VIDEO INGESTION PIPELINE
# ============================================================

def process_video(video_path):
    """
    Full pipeline for uploaded video:
    Video → Audio → Whisper Transcription → JSON → Embeddings
    """

    title = os.path.splitext(os.path.basename(video_path))[0]

    # ---------- Step 1: Extract Audio ----------
    with st.status("🎧 Extracting audio from video...") as audio_status:
        try:
            audio_path = subprocess.check_output(
                [sys.executable, "video_to_audio.py", video_path],
                text=True
            ).strip()
        except subprocess.CalledProcessError as e:
            audio_status.update(label="❌ Audio extraction failed", state="error")
            st.code(e.output)
            return

    audio_status.update(label="Audio extracted successfully ✅", state="complete")

    # ---------- Step 2: Whisper Transcription ----------
    with st.status("⏳ Extracting Text & Timestamp from Audio...\n\nNote: Takes time for long videos") as whisper_status:
        result = subprocess.run(
            [sys.executable, "audio_to_json_uploaded.py", os.path.basename(audio_path)],
            capture_output=True,
            text=True
        )

    if result.returncode != 0:
        whisper_status.update(label="❌ Transcription failed", state="error")
        st.code(result.stderr)
        return

    whisper_status.update(label="JSON created with timestamps ✅", state="complete")
    json_path = result.stdout.strip()

    # ---------- Step 3: Embedding Generation ----------
    with st.status("🧠 Creating embeddings for semantic search...") as embed_status:
        count = embed_json_file(json_path)
        embed_status.update(label=f"Embeddings stored in vector DB ({count} chunks) ✅", state="complete")



    #st.cache_resource.clear()
    st.session_state["sidebar_notice_video"] = f"🎥 Video **{title}** added to the knowledge base successfully!"
    st.session_state["video_done"] = True
    st.session_state["reset_uploader"] = True
    st.rerun()
    



# ============================================================
# AUDIO INGESTION PIPELINE (No Video)
# ============================================================

def process_audio(audio_path):
    """
    Audio → Whisper → JSON → Embeddings
    """
    title = os.path.splitext(os.path.basename(audio_path))[0]
    with st.status("⏳ Extracting Text & Timestamp from Audio...") as status:
        result = subprocess.run(
            [sys.executable, "audio_to_json_uploaded.py", os.path.basename(audio_path)],
            capture_output=True,
            text=True
        )

    if result.returncode != 0:
        status.update(label="❌ Transcription failed", state="error")
        st.code(result.stderr)
        return

    status.update(label="JSON created ✅", state="complete")
    json_path = result.stdout.strip()

    with st.status("🧠 Creating embeddings for semantic search...") as embed_status:
        count = embed_json_file(json_path)
        embed_status.update(label=f"Embeddings stored in vector DB ({count} chunks) ✅", state="complete")

    
    # Persist success across rerun
    #st.cache_resource.clear()
    st.session_state["sidebar_notice_audio"] = f"🔊 Audio **{title}** added to the knowledge base successfully!"
    st.session_state["audio_done"] = False
    st.session_state["reset_audio_uploader"] = True
    st.rerun()



# ============================================================
# VIDEO LIBRARY HELPER
# ============================================================

def get_video_catalog():
    """Read all video files and clean names for UI."""
    videos = os.listdir(VIDEOS_DIR)
    catalog = []
    for v in videos:
        name = os.path.splitext(v)[0]
        clean_name = name.replace("_", " ").replace("-", " ")
        catalog.append({"file": v, "title": clean_name})
    return catalog


# ============================================================
# UI: HEADER & THEME
# ============================================================

if "question_history" not in st.session_state:
    st.session_state.question_history = []




st.markdown(""" 
<style> 
.header { 
display: flex; 
align-items: center; 
justify-content: 
space-between; 
padding: 12px 20px; 
border-bottom: 1px solid #ddd; 
} 
.title { 
font-size: 28px; 
font-weight: 700; 
} 
.subtitle { 
color: #888; 
margin-top: -5px; 
} 
</style> 
""", unsafe_allow_html=True)

left, right = st.columns([8,1])

with left:
    st.markdown("""
    <div style="padding:25px;
    border-radius:14px;background:
    linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    box-shadow:0 8px 25px rgba(0,0,0,0.25);
    margin-top:20px;
    margin-bottom:20px;">
    <h2 style="color:white;margin-bottom:5px;">🎓 RAG based AI Teaching Assistant</h2>
    <p style="color:#cfd8dc;font-size:15px;">
            Get concept-level answers with exact lecture timestamps using semantic retrieval.
    </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    theme = st.toggle("🌙", key="dark_mode_toggle")
    
    
if theme: st.markdown(""" 
<style> 

/* App background */ 
.stApp { 
background-color: #0E1117 !important; 
} 

/* Header */ 
header[data-testid="stHeader"] { 
background-color: #0E1117 !important; 
border-bottom: 1px solid #222; 
} 
div[data-testid="stFileUploader"] > 
div { background-color: #1E1E1E !important; 
border: 2px dashed #6B7280 !important; 
/* visible dashed border */ 
border-radius: 10px !important; 
padding: 12px !important; 
} 

/* Sidebar */ 

section[data-testid="stSidebar"] { 
background-color: #0B0E14 !important; 
} 

/* Make ALL text white */ 
body, p, span, label, h1, h2, h3, h4, h5, h6, div { color: #FFFFFF !important; 
} 

/* Fix File Uploader (whole white box) */ 
div[data-testid="stFileUploader"] > div { background-color: #1E1E1E !important; 
border: 1px dashed #444 !important; 
} 
div[data-testid="stFileUploader"] * { 
color: #FFFFFF !important; 
background-color: transparent !important; 
} 

/* Fix Selectbox (dropdown) */ 
div[data-baseweb="select"] > div { 
background-color: #1E1E1E !important; 
color: #FFFFFF !important; 
border: 1px solid #444 !important; 
} 
div[data-baseweb="select"] span { 
color: #FFFFFF !important; 
} 
ul[role="listbox"] { 
background-color: #1E1E1E !important; 
} 
li[role="option"] { 
color: #FFFFFF !important; 
background-color: #1E1E1E !important; 
} 
li[role="option"]:hover { 
background-color: #333333 !important; 
} 

/* Inputs */ 
input, textarea { 
background-color: #1E1E1E !important; 
color: #FFFFFF !important; border: 1px solid #444 !important; 
} 

/* Buttons */ 
button { 
background-color: #1f77b4 !important; 
color: #FFFFFF !important; 
} 
</style>
""", unsafe_allow_html=True) 

st.markdown(""" 
<style> 
.card { 
padding: 14px; 
border-radius: 10px; 
background-color: #f5f5f5; 
margin-bottom: 8px; } 
</style> 
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR: UPLOAD & LIBRARY
# ============================================================

with st.sidebar:
    st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        padding:22px;
        border-radius:14px;
        background:linear-gradient(135deg,#1e3c72,#2a5298);
        color:white;
        box-shadow:0 10px 25px rgba(0,0,0,0.25);
        margin-bottom:25px;
    ">
        <h3 style="margin-bottom:6px;">📺 Import Lecture from YouTube</h3>
        <p style="color:#e0e7ff;font-size:14px;">
            Paste a YouTube lecture link to automatically download, transcribe, and index it into the AI knowledge base.
        </p>
    </div>
    """, unsafe_allow_html=True)

    youtube_url = st.text_input("🔗 Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...",label_visibility="collapsed")

    if st.button("⬇️ Download & Process Lecture", use_container_width=True):
        if youtube_url.strip():
            try:
                with st.status("📡 Fetching video from YouTube servers...", expanded=True) as yt_status:
                    video_path = download_youtube_video(youtube_url)
                    yt_status.update(label="📁 Video successfully downloaded and stored", state="complete")

                st.success("🎉 Video Imported Successfully!")

                st.markdown(f"""
                <div style="padding:12px;border-radius:10px;background:#0e1117;border-left:4px solid #22c55e;">
                    <b>Saved File:</b><br>
                    <code>{os.path.basename(video_path)}</code>
                </div>
                """, unsafe_allow_html=True)

                with st.status("🧠 Running transcription, chunking & vector indexing...", expanded=True):
                    process_video(video_path)

            except Exception as e:
                st.error("❌ Unable to download this YouTube video automatically.")
                st.caption("This may be due to regional restrictions, DRM protection, or network issues.")

                st.markdown("""
                <div style="padding:15px;border-radius:10px;background:#1f2933;border-left:4px solid #f59e0b;">
                    <b>Manual Fallback Option</b><br><br>
                    Download the video manually using the trusted tool below, then upload it in the <i>Upload Lecture</i> section.
                    <br><br>
                    🔗 <a href="https://vidssave.com/yt" target="_blank" style="color:#60a5fa;font-weight:600;">
                    https://vidssave.com/yt
                    </a>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔧 Technical Error Details"):
                    st.code(str(e))
    st.divider()

    st.markdown("""
    <div style="
        padding:22px;
        border-radius:16px;
        background:linear-gradient(135deg,#0f766e,#134e4a,#022c22);
        box-shadow:0 12px 30px rgba(2,44,34,0.35);
        border:1px solid rgba(94,234,212,0.35);
    ">
        <h3 style="margin-bottom:6px;color:#ecfeff;">📥 Ingest New Lecture</h3>
        <p style="color:#a7f3d0;font-size:13px;line-height:1.5;">
            Upload raw lecture assets to automatically transcribe, embed, and index them for semantic retrieval.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # -------- Video Upload --------
    st.markdown("## 🎥 Upload Lecture Video (MP4)")
    
    if st.session_state.get("reset_uploader"):
        st.session_state.pop("video_uploader", None)
        st.session_state["reset_uploader"] = False
        
    uploaded_video = st.file_uploader(
        "🎥 Upload Lecture Video (MP4)",
        type=["mp4"],
        label_visibility="collapsed",
        key="video_uploader"
    )

    if uploaded_video and not st.session_state.get("video_done"):

        title = os.path.splitext(uploaded_video.name)[0]

        existing = collection.get(where={"title": title})
        if existing and len(existing.get("ids", [])) > 0:
            st.error(f"⚠️ Lecture '{title}' already exists in the knowledge base.")
            st.stop()

        save_path = os.path.join(VIDEOS_DIR, uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        if st.button("⚙️ Process Video", use_container_width=True):
            st.session_state["video_done"] = True
            process_video(save_path)

            
    if "sidebar_notice_video" in st.session_state:
        st.success(st.session_state["sidebar_notice_video"])
        del st.session_state["sidebar_notice_video"]
    

    # -------- Audio Upload --------
    st.markdown("## 🔊 Upload Lecture Audio (MP3)")
    
    if st.session_state.get("reset_audio_uploader"):
        st.session_state.pop("audio_uploader", None)
        st.session_state["reset_audio_uploader"] = False
    
    uploaded_audio = st.file_uploader(
        "🔊 Upload Lecture Audio (MP3)",
        type=["mp3"],
        label_visibility="collapsed",
        key="audio_uploader"
    )

    if uploaded_audio and not st.session_state.get("audio_done"):

        title = os.path.splitext(uploaded_audio.name)[0]

        # Check duplicate BEFORE saving
        existing = collection.get(where={"title": title})
        if existing and len(existing.get("ids", [])) > 0:
            st.error(f"⚠️ Lecture '{title}' already exists in the knowledge base.")
            st.stop()

        save_path = os.path.join(AUDIOS_DIR, uploaded_audio.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())

        st.success("📁 Audio uploaded successfully.")

        if st.button("⚙️ Process Audio", use_container_width=True):
            st.session_state["audio_done"] = True
            process_audio(save_path)

           
    if "sidebar_notice_audio" in st.session_state:
        st.success(st.session_state["sidebar_notice_audio"])
        del st.session_state["sidebar_notice_audio"]
         
    st.divider()

    # -------- Video Library --------
    st.markdown("""
    <div style="
        padding:20px;
        border-radius:16px;
        background:linear-gradient(135deg,#1e293b,#0f172a,#020617);
        border:1px solid rgba(148,163,184,0.25);
        box-shadow:0 10px 28px rgba(0,0,0,0.30);
    ">
        <h3 style="margin-bottom:6px;color:#f8fafc;">📚 Indexed Lecture Library</h3>
        <p style="color:#cbd5f5;font-size:13px;line-height:1.5;">
            “Restrict semantic search and full-lecture summarization to a specific lecture, or query across the entire knowledge corpus.”
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    # -------- Unified Lecture Library (Unique Video + Audio) --------

    # Load embedded titles (source of truth)
    
    all_meta = collection.get(include=["metadatas"])
    embedded_titles = sorted(set(m["title"] for m in all_meta["metadatas"]))

    
    video_files = os.listdir(VIDEOS_DIR)
    audio_files = os.listdir(AUDIOS_DIR)

    
    video_titles = {os.path.splitext(v)[0] for v in video_files}
    audio_titles = {os.path.splitext(a)[0] for a in audio_files}
    
    display_map = {}
    
    for title in embedded_titles:
        if title in video_titles:
            display_map[f"🎥 {title}"] = title
        elif title in audio_titles:
            display_map[f"🔊 {title}"] = title
        else:
            # Edge case: embedded but media missing
            display_map[f"📄 {title}"] = title
    
    all_display_titles = list(display_map.keys())
    
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = "All Lectures"
    
    # Handle reset after delete
    if st.session_state.get("reset_topic"):
        st.session_state["selected_topic"] = "All Lectures"
        st.session_state["reset_topic"] = False
    
    selected_display = st.selectbox(
        "📚 Select Lecture for Scoped Search",
        ["All Lectures"] + all_display_titles,
        key="selected_topic",
        label_visibility="collapsed"
    )
    # Clean value used for filtering
    if selected_display == "All Lectures":
        selected_topic = "All Lectures"
    else:
        selected_topic = display_map[selected_display]
        
    if "delete_msg" in st.session_state:
        st.success(st.session_state["delete_msg"])
        del st.session_state["delete_msg"]
        
        
#--------------------- Delete & Re-index Lecture ---------------------#

    if selected_topic != "All Lectures":
        st.markdown(f"""
        <div style="
            margin-top:14px;
            padding:16px;
            border-radius:14px;
            background:linear-gradient(135deg,#0b1220,#1e3a8a);
            border:1px solid #60a5fa;
            box-shadow:0 8px 24px rgba(30,58,138,0.35);
        ">
            <h4 style="color:#f8fafc;margin-bottom:4px;">⚙️ Manage Selected Lecture</h4>
            <p style="font-size:13px;color:#dbeafe;">
                Selected: <b>{selected_topic}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        confirm_delete = st.checkbox("⚠️ I understand this will permanently delete this lecture", key="confirm_delete")

        if confirm_delete:
            # DANGER BUTTON
            st.markdown("""
            <style>
            div.stButton > button {
                background-color: #7f1d1d;
                border: 1px solid #ef4444;
                color: white;
                font-weight: 600;
                border-radius: 10px;
            }
            div.stButton > button:hover {
                background-color: #991b1b;
                border-color: #f87171;
            }
            </style>
            """, unsafe_allow_html=True)

            if st.button("🗑 Permanently Delete Lecture", use_container_width=True):
                delete_lecture(selected_topic)
        else:
            # SAFE BUTTON
            st.markdown("""
            <style>
            div.stButton > button {
                background: linear-gradient(135deg, #020617, #0f172a, #1e3a8a);
                border: 1px solid rgba(96,165,250,0.6);
                color: #e5e7eb;
                font-weight: 600;
                border-radius: 12px;
                box-shadow: 0 0 0 1px rgba(96,165,250,0.25), 0 6px 18px rgba(0,0,0,0.35);
                transition: all 0.2s ease;
            }

            div.stButton > button:hover {
                background: linear-gradient(135deg, #0f172a, #1e40af);
                border-color: #93c5fd;
                box-shadow: 0 0 12px rgba(147,197,253,0.4), 0 10px 25px rgba(0,0,0,0.45);
            }
            </style>
            """, unsafe_allow_html=True)


            if st.button("🔄 Re-index Lecture", use_container_width=True):
                reindex_lecture(selected_topic)



# ============================================================
# MAIN: QUESTION ANSWERING
# ============================================================


 
# ================= Lecture Summary Panel =================



if selected_topic != "All Lectures":
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#0b0f1a,#111827);
        border:1px solid #1f2937;
        border-left:5px solid #22c55e;
        border-radius:14px;
        padding:18px 20px;
        margin-bottom:14px;
        box-shadow:0 8px 22px rgba(0,0,0,0.45);
    ">
        <div style="
            font-size:16px;
            font-weight:700;
            color:#f8fafc;
            display:flex;
            align-items:center;
            gap:8px;
        ">
            📘 Lecture Summary
        </div>
        <div style="
            font-size:13px;
            color:#9ca3af;
            margin-top:6px;
        ">
            Generate a complete structured summary for:
            <span style="color:#a7f3d0;font-weight:600;">{selected_topic}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    div.summary-btn > button {
        background: linear-gradient(135deg, #064e3b, #022c22);
        border: 1px solid #22c55e;
        color: #ecfdf5;
        font-weight: 600;
        padding: 6px 18px;
        border-radius: 10px;
        font-size: 13px;
    }
    div.summary-btn > button:hover {
        background: linear-gradient(135deg, #065f46, #022c22);
        box-shadow: 0 0 12px rgba(34,197,94,0.45);
    }
    </style>
    """, unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([3,2,3])
    with col_mid:
        if st.button("🧠 Generate Summary", key="summary_btn", use_container_width=False):
            with st.status("📚 Summarizing..."):
                quick_summary, full_summary = summarize_lecture_both(selected_topic)

                st.session_state["lecture_summary_quick"] = quick_summary
                st.session_state["lecture_summary_full"] = full_summary

    # ---- Check if summary exists ----
    has_summary = (
        isinstance(st.session_state.get("lecture_summary_quick"), str)
        and isinstance(st.session_state.get("lecture_summary_full"), str)
        and st.session_state["lecture_summary_quick"].strip() != ""
    )
    
    if has_summary:
        show_summary = st.toggle(
            "Show Lecture Summary",
            value=True,
            help="Toggle to display or hide the generated lecture summary"
        )

    else:
        show_summary = False

    if show_summary and st.session_state["lecture_summary_quick"] and st.session_state["lecture_summary_full"]:

        st.markdown("""
        <style>
        /* Radio pills styling */
        div[role="radiogroup"] > label {
            background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 6px 16px;
            margin-right: 10px;
            color: #0f172a;        /* dark text for clarity */
            font-weight: 600;
        }

        div[role="radiogroup"] > label:hover {
            background: linear-gradient(135deg, #e0f2fe, #bae6fd);
            border-color: #38bdf8;
            color: #0c4a6e;
        }

        /* Selected radio */
        div[role="radiogroup"] input:checked + label {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            border-color: #1e40af;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)


        view_mode = st.radio(
            "",
            ["⚡ Quick Summary (1–2 mins read)", "📚 Detailed Notes (Full)"],
            horizontal=True,
            key="summary_view_mode"
        )

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        if view_mode == "⚡ Quick Summary (1–2 mins read)":
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,#0b1220,#020617);
                border: 1px solid #14532d;
                border-left: 5px solid #22c55e;
                border-radius: 14px;
                padding: 18px;
                color: #e5e7eb;
                line-height: 1.7;
                box-shadow: 0 8px 24px rgba(0,0,0,0.45);
            ">
                {st.session_state["lecture_summary_quick"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,#0b1220,#020617);
                border: 1px solid #1e3a8a;
                border-left: 5px solid #3b82f6;
                border-radius: 14px;
                padding: 18px;
                color: #e5e7eb;
                line-height: 1.75;
                box-shadow: 0 8px 24px rgba(0,0,0,0.45);
            ">
                {st.session_state["lecture_summary_full"]}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)


        if view_mode == "⚡ Quick Summary (1–2 mins read)":
            quick_pdf = generate_pdf_bytes(
                f"{selected_topic} – Quick Summary",
                st.session_state["lecture_summary_quick"]
            )
            st.download_button(
                "⬇️ Download Quick Summary PDF",
                data=quick_pdf,
                file_name=f"{selected_topic}_quick_summary.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            full_pdf = generate_pdf_bytes(
                f"{selected_topic} – Detailed Notes",
                st.session_state["lecture_summary_full"]
            )
            st.download_button(
                "⬇️ Download Detailed Notes PDF",
                data=full_pdf,
                file_name=f"{selected_topic}_detailed_notes.pdf",
                mime="application/pdf",
                use_container_width=True
            )





    
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
# ================= Scoped Search Banner =================

if selected_topic != "All Lectures":
    st.markdown(f"""
    <div style="
        padding:16px 22px;
        border-radius:14px;
        background:linear-gradient(135deg,#020617,#0b1220,#1e3a8a);
        border:1px solid rgba(56,189,248,0.35);
        box-shadow:0 12px 28px rgba(0,0,0,0.45);
        margin-bottom:18px;
    ">
        <div style="
            color:#7dd3fc;
            font-size:12px;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin-bottom:4px;
        ">
            🔍 Searching Within
        </div>
        <div style="
            font-size:18px;
            font-weight:700;
            color:#f8fafc;
        ">
            {selected_topic}
        </div>
    </div>
    """, unsafe_allow_html=True)



else:
    st.markdown("""
    <div style="
        padding:12px 16px;
        border-radius:10px;
        background:linear-gradient(135deg,#020617,#0b1220,#1e3a8a);
        box-shadow:0 12px 28px rgba(0,0,0,0.45);
        border:1px dashed #334155;
        margin-bottom:10px;
        color:#94a3b8;
        font-size:13px;
    ">
        🔍 Searching across <b>all lectures</b>
    </div>
    """, unsafe_allow_html=True)

    
# ================= Question Input Section =================


st.markdown("""
<div style="
     margin-top:10px;
     margin-bottom:4px;">
    <h3 style="font-weight:700,margin-bottom:4px;">🔎 Ask About This Lecture</h3>
    <p style="color:#6b7280,margin-bottom:6px;">
        Example: At what timestamp is bias–variance tradeoff explained in this lecture?
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Previous Questions Dropdown ----
if st.session_state.question_history:
    st.markdown("#### 🕘 Query History")
    selected_prev = st.selectbox(
        "Reuse a previous query",
        ["Select a question"] + st.session_state.question_history,
        label_visibility="collapsed"
    )
else:
    selected_prev = "Select a question"

# ---- Query Input ----
query = st.text_input(
    "Question: ",
    placeholder="Type your question here...",
    value="" if selected_prev == "Select a question" else selected_prev,
    label_visibility="collapsed"
)

# ---- Ask Button ----
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([1,8])
with col1:
    ask_btn = st.button("Search")


# ---- Processing Pipeline ----
if ask_btn and query.strip():
    if query not in st.session_state.question_history:
        st.session_state.question_history.insert(0, query)
        
    collection = st.session_state["collection"]
    if collection.count() == 0:
        st.info("📭 No lectures indexed yet. Upload a video or audio to begin.")
        st.stop()


    with st.status("🧠 Performing vector similarity search across transcript embeddings using ChromaDB...") as search_status:

        collection = st.session_state["collection"]  # 🔴 force fresh handle
        
        # Encode query
        q_emb = create_embedding([query])[0]

        # Vector search (scoped or global)
        if selected_topic == "All Lectures":
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=5
            )
        else:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=5,
                where={"title": selected_topic}
            )

        if not results["documents"] or not results["documents"][0]:
            st.warning("No chunks found. Try another query.")
            st.stop()

        # Build chunks list
        top_chunks = []
        for meta, text in zip(results["metadatas"][0], results["documents"][0]):
            top_chunks.append({
                "title": meta["title"],
                "number": meta["number"],
                "start": meta["start"],
                "end": meta["end"],
                "text": text
            })

        best_chunk = top_chunks[0]
        search_status.update(label="Top relevant transcript segments retrieved 🔎", state="complete")

    # ---- LLM Prompt ----
    with st.status("🤖 Generating context-grounded answer using LLM (gpt-5 model) reasoning...") as llm_status:

        prompt = f"""
        You are an expert AI Teaching Assistant specialized in explaining lecture videos with timestamp grounding.

        Context (most relevant transcript segments):
        {json.dumps(top_chunks, indent=2)}

        User Question:
        {query}

        Task:
        Generate a factually grounded answer strictly based on the provided transcript segments.

        Response Requirements:
        1. Clearly identify the exact concept or topic being asked.
        2. Mention the lecture title(s) where the answer is found.
        3. Provide the precise timestamp range(s) for each explanation.
        4. Explain the concept concisely but technically (no oversimplification, no fluff).
        5. Use well-structured bullet points with sub-bullets where helpful.
        6. Output in clean Markdown format.
        7. English only.
        8. Each bullet should be on its own line (no inline merging).
        9. Do not hallucinate or add information not present in the transcript.
        10. If the answer is partially present, explicitly state what is missing.

        Formatting:
        - Use **bold** for key terms.
        - Use `code` formatting for formulas, algorithms, or variables.
        - Keep each point short, precise, and exam-ready.

        """

        answer = inference(prompt)
        
        llm_status.update(label="Answer generated with timestamp grounding ✅", state="complete")

    # ============================================================
    # RESULT DISPLAY
    # ============================================================

    st.markdown("## 🧑 User Question")
    st.success(query)

    st.markdown("## 🤖 AI Assistant Response")
    st.info(answer)

    st.markdown("""
    <div style="padding:16px;
    border-radius:10px;
    background:#111827;
    border-left:4px solid #22c55e;
    margin-top:30px;
    margin-bottom:16px;
    ">
        <h4 style="color:#22c55e;">🎯 Exact Lecture Location</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    📘 **Lecture Title:** `{best_chunk['title']}`  
    ⏱ **Timestamp:** `{round(best_chunk['start'],2)}s – {round(best_chunk['end'],2)}s`
    """)

    video_file = best_chunk["title"] + ".mp4"
    audio_file = best_chunk["title"] + ".mp3"

    video_path = os.path.join(VIDEOS_DIR, video_file)
    audio_path = os.path.join(AUDIOS_DIR, audio_file)


    if os.path.exists(video_path):
        st.markdown("""
        <div style="padding:13px;
        border-radius:10px;
        background:#111827;
        border-left:4px solid #2297c5;
        margin-top:30px;
        margin-bottom:10px;
        ">
            <h4 style="color:#2297c5;">▶️ Video Playback</h4>
        </div>
        """, unsafe_allow_html=True)
        st.video(video_path, start_time=int(best_chunk["start"]))
        
    elif os.path.exists(audio_path):
        st.warning("🎥 Video file not available for this lecture.")
        st.markdown("""
        <div style="padding:16px;
        border-radius:10px;
        background:#111827;
        border-left:4px solid #2297c5;
        margin-top:30px;
        margin-bottom:10px;
        ">
            <h4 style="color:#2297c5;">🔊 Audio Playback</h4>
        </div>
        """, unsafe_allow_html=True)
        st.audio(audio_path, start_time=int(best_chunk["start"]))
    else:
        st.error("❌ No corresponding video or audio file found in the library.")
