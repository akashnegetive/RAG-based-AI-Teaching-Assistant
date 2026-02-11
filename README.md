# 🎓 RAG-Based AI Teaching Assistant

🚀 **Live App**  
https://rag-based-ai-teaching-akash.streamlit.app/

---

## 🚀 Overview

This project builds an end-to-end **Retrieval-Augmented Generation (RAG)** system for lecture understanding and study assistance.  
It enables users to upload or import lectures, perform semantic search with timestamp grounding, generate structured summaries, and navigate lectures at a concept level.

The platform is designed for **real-world academic and enterprise knowledge-assistant use cases**.

Users can:

- Upload or import lecture videos and audio (including YouTube)
- Automatically transcribe and chunk lectures with timestamps
- Store embeddings in a vector database
- Ask natural-language questions
- Retrieve answers with exact lecture timestamps
- Generate full lecture summaries and study notes
- Export summaries as PDFs
- Navigate lectures using concept-level indexing

---

## ✨ Key Features

- 🔍 **Timestamp-grounded semantic Q&A**
- 📚 **Lecture-level scoped search**
- 🧠 **Full lecture summarization**
  - Quick summary (1–2 min read)
  - Detailed study notes
- 📑 **PDF export for summaries**
- 🧭 **Concept index / chapter navigation**
- 🎥 **Synchronized video & audio playback**
- 🗂️ **Lecture lifecycle management**
  - Upload
  - Delete
  - Re-index
- ▶️ **Multimodal ingestion**
  - Video
  - Audio
  - YouTube links

---

## 🏗️ System Architecture
   - Video / Audio / YouTube
   -  FFmpeg
   - Whisper ASR
 - Timestamped JSON
 -  Chunking + Metadata
- OpenAI Embeddings (text-embedding-3-large)
-   ChromaDB
- Semantic Retrieval + Filters
- GPT-5 Inference
- UI + Playback + PDF Export


---

## 🧰 Tech Stack

### Core
- Python
- Streamlit
- ChromaDB

### AI & LLM
- OpenAI GPT-5
- OpenAI `text-embedding-3-large`

### Speech & Media
- FFmpeg
- Whisper (ASR)

### Ingestion
- yt-dlp (YouTube ingestion)

### Document Export
- ReportLab (PDF generation)

---
## Website Link   [🔗](https://rag-based-ai-teaching-akash.streamlit.app/)

### 1) Main Page
  
 <img width="1845" height="817" alt="image" src="https://github.com/user-attachments/assets/e694bd24-a67c-46ef-8f63-862599692df9" />
  
### 2) Media Inputs
 
  <img width="358" height="815" alt="image" src="https://github.com/user-attachments/assets/6415db94-d49f-4743-a4fe-c1a79afc255a" />

### 3) Scoped Filtering & Lecture Management (Delete + Re-index )
  
<img width="371" height="778" alt="image" src="https://github.com/user-attachments/assets/38d118a2-923c-4531-a4bc-c909dcedecca" />

### 4) Summarization & Pdf Export
  
<img width="1475" height="688" alt="image" src="https://github.com/user-attachments/assets/ce540447-fe8d-45fc-8fd9-fe074e072cca" />

### 5) Question Answering Using LLM (Chatgpt-5)
  
<img width="1582" height="788" alt="image" src="https://github.com/user-attachments/assets/ca61e051-577f-451a-8db4-60e8cc2ff70b" />

### 6) Exact Timestamp with Media Playback

<img width="1290" height="243" alt="image" src="https://github.com/user-attachments/assets/a9099866-f207-4cc9-82ce-0d4f7d59da0a" />
<img width="1141" height="727" alt="image" src="https://github.com/user-attachments/assets/e7eaa242-6c45-4ca1-80a3-572d6d9eab9a" />

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/akashnegetive/RAG-based-AI-Teaching-Assistant.git
cd RAG-based-AI-Teaching-Assistant
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Create a .env file:
```bash
api_key=YOUR_OPENAI_API_KEY
```
### 5️⃣ Run the application

```bash
streamlit run app.py

```

### 📥 Supported Inputs
- Local video files (MP4)
- Local audio files (MP3 / WAV)
- YouTube lecture URLs

---

## 📖 How It Works

### 🔹 Lecture Ingestion Pipeline

- Video → audio extraction (FFmpeg)  
- Audio → timestamped transcript (Whisper ASR)  
- Transcript → chunks + structured metadata  
- Chunks → vector embeddings (OpenAI `text-embedding-3-large`)  
- Embeddings → stored in ChromaDB  



### 🔹 Question Answering Flow

- User query → query embedding  
- Vector similarity search over ChromaDB  
- Optional lecture-scoped filtering  
- GPT-5 grounded answer generation  
- Timestamp references returned with synchronized video/audio playback  


### 🔹 Lecture Summarization

- Entire lecture transcript is loaded
- Two parallel summarization pipelines are generated:
  - ⚡ Quick Summary (1–2 min read)
  - 📚 Detailed Notes (full study notes)
- Both summaries can be exported as PDF

---

## 📑 PDF Export

The system generates downloadable PDFs for:

- Quick lecture summary  
- Detailed study notes  

PDFs are generated fully in memory and streamed directly to the user  
(no persistent server storage is required).

---

## 🧭 Concept Index (Chapter Navigation)

The system automatically extracts structured lecture segments and builds a lightweight concept index that enables:

- Viewing the major topics covered in a lecture  
- Jumping directly to the corresponding timestamps in the video player  

---

## 👤 Author

**Akash Gupta**

**Project:**  
RAG-Based AI Teaching Assistant  

🔗 GitHub: https://github.com/akashnegetive/RAG-based-AI-Teaching-Assistant


This provides a Coursera / Udemy-style chapter navigation experience.





