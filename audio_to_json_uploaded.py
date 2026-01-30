import sys
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- Writable base directory (Streamlit Cloud safe) ----------
BASE_DATA_DIR = os.path.join(os.path.expanduser("~"), "rag_data")
AUDIOS_DIR = os.path.join(BASE_DATA_DIR, "audios")
JSONS_DIR  = os.path.join(BASE_DATA_DIR, "jsons")

os.makedirs(AUDIOS_DIR, exist_ok=True)
os.makedirs(JSONS_DIR, exist_ok=True)

# ---------- OpenAI client ----------
client = OpenAI()

# ---------- Inputs ----------
audio_file = sys.argv[1]
audio_path = os.path.join(AUDIOS_DIR, audio_file)
title = os.path.splitext(audio_file)[0]

try:
    with open(audio_path, "rb") as f:
        transcript = client.audio.translations.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
        )

    chunks = []
    for seg in transcript.segments:
        chunks.append({
            "number": title.split("_")[0] if title.split("_")[0].isdigit() else "NA",
            "title": title,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    json_path = os.path.join(JSONS_DIR, f"{title}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

    # IMPORTANT: print full path so app.py receives the correct file
    print(json_path)

except Exception as e:
    print("OPENAI_ERROR:", str(e))
    sys.exit(1)
