import sys
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"))

audio_file = sys.argv[1]
audio_path = os.path.join("audios", audio_file)
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

    json_path = os.path.join("jsons", f"{title}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

    print(json_path)

except Exception as e:
    print("OPENAI_ERROR:", str(e))
    sys.exit(1)