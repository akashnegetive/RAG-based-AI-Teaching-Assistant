import sys
import os
import subprocess

video_path = sys.argv[1]

# ---------- Writable base directory ----------
BASE_DATA_DIR = os.path.join(os.path.expanduser("~"), "rag_data")
AUDIOS_DIR = os.path.join(BASE_DATA_DIR, "audios")
os.makedirs(AUDIOS_DIR, exist_ok=True)

audio_name = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
audio_path = os.path.join(AUDIOS_DIR, audio_name)

result = subprocess.run(
    ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("FFmpeg error:", result.stderr)
    sys.exit(1)

# IMPORTANT: print full path so app.py receives the correct path
print(audio_path)
