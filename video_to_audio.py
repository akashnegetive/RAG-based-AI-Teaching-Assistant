import sys
import os
import subprocess

video_path = sys.argv[1]

audio_name = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
audio_path = os.path.join("audios", audio_name)

result = subprocess.run(
    ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("FFmpeg error:", result.stderr)
    sys.exit(1)

print(audio_path)   # <-- THIS IS CRITICAL