import os
import subprocess

video_folder = "videos"
audio_folder = "audios"

for video_file in os.listdir(video_folder):
    if not video_file.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
        continue

    video_path = os.path.join(video_folder, video_file)
    audio_name = os.path.splitext(video_file)[0] + ".mp3"
    audio_path = os.path.join(audio_folder, audio_name)

    print(f"🎬 Converting: {video_file} → {audio_name}")

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ FFmpeg error for {video_file}")
        print(result.stderr)
        continue

    print(f"✅ Saved: {audio_path}")

print("🎉 All videos converted to audio.")
