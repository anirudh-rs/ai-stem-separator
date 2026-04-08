import subprocess
import os
import shutil
import time
import tempfile

def separate_stems(input_path: str, output_dir: str = "outputs") -> dict:

    # ── Wipe outputs folder safely ────────────────────────────
    if os.path.exists(output_dir):
        for attempt in range(10):
            try:
                shutil.rmtree(output_dir)
                break
            except PermissionError:
                time.sleep(2)

    os.makedirs(output_dir, exist_ok=True)

    # ── Copy input to a safe temp location ───────────────────
    # Avoids Demucs locking the original upload temp file
    safe_input = os.path.join(output_dir, "input" + os.path.splitext(input_path)[1])
    shutil.copy2(input_path, safe_input)

    from pydub import AudioSegment

# ── Convert to standard WAV before processing ────────────────
    wav_input = os.path.join(output_dir, "input.wav")
    audio = AudioSegment.from_file(safe_input)
    audio.export(wav_input, format="wav")
    safe_input = wav_input  # use converted file instead

    command = [
        "python", "-m", "demucs",
        "--out", output_dir,
        "--mp3",
        "-n", "htdemucs_6s",
        safe_input
    ]

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    print("=== DEMUCS STDOUT ===")
    print(result.stdout)
    print("=== DEMUCS STDERR ===")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    # ── Auto-find stems ───────────────────────────────────────
    stem_files = {}
    found_dir = None

    for root, dirs, files in os.walk(output_dir):
       if root == output_dir:
          continue
       
       print(f"Scanning: {root}")
       print(f"Files found: {files}")
        
       mp3_files = [f for f in files if f.endswith(".mp3")]
       if mp3_files:
            found_dir = root
            print(f"✅ Found stems in: {root}")
            for f in mp3_files:
                print(f"   → {f}")
            break

    if not found_dir:
        raise RuntimeError("Demucs ran but no output files were found.")

    for stem in ["vocals", "drums", "bass", "other", "guitar", "piano"]:
        stem_path = os.path.join(found_dir, f"{stem}.mp3")
        stem_files[stem] = stem_path if os.path.exists(stem_path) else None

    return stem_files