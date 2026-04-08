# test_audio.py
from pydub import AudioSegment

audio = AudioSegment.from_file(r"C:\Users\aniru\Downloads\Blackbird (Remastered 2009).mp3")
print("Duration:", len(audio) / 1000, "seconds")
print("Channels:", audio.channels)
print("Sample rate:", audio.frame_rate)