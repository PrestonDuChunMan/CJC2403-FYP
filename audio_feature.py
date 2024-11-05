import librosa

# Load audio from a video file
audio_path = './emotional.mp3'
y, sr = librosa.load(audio_path)

# Estimate tempo (in beats per minute)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f'Tempo: {tempo} BPM')