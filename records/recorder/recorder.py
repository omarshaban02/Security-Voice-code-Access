import sounddevice as sd
import wavio as wv


def record_audio(duration=10, filename="recording.wav"):
    """
  Records audio for `duration` seconds and saves it to `filename`.
  """
    fs = 44100  # Sampling rate
    print("Recording starts...")
    recording = sd.rec(duration * fs, samplerate=fs, channels=2)
    sd.wait()
    print("Recording ends...")
    wv.write(filename, recording, fs, sampwidth=2)
    print(f"{filename} is saved...")


if __name__ == "__main__":
    record_audio(duration=10, filename="recording.wav")
