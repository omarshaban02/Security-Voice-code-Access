import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sounddevice as sd
import wavio as wav


def extract_mfcc(y, sr, n_mfcc=13):
    # Load audio file using librosa
    # y, sr = librosa.load(audio_file)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Take the mean along the time axis
    mean_mfcc = np.mean(mfcc, axis=1)

    return mean_mfcc


def cosine_similarity_score(features1, features2):
    # Calculate cosine similarity between feature vectors
    similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
    return similarity_matrix[0, 0]


def record_audio(duration=10, filename="recording.wav"):
    """
    Records audio for `duration` seconds and saves it to `filename`.
    """
    fs = 44100  # Sampling rate
    recording = sd.rec(duration * fs, samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, recording, fs, sampwidth=2)
    return recording, fs


def main():
    # voice1_file = 'D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/unlock2.wav'
    # voice2_file = 'D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/not_omar.wav'

    # Extract MFCC features
    # mfcc_features1 = extract_mfcc(voice1_file)
    # mfcc_features2 = extract_mfcc(voice2_file)

    # Paths to two voice recordings in WAV format
    voice1_data, voice1_sr = librosa.load('D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/unlock2.wav')
    voice2_data, voice2_sr = record_audio(duration=4, filename="not_omar2.wav")

    # print(voice1_data.shape, voice2_data)
    voice2_data = voice2_data.reshape(voice2_data.shape[0])
    # Extract MFCC features
    mfcc_features1 = extract_mfcc(voice1_data, voice1_sr)
    mfcc_features2 = extract_mfcc(voice2_data, voice2_sr)

    # Calculate similarity score using cosine similarity
    similarity_score = cosine_similarity_score(mfcc_features1, mfcc_features2)

    print(f"Similarity Score: {similarity_score * 100:.2f}%")


if __name__ == "__main__":
    main()
