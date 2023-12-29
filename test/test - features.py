import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd


def record_voice(duration=5, sr=44100):
    print("Recording... Speak now.")
    audio_data = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    return audio_data.flatten()


def compute_spectrogram(audio_data, sr=22050, n_fft=2048, hop_length=512):
    spectrogram = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
    return spectrogram


def extract_features(audio_data, sr=22050, n_fft=2048, hop_length=512):
    # MFCCs : are a set of coefficients that represent the short-term power spectrum of a sound signal
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    # Chroma feature : represent the distribution of energy in the pitch classes
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Spectral contrast : measures the difference in amplitude between peaks
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Rhythm features
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    mean_spec = np.mean(audio_data)
    std_spec = np.std(audio_data)

    # Create DataFrames for each feature
    df_mfcc = pd.DataFrame(data=mfcc.T, columns=[f'mfcc_{i}' for i in range(mfcc.shape[0])])
    df_chroma = pd.DataFrame(data=chroma.T, columns=[f'chroma_{i}' for i in range(chroma.shape[0])])
    df_spectral_contrast = pd.DataFrame(data=spectral_contrast.T,
                                        columns=[f'spectral_contrast_{i}' for i in range(spectral_contrast.shape[0])])

    df_rhythm_stat = pd.DataFrame(data=[[tempo, mean_spec, std_spec]], columns=['tempo', 'mean_spec', 'std_spec'])

    return df_mfcc, df_chroma, df_spectral_contrast, df_rhythm_stat


print("Recording the first sentence:")
voice1 = record_voice()

df_mfcc, df_chroma, df_spectral_contrast, df_rhythm_stat = extract_features(voice1)

print("DataFrame - MFCC:")
print(df_mfcc)

print("DataFrame - Chroma:")
print(df_chroma)

print("DataFrame - Spectral Contrast:")
print(df_spectral_contrast)

print("DataFrame - Rhythm and Statistical Features:")
print(df_rhythm_stat)
