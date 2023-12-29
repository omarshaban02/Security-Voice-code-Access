import librosa
import numpy as np
from pydub import AudioSegment


def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Extract pitch
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])

    # Extract formant frequencies
    formants = librosa.effects.find_f0(y, sr=sr)
    formants = formants[formants != 0]  # Remove zero values
    formant_freq = np.mean(formants)

    # Extract jitter and shimmer
    jitter = librosa.effects.jitter(y)
    shimmer = librosa.effects.shimmer(y)

    # Extract prosodic features
    audio_segment = AudioSegment.from_wav(file_path)
    pitch_contour = audio_segment._spawn(pitches)
    energy_contour = audio_segment.dBFS

    # Calculate mean and standard deviation of MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Calculate mean and standard deviation of Chroma
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Calculate tempo
    tempo, _ = librosa.beat.beat_track(y, sr=sr)

    # Extract spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Calculate mean and standard deviation of spectral contrast
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_std = np.std(spectral_contrast, axis=1)

    # Combine all features into a single vector
    all_features = np.concatenate([
        mfccs_mean, mfccs_std, chroma_mean, chroma_std,
        [tempo], spectral_contrast_mean, spectral_contrast_std,
        [pitch, formant_freq, jitter, shimmer, pitch_contour, energy_contour]
    ])

    return all_features


if __name__ == "__main__":
    # Replace 'your_audio_file.wav' with the path to your WAV file
    audio_file_path = 'your_audio_file.wav'

    # Extract features
    features = extract_features(audio_file_path)

    # Print extracted features
    print("Extracted Features:")
    print(features)
