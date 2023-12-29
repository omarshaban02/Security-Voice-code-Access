import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def compute_mfcc(audio, fs):
    mfcc = librosa.feature.mfcc(audio, sr=fs,
                                n_mfcc=13)  # You can adjust the number of MFCC coefficients (n_mfcc) as needed
    return mfcc


def extract_spectrogram_features(spectro):
    # Calculate mean, standard deviation, skewness, and kurtosis along the time axis
    mean_features = np.mean(spectro, axis=1)
    std_features = np.std(spectro, axis=1)
    skewness_features = skew(spectro, axis=1)
    kurtosis_features = kurtosis(spectro, axis=1)

    # Concatenate all features into a single array
    features = np.concatenate([mean_features, std_features, skewness_features, kurtosis_features])

    return features


def extract_mfcc_features(audio, fs):
    mfcc = compute_mfcc(audio, fs)
    # Calculate mean, standard deviation, skewness, and kurtosis along the time axis for MFCCs
    mean_mfcc = np.mean(mfcc, axis=1)
    std_mfcc = np.std(mfcc, axis=1)
    skewness_mfcc = skew(mfcc, axis=1)
    kurtosis_mfcc = kurtosis(mfcc, axis=1)

    # Concatenate all MFCC features into a single array
    return np.concatenate([mean_mfcc, std_mfcc, skewness_mfcc, kurtosis_mfcc])


# Example usage
# Replace 'your_audio' with the actual audio signal and 'your_fs' with the sampling rate
your_audio, your_fs = librosa.load('path/to/your/audio.wav', sr=None)
audios_list = [librosa.load('path/to/your/audio.wav', sr=None)]
output = librosa.load('path/to/your/audio.wav', sr=None)

# Compute spectrogram features
spectrogram = np.abs(librosa.stft(your_audio))
spectrogram_features = extract_spectrogram_features(spectrogram)

# Compute MFCC features
mfcc_features = extract_mfcc_features(your_audio, your_fs)

# Concatenate both sets of features
all_features = np.concatenate([spectrogram_features, mfcc_features])
