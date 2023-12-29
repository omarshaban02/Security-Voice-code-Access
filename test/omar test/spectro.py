import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import spectrogram
from sklearn.metrics.pairwise import cosine_similarity


def compute_spectrogram(audio, fs):
    _, _, spec = spectrogram(audio, fs)
    return spec


def extract_features(audio_file):
    # Load audio file using librosa
    y, sr = librosa.load(audio_file)

    # Compute the spectrogram
    spec = compute_spectrogram(y, sr)

    # Extract statistical features
    mean_features = np.mean(spec, axis=1)
    std_features = np.std(spec, axis=1)
    skewness_features = skew(spec, axis=1)
    kurtosis_features = kurtosis(spec, axis=1)

    # Concatenate all features
    features = np.concatenate([mean_features, std_features, skewness_features, kurtosis_features])

    return features


def cosine_similarity_score(features1, features2):
    # Calculate cosine similarity between feature vectors
    similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
    return similarity_matrix


def main():
    # Replace these paths with your actual audio file paths
    audio_file1 = "D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/unlock2.wav"
    audio_file2 = "D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/unlock1.wav"

    # Extract features from the two audio files
    features1 = extract_features(audio_file1)
    features2 = extract_features(audio_file2)

    # Calculate similarity score using cosine similarity
    similarity_score = cosine_similarity_score(features1, features2)

    # Convert similarity score to percentage
    # similarity_percentage = similarity_score * 100

    # print(f"Similarity Percentage: {similarity_percentage:.2f}%")
    print(similarity_score)


if __name__ == "__main__":
    main()
