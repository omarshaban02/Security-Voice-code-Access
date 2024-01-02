import librosa
import numpy as np
from scipy.spatial.distance import cosine
import sounddevice as sd
from scipy.io import wavfile as wav


def record_audio(duration=3, filename="recording.wav"):
    fs = 44100  # Sampling rate
    recording = sd.rec(duration * fs, samplerate=fs, channels=2)
    print("Recording starts...")
    sd.wait()
    print("Recording ends...")
    wav.write(filename, fs, recording)
    return fs, recording.T


def extract_mfcc_delta_delta(audio_file=None, fs=None, data=None, n_mfcc=13, delta_order=1):
    y = np.zeros(5)
    if audio_file:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        print("y in function", y.shape)

    elif fs and data:
        # Pre-emphasis
        y = librosa.effects.preemphasis(data.flatten())

    # Parameters
    frame_length = 2048
    hop_length = 512

    # Manually create frames
    num_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(y, (frame_length,))
    frames = frames[::hop_length]

    # Hamming window
    hamming_window = np.hamming(frame_length)[:, np.newaxis]

    # Apply window to frames
    # frames *= hamming_window.T

    # Compute power spectrum
    power_spectrum = np.abs(np.fft.fft(frames, axis=0)) ** 2

    # Mel filterbank
    mel_filterbank = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=23)
    # mel_energies = mel_filterbank @ power_spectrum
    print("Mel filterbank", mel_filterbank.shape)
    print("power spectrum", power_spectrum.shape)
    # Logarithm
    log_mel_energies = np.log(mel_energies + 1e-10)  # Adding a small constant to avoid log(0)

    # Discrete Cosine Transform (DCT)
    mfccs = librosa.feature.mfcc(S=log_mel_energies, n_mfcc=n_mfcc)

    # Delta and delta-delta features
    delta = librosa.feature.delta(mfccs, order=delta_order)
    delta_delta = librosa.feature.delta(mfccs, order=2)

    # Stack the features
    features = np.vstack([mfccs, delta, delta_delta])

    return features.flatten()


def calculate_cosine_similarity(feature_vector1, feature_vector2):
    # Calculate cosine similarity between two feature vectors
    return 1 - cosine(feature_vector1, feature_vector2)


def compare_with_saved_voices(input_features, saved_voice_features):
    # Compare input voice with saved voices using cosine similarity
    similarity_scores = [calculate_cosine_similarity(input_features, saved_features) for saved_features in
                         saved_voice_features]

    # Scale similarity scores to the range [0, 1]
    scaled_similarity_scores = [(score + 1) / 2 for score in similarity_scores]

    # Rank saved voices based on similarity scores
    ranked_voices = sorted(enumerate(scaled_similarity_scores), key=lambda x: x[1], reverse=True)

    return ranked_voices


if __name__ == "__main__":
    # Replace 'path/to/input/audio/file.wav' with the path to your input voice recording
    # input_audio_path = 'path/to/input/audio/file.wav'

    # Replace 'path/to/saved/voice1/file.wav', etc., with the paths to your saved voice samples
    saved_voice_paths = ['open1.wav', 'open2.wav']

    # Extract MFCCs with delta and delta-delta features for the saved voices
    saved_voice_features = [extract_mfcc_delta_delta(saved_voice_path) for saved_voice_path in saved_voice_paths]

    input_audio_sr, input_audio_data = record_audio(duration=4, filename="open_test.wav")
    print("data from wsd", input_audio_data)
    # Extract MFCCs with delta and delta-delta features for the input voice
    input_features = extract_mfcc_delta_delta(sr=input_audio_sr, y=input_audio_data)


    # Compare the input voice with saved voices using cosine similarity
    ranked_voices = compare_with_saved_voices(input_features, saved_voice_features)

    # Display the ranked voices
    for index, (voice_index, similarity_score) in enumerate(ranked_voices):
        print(f"Rank {index + 1}: Voice {voice_index + 1}, Similarity Score: {similarity_score * 100:.2f}%")
