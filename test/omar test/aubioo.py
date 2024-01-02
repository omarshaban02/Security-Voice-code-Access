import aubio
import numpy as np
from scipy.stats import wasserstein_distance


def compute_pitch_histogram(audio_file, hop_size=512, pitch_method="yin"):
    # Set up parameters
    samplerate = 44100  # Standard audio CD sampling rate

    # Create an aubio source
    source = aubio.source(audio_file, samplerate, hop_size)

    # Create pitch detection object
    pitch_o = aubio.pitch(pitch_method, 2048, hop_size, samplerate)

    # Initialize variables
    pitches = []

    # Process the audio file
    total_frames = 0
    while True:
        samples, read = source()
        pitch = pitch_o(samples)[0]
        pitches.append(pitch)

        total_frames += read
        if read < hop_size:
            break

    # Compute pitch histogram
    hist, bin_edges = np.histogram(pitches, bins=50, range=(80, 400), density=True)

    return hist


def compare_audio_similarity(file1, file2):
    # Compute pitch histograms for both audio files
    hist1 = compute_pitch_histogram(file1)
    hist2 = compute_pitch_histogram(file2)

    # Compute Wasserstein distance between the two histograms
    distance = wasserstein_distance(hist1, hist2)

    # Return a similarity score (inverse of distance)
    similarity_score = 1 / (1 + distance) * 100

    return similarity_score


# Example usage
file_path1 = "open2.wav"
file_path2 = "unlock2.wav"

similarity_score = compare_audio_similarity(file_path1, file_path2)
print(f"Similarity Score: {similarity_score:.2f}")
