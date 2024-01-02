import numpy as np
from scipy.io import wavfile as wav
from scipy.signal import spectrogram, find_peaks
import hashlib


class AudioFingerprint:
    def __init__(self, song_id, file_path):
        self.song_id = song_id
        self.fingerprints = []
        self.file_path = file_path
        self.rate, self.audio_data = self.read_audio(self.file_path)
        self.create_and_add_fingerprints()

    def read_audio(self, file_path):
        rate, audio_data = wav.read(file_path)
        return rate, audio_data.T

    def create_spectrogram(self, audio_data):
        _, _, spectrogram_data = spectrogram(audio_data, fs=self.rate)
        return spectrogram_data

    def find_spectrogram_peaks(self, spectrogram_data):
        peaks, _ = find_peaks(np.log1p(spectrogram_data.ravel()), height=5)
        return peaks

    def hash_peaks(self, peaks, max_time_difference=100):
        hash_values = [hashlib.sha1(f"{peaks[i]}|{peaks[j]}|{peaks[j] - peaks[i]}".encode()).hexdigest()[:20]
                       for i in range(len(peaks)) for j in range(i + 1, len(peaks))
                       if 0 < (peaks[j] - peaks[i]) <= max_time_difference]
        return hash_values

    def add_fingerprints(self, fingerprints):
        self.fingerprints.extend(fingerprints)

    def compare_with_saved_fingerprints(self, input_audio_data):
        input_spectrogram = self.create_spectrogram(input_audio_data)
        input_peaks = self.find_spectrogram_peaks(input_spectrogram)
        input_hashes = self.hash_peaks(input_peaks)

        similarity_scores = [(saved_offset, input_hash) for saved_offset, saved_hash in self.fingerprints
                             for input_hash in input_hashes if saved_hash == input_hash]

        return similarity_scores

    def calculate_jaccard_similarity(self, similarity_scores, fingerprints_saved):
        saved_hashes = set(hash for _, hash in fingerprints_saved)
        input_hashes = set(hash for _, hash in similarity_scores)

        intersection_size = len(saved_hashes.intersection(input_hashes))
        union_size = len(saved_hashes.union(input_hashes))

        jaccard_similarity = intersection_size / union_size if union_size != 0 else 0
        percentage_similarity = jaccard_similarity * 100

        return percentage_similarity, jaccard_similarity

    def create_and_add_fingerprints(self):
        rate, audio_data = self.read_audio(self.file_path)
        self.rate = rate

        spectrogram_data = self.create_spectrogram(audio_data)
        peaks = self.find_spectrogram_peaks(spectrogram_data)
        hash_values = self.hash_peaks(peaks)
        fingerprints = [(offset, fingerprint_hash) for offset, fingerprint_hash in enumerate(hash_values)]
        self.add_fingerprints(fingerprints)


# Example Usage
file_path_saved = 'open1.wav'
file_path_input = 'unlock2.wav'
saved_song_id = 1

audio_fingerprint_saved = AudioFingerprint(saved_song_id, file_path_saved)
rate_input, audio_data_input = audio_fingerprint_saved.read_audio(file_path_input)
similarity_scores = audio_fingerprint_saved.compare_with_saved_fingerprints(audio_data_input)

percentage_similarity, jaccard_similarity = audio_fingerprint_saved.calculate_jaccard_similarity(similarity_scores,
                                                                             audio_fingerprint_saved.fingerprints)
print(f"Jaccard Similarity: {jaccard_similarity}")
print(f"Percentage Similarity: {percentage_similarity}%")
