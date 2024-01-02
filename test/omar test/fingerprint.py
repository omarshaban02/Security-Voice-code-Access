import numpy as np
from scipy.io import wavfile as wav
from scipy.signal import spectrogram, find_peaks
import hashlib
import sounddevice as sd


class AudioFingerprint:
    def __init__(self, song_id, file_path):
        self.song_id = song_id
        self.fingerprints = []
        self.file_path = file_path
        self.rate, self.audio_data = self.read_audio(self.file_path)

    def read_audio(self, file_path):
        rate, audio_data = wav.read(file_path)
        return rate, audio_data.T

    def record_audio(self, duration=3, filename="recording.wav"):
        fs = 44100  # Sampling rate
        recording = sd.rec(duration * fs, samplerate=fs, channels=2)
        print("Recording starts...")
        sd.wait()
        print("Recording ends...")
        wav.write(filename, fs, recording)
        return fs, recording.T
    
    def create_spectrogram(self, audio_data):
        _, _, spectrogram_data = spectrogram(audio_data, fs=self.rate)
        return spectrogram_data

    def find_spectrogram_peaks(self, spectrogram_data):
        peaks, _ = find_peaks(np.log1p(spectrogram_data.ravel()), height=5)
        return peaks

    def hash_peaks(self, peaks, max_time_difference=100):
        hash_values = []
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                time_diff = peaks[j] - peaks[i]
                if 0 < time_diff <= max_time_difference:
                    hash_object = hashlib.sha1(f"{peaks[i]}|{peaks[j]}|{time_diff}".encode())
                    hash_values.append(hash_object.hexdigest()[:20])
        return hash_values

    def add_fingerprints(self, fingerprints):
        self.fingerprints.extend(fingerprints)

    def compare_with_saved_fingerprints(self, input_audio_data):
        input_spectrogram = self.create_spectrogram(input_audio_data)
        input_peaks = self.find_spectrogram_peaks(input_spectrogram)
        input_hashes = self.hash_peaks(input_peaks)

        # Compare input hashes with saved fingerprints
        similarity_scores = []
        for saved_offset, saved_hash in self.fingerprints:
            for input_hash in input_hashes:
                if saved_hash == input_hash:
                    similarity_scores.append((saved_offset, input_hash))

        # You may want to further process similarity scores to get a meaningful result
        return similarity_scores

    def calculate_jaccard_similarity(self, similarity_scores, fingerprints_saved):
        saved_hashes = set([hash for _, hash in fingerprints_saved])
        input_hashes = set([hash for _, hash in similarity_scores])

        intersection_size = len(saved_hashes.intersection(input_hashes))
        union_size = len(saved_hashes.union(input_hashes))

        jaccard_similarity = intersection_size / union_size if union_size != 0 else 0
        percentage_similarity = jaccard_similarity * 100

        return percentage_similarity

    def save_fingerprints(self):
        # In a real-world scenario, you might want to save fingerprints to a database or file.
        print("Fingerprints saved.")


# print("Start of Test")
#
# file_path = 'unlock1.wav'
#
# rate, audio_data = wav.read(file_path)
# print("len of input audio data", audio_data.shape)
# _, _, spectrogram_data = spectrogram(audio_data.T, fs=rate)
#
# print(spectrogram_data[1])
#
# print("End of Test")

# Example Usage
file_path_saved = 'unlock2.wav'
file_path_input = 'open1.wav'
saved_song_id = 1

audio_fingerprint_saved = AudioFingerprint(saved_song_id, file_path_saved)
rate_saved, audio_data_saved = audio_fingerprint_saved.read_audio(file_path_saved)
audio_fingerprint_saved.rate = rate_saved
spectrogram_data_saved = audio_fingerprint_saved.create_spectrogram(audio_data_saved)
peaks_saved = audio_fingerprint_saved.find_spectrogram_peaks(spectrogram_data_saved)
hash_values_saved = audio_fingerprint_saved.hash_peaks(peaks_saved)
fingerprints_saved = [(offset, fingerprint_hash) for offset, fingerprint_hash in enumerate(hash_values_saved)]
audio_fingerprint_saved.add_fingerprints(fingerprints_saved)

# Simulate a new recording for input
rate_input, audio_data_input = audio_fingerprint_saved.read_audio(file_path_input)
# rate_input, audio_data_input = audio_fingerprint_saved.record_audio(4, "not_omar3.wav")
similarity_scores = audio_fingerprint_saved.compare_with_saved_fingerprints(audio_data_input)

# Calculate and print the percentage similarity
percentage_similarity = audio_fingerprint_saved.calculate_jaccard_similarity(similarity_scores, fingerprints_saved)
print(f"Percentage Similarity: {percentage_similarity}%")
