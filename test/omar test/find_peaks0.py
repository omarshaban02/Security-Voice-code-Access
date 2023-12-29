import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
import matplotlib.pyplot as plt

# Load two voices in wav format
voice1_rate, voice1_data = wavfile.read('D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/open1.wav')
voice2_rate, voice2_data = wavfile.read('D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/open2.wav')

voice2_data = voice2_data.T

_, _, spec1 = spectrogram(voice1_data, window='boxcar')
_, _, spec2 = spectrogram(voice2_data, window='boxcar')

peaks1, _ = find_peaks(spec1.flatten())
peaks2, _ = find_peaks(spec2.flatten())


def visualize_spectrogram(spec):
    plt.imshow(np.log1p(spec), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


# # Visualize spectrograms
# visualize_spectrogram(spec1)
# visualize_spectrogram(spec2)

# Compare the percentage of common peaks
common_peaks = len(set(peaks1) & set(peaks2))
total_peaks = max(len(peaks1), len(peaks2))
similarity_percentage = (common_peaks / total_peaks) * 100

print(f"Similarity Percentage: {similarity_percentage:.2f}%")
