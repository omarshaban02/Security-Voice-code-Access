import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
import matplotlib.pyplot as plt


def compute_spectrogram(audio):
    _, _, spec = spectrogram(audio, window='hamming')
    return spec


def dynamic_threshold(peaks, spec, local_avg_factor=0.5, local_std_factor=1.5):
    # Ensure peaks are within the valid range
    peaks = peaks[(peaks >= 0) & (peaks < len(spec))]

    if len(peaks) == 0:
        return np.array([])

    # Calculate local average and standard deviation
    local_avg = np.mean(spec)
    local_std = np.std(spec)

    # Adjust threshold based on local characteristics
    threshold = local_avg_factor * local_avg + local_std_factor * local_std

    # Keep only peaks above the dynamic threshold
    filtered_peaks = peaks[spec[peaks] > threshold]

    return filtered_peaks


def visualize_spectrogram(spec):
    plt.imshow(np.log1p(spec), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def main():
    # Load voice in wav format
    voice1_rate, voice1_data = wavfile.read('D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/open1.wav')
    voice2_rate, voice2_data = wavfile.read('D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/open2.wav')

    # Compute spectrogram
    spec1 = compute_spectrogram(voice1_data)
    spec2 = compute_spectrogram(voice2_data)

    # Find local maxima
    peaks1, _ = find_peaks(spec1.flatten())
    peaks2, _ = find_peaks(spec2.flatten())

    # Apply dynamic thresholding
    filtered_peaks1 = dynamic_threshold(peaks1, spec1.flatten())
    filtered_peaks2 = dynamic_threshold(peaks2, spec2.flatten())

    # Visualize spectrogram
    # visualize_spectrogram(spec1)
    # visualize_spectrogram(spec2)

    filtered_peaks1.sort()
    filtered_peaks2.sort()

    common = set(filtered_peaks1) & set(filtered_peaks2)
    print("1", set(filtered_peaks1))
    print("2", set(filtered_peaks2))
    print("common:", set(common))
    # Compare the percentage of common peaks after dynamic thresholding
    common_peaks = len(set(filtered_peaks1) & set(filtered_peaks2))
    total_peaks = max(len(filtered_peaks1), len(filtered_peaks2))
    similarity_percentage = (common_peaks / total_peaks) * 100

    print(f"Similarity Percentage: {similarity_percentage:.2f}%")


if __name__ == "__main__":
    main()
