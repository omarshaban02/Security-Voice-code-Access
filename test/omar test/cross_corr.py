from __future__ import division
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import spectrogram, convolve2d, convolve, fftconvolve
import matplotlib.pyplot as plt
import sounddevice as sd


def record_audio(duration=3, filename="recording.wav"):
    """
    Records audio for `duration` seconds and saves it to `filename`.
    """
    fs = 44100  # Sampling rate
    recording = sd.rec(duration * fs, samplerate=fs, channels=2)
    sd.wait()
    wav.write(filename, fs, recording)
    return recording, fs


# Load the wave file
saved_voice = wav.read("open1.wav")
rate2 = saved_voice[0]
data2 = saved_voice[1]

# Zero-padding to increase the length of the signal
data2_padded = np.pad(data2, (0, max(0, 256 - len(data2))), 'constant')

# Compute the spectrogram
f2, t2, Sxx2 = spectrogram(data2_padded, fs=rate2, nperseg=256)

data1, rate1 = record_audio(duration=4, filename="not_omar.wav")
data1 = data1.flatten()

# Zero-padding to increase the length of the signal
desired_length = 256
data1_padded = np.pad(data1, (0, max(0, desired_length - len(data1))), 'constant')

f1, t1, Sxx1 = spectrogram(data1_padded, fs=rate1, nperseg=desired_length)


def cross_correlation(signal1, signal2):
    return np.correlate(signal1, signal2, mode='full')


def cross_correlation_coefficient(signal1, signal2):
    # Compute means
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)

    # Compute cross-correlation
    cross_corr = cross_correlation(signal1 - mean1, signal2 - mean2)

    # Compute cross-correlation coefficient
    corr_coeff = cross_corr[len(cross_corr) // 2] / (
        np.sqrt(np.sum((signal1 - mean1) ** 2) * np.sum((signal2 - mean2) ** 2)))

    return corr_coeff


corr_coeff_result = cross_correlation_coefficient(Sxx1, Sxx2)
print("correlation coefficient", corr_coeff_result)

#
## Plot the spectrogram
# plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title('Spectrogram')
# plt.colorbar(label='Power/Frequency (dB/Hz)')
# plt.show()
