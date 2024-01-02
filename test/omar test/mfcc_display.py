import librosa
import numpy as np
from scipy.spatial.distance import cosine
from scipy.signal import convolve2d


# Load audio files
audio_file = 'unlock1.wav'
audio_file2 = 'open2.wav'
y, sr = librosa.load(audio_file, sr=None)
y2, sr2 = librosa.load(audio_file2, sr=None)

y_normalized = librosa.util.normalize(y)
y_normalized2 = librosa.util.normalize(y2)


# Calculate MFCCs and delta coefficients
mfccs = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=13)
delta_mfccs = librosa.feature.delta(mfccs)
combined_mfccs = np.concatenate((mfccs, delta_mfccs), axis=0)

mfccs2 = librosa.feature.mfcc(y=y_normalized2, sr=sr2, n_mfcc=13)
delta_mfccs2 = librosa.feature.delta(mfccs2)
combined_mfccs2 = np.concatenate((mfccs2, delta_mfccs2), axis=0)

# Display the combined MFCCs
librosa.display.specshow(combined_mfccs, x_axis='time')
librosa.display.specshow(combined_mfccs2, x_axis='time')

# Reshape combined MFCCs for cosine similarity calculation
combined_mfccs_flat = combined_mfccs.flatten()
combined_mfccs2_flat = combined_mfccs2.flatten()

# Calculate cosine similarity
similarity_score = 1 - cosine(combined_mfccs_flat, combined_mfccs2_flat)

# Print the similarity score
print(f"Cosine Similarity Score: {similarity_score * 100:.2f}%")


# Calculate 2D convolution
convolution_result = convolve2d(combined_mfccs, combined_mfccs2, mode='valid')

# Find the index of the maximum convolution result
max_convolution_index = np.unravel_index(np.argmax(convolution_result), convolution_result.shape)

# Shift one signal to align with the other
shifted_mfccs2 = np.roll(combined_mfccs2, max_convolution_index, axis=(0, 1))

# Calculate similarity score based on the shifted signal
similarity_score = np.corrcoef(combined_mfccs.flatten(), shifted_mfccs2.flatten())[0, 1]

# Print the similarity score
print(f"2D Convolution Similarity Score: {similarity_score * 100:.2f}%")

#
#
# # Calculate cosine similarity for each frame
# frame_similarities = [1 - cosine(combined_mfccs[:, i], combined_mfccs2[:, i]) for i in range(min(mfccs.shape[1], mfccs2.shape[1]))]
#
# # Aggregate frame similarities (e.g., take the mean)
# mean_similarity = np.mean(frame_similarities)
#
# # Print the mean similarity score
# print(f"Enhanced Cosine Similarity Score: {mean_similarity * 100:.2f}%")