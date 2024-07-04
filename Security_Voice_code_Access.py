# Import necessary libraries
import joblib
import pandas as pd
import numpy as np
import librosa
from scipy.signal import stft, find_peaks
import sounddevice as sd
import wavio as wav
from sklearn.metrics import accuracy_score
from scipy.signal import spectrogram

import pyaudio
import wave


class Sound(object):
    def __init__(self):
        self._file_path = "tst.wav"
        self._data = np.ndarray([])
        self._sr = 44100
        self._p_features = pd.DataFrame()
        self._w_features = pd.DataFrame()
        self._spectro = np.ndarray([])

    @property
    def file_path(self):
        return self._file_path

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sampling_rate):
        self._sr = sampling_rate

    @property
    def p_features(self):
        return self._p_features

    @p_features.setter
    def p_features(self, value):
        self._p_features = value

    @property
    def w_features(self):
        return self._w_features

    @w_features.setter
    def w_features(self, value):
        self._w_features = value

    @property
    def spectro(self):
        self._spectro = spectrogram(self.data, self.sr)
        return self._spectro

    def record(self, duration=2, out_file="tst.wav"):
        FRAMES_PER_BUFFER = 3200
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        print("Start recording...")

        seconds = duration
        frames = []
        end = int(RATE / FRAMES_PER_BUFFER * seconds)
        for i in range(0, end):
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(data)
        print("End recording")

        stream.stop_stream()
        stream.close()

        obj = wave.open(out_file, "wb")
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(p.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b"".join(frames))
        obj.close()

        self.file_path = out_file

    def extract_features(self):
        # Load audio file
        self.data, self.sr = librosa.load(self.file_path, sr=44100)

        # Extract some audio features (e.g., mean and standard deviation of MFCCs)
        mfccs = librosa.feature.mfcc(y=self.data, sr=self.sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)
        mfccs_std = mfccs.std(axis=1)

        _, _, sxx = stft(self.data, self.sr, nfft=512)

        sxx = np.abs(sxx)

        melsxx = librosa.feature.melspectrogram(y=self.data, sr=self.sr)

        # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        combined_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

        # Extract pitch
        pitches, magnitudes = librosa.core.piptrack(y=self.data, sr=self.sr)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])

        # Extract Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=self.data)

        # Extract spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=self.data, sr=self.sr)

        peaks, _ = find_peaks(sxx.flatten())

        # Extract Energy
        energy = librosa.feature.rms(y=self.data)

        # Calculate the spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(S=sxx)[0]

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.data, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.data, sr=self.sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=self.data)
        chroma_stft = librosa.feature.chroma_stft(y=self.data, sr=self.sr, n_fft=512)

        # Compute the tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=self.data, sr=self.sr)

        # -----------------------------------

        p_features = zcr.flatten().tolist() + \
            mfccs.flatten().tolist() + \
            mfccs_mean.flatten().tolist() + \
            mfccs_std.flatten().tolist() + \
            pitch.flatten().tolist() + \
            energy.flatten().tolist()

        w_features = sxx.flatten().tolist() + \
            melsxx.flatten().tolist() + \
            zcr.flatten().tolist() + \
            combined_mfccs.flatten().tolist() + \
            spectral_contrast.flatten().tolist() + \
            spectral_centroid.flatten().tolist() + \
            spectral_bandwidth.flatten().tolist() + \
            spectral_rolloff.flatten().tolist() + \
            spectral_flatness.flatten().tolist() + \
            chroma_stft.flatten().tolist() + \
            mfccs_mean.flatten().tolist() + \
            mfccs_std.flatten().tolist() + \
            pitch.flatten().tolist() + \
            energy.flatten().tolist() + \
            tempo.flatten().tolist()

        self.p_features, self.w_features = pd.DataFrame(p_features).T, pd.DataFrame(w_features).T

    @file_path.setter
    def file_path(self, value):
        self._file_path = value


class Model(object):

    def __init__(self, pkl_path):
        self._model_path = pkl_path
        self._model = joblib.load(pkl_path)
        self.p_w_state = 'p'
        self._sound = Sound()
        self._result = ""

    @property
    def model(self):
        return self._model

    @property
    def sound(self):
        return self._sound

    @sound.setter
    def sound(self, s):
        self._sound = s

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, r):
        self._result = r

    def make_prediction(self):
        pred = ()
        if self.p_w_state == 'p':
            p_pred_proba = self.model.predict_proba(self.sound.p_features)[0]
            p_pred = np.argmax(p_pred_proba)
            pred = (p_pred, p_pred_proba)
        elif self.p_w_state == 'w':
            w_pred_proba = self.model.predict_proba(self.sound.w_features)[0]
            w_pred = np.argmax(w_pred_proba)
            pred = (w_pred, w_pred_proba)
        if pred:
            return pred
