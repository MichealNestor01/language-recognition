import soundfile

from modules.features import *

def get_features(file):
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate
        chronogram = feature_chronogram(waveform, sample_rate)
        melspectrogram = feature_melspectrogram(waveform, sample_rate)
        mfc_coefficients = feature_mfcc(waveform, sample_rate)

        feature_matrix = np.array([])
        feature_matrix = np.hstack((chronogram, melspectrogram, mfc_coefficients))

        return feature_matrix