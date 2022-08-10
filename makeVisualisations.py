import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import librosa.display
import soundfile
import os
# matplotlib complains about the behaviour of librosa.display, so we'll ignore those warnings:
import warnings; warnings.filterwarnings('ignore')

audio = soundfile.SoundFile('./dataset/archive(1)/test/test/de_f_63f5b79c76cf5a1a4bbd1c40f54b166e.fragment1.flac') 
waveform = audio.read(dtype="float32")
sample_rate = audio.samplerate
plt.figure(figsize=(15,30))
plt.subplot(5, 1, 1)
librosa.display.waveshow(waveform, sr=sample_rate)
plt.title('TEST WAVEFORM')

#STFT visualisation
stft_spectrum_matrix = librosa.stft(waveform)
plt.subplot(5, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_spectrum_matrix), ref=np.max), y_axis='log', x_axis='time')
plt.title('STFT Transformation Power Spectogram')
plt.colorbar(format='%+2.0f dB')

#MFCC visualisation 
mfc_coefficients = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
plt.subplot(5, 1, 3)
librosa.display.specshow(mfc_coefficients, x_axis='time', norm=Normalize(vmin=-30,vmax=30))
plt.colorbar()
plt.yticks(())
plt.ylabel('MFC Coefficient')
plt.title('MFC Coefficients')

#MEL spectogram 
melspectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000)
plt.subplot(5, 1, 4)
librosa.display.specshow(librosa.power_to_db(S=melspectrogram, ref=np.mean), y_axis='mel', fmax=8000, x_axis='time', norm=Normalize(vmin=-20, vmax=20))
plt.colorbar(format='%+2.0f dB', label='Amplitude')
plt.ylabel('Mels')
plt.title('Mels Spectogram')

#chronogram
chronogram = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
plt.subplot(5, 1, 5)
librosa.display.specshow(chronogram, y_axis='chroma', x_axis='time')
plt.colorbar(label='Relative intensity')
plt.title('Chronogram')

plt.tight_layout()
plt.subplots_adjust(hspace=0.521, top=0.974, bottom=0.052)
plt.show()
