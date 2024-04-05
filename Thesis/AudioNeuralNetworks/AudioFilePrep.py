import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

file = "../Audio/blues.00000.wav"

# waveform
signal, Fs = librosa.load(file, sr=22050)  # Fs * t -> 22050 * 30 seconds = Total samples
# librosa.display.waveshow(signal, sr=Fs, color='blue')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
#plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, Fs, len(magnitude))

freq_Nyq = frequency[:int(len(frequency) / 2)]
magnitude_Nyq = magnitude[:int(len(frequency) / 2)]

# plt.plot(freq_Nyq, magnitude_Nyq)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
#plt.show()

# stft -> spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=Fs, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Freq")
# plt.colorbar()
# plt.show()

# MFCCs
MFFCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFFCs, sr=Fs, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFFC")
plt.colorbar()
plt.show()