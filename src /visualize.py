import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform_compare(before, after, sr, save_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(2,1,1)
    plt.title("Waveform - Before Denoising")
    librosa.display.waveshow(before, sr=sr)

    plt.subplot(2,1,2)
    plt.title("Waveform - After Denoising")
    librosa.display.waveshow(after, sr=sr)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram_compare(before, after, sr, save_path):
    plt.figure(figsize=(12, 8))

    # before
    plt.subplot(2,1,1)
    plt.title("Spectrogram Before")
    spec_before = librosa.amplitude_to_db(np.abs(librosa.stft(before)), ref=np.max)
    librosa.display.specshow(spec_before, sr=sr, x_axis="time", y_axis="hz", cmap="magma")

    # after
    plt.subplot(2,1,2)
    plt.title("Spectrogram After")
    spec_after = librosa.amplitude_to_db(np.abs(librosa.stft(after)), ref=np.max)
    librosa.display.specshow(spec_after, sr=sr, x_axis="time", y_axis="hz", cmap="magma")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()