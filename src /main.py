import os
from bandpass import apply_bandpass_filter
from denoise import apply_noise_reduction
from visualize import plot_waveform_compare, plot_spectrogram_compare
import librosa
import soundfile as sf

INPUT_PATH = "data/input/XC128013.ogg"
BANDPASS_PATH = "data/output/bandpassed.wav"
DENOISED_PATH = "data/output/denoised.wav"

def main():

    print("📥 Loading audio ...")
    audio, sr = librosa.load(INPUT_PATH, sr=None)

    ##
    # 1. Bandpass Filter (1–8 kHz)
    ##
    print("🎚 Applying Bandpass Filter ...")
    bandpassed = apply_bandpass_filter(audio, sr, lowcut=1000, highcut=8000)
    sf.write(BANDPASS_PATH, bandpassed, sr)

    ##
    # 2. Noise Reduction (Spectral Gating – noisereduce)
    ##
    print("🔉 Applying Noise Reduction ...")
    denoised = apply_noise_reduction(bandpassed, sr)
    sf.write(DENOISED_PATH, denoised, sr)

    ##
    # 3. Visualization
    ##
    print("📊 Generating waveform & spectrogram plots ...")
    plot_waveform_compare(audio, denoised, sr, "results/waveform_before_after.png")
    plot_spectrogram_compare(audio, denoised, sr, "results/spectrogram_before_after.png")

    print("✅ All done! Check data/output/ and results/ folders.")

if __name__ == "__main__":
    main()