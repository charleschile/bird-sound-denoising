import noisereduce as nr
import numpy as np

def apply_noise_reduction(audio, sr):
    # use first 0.5 seconds as noise profile
    noise_len = int(sr * 0.5)
    noise_sample = audio[:noise_len]

    reduced = nr.reduce_noise(
        y=audio,
        y_noise=noise_sample,
        sr=sr,
        prop_decrease=0.8
    )
    return reduced