from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, sr, lowcut=1000, highcut=8000):
    b, a = butter_bandpass(lowcut, highcut, sr, order=4)
    filtered = filtfilt(b, a, data)
    return filtered