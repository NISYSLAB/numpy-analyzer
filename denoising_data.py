import numpy as np
from scipy.fftpack import dct, idct
from scipy import signal
def ibdct(sig):
    signal = np.array(sig)
    l = len(signal)
    fs = 2
    f_low = 0.03
    f_high = 0.25
    signal_dct = dct(signal,norm='ortho')
    signal_altered = np.copy(signal_dct)
    k_low = int((2*l*f_low)/fs)
    k_high = int((2*l*f_high)/fs)
    signal_altered[:k_low] = 0
    signal_altered[k_high:] = 0
    final_signal = idct(signal_altered,norm='ortho')
    return final_signal

def butterworth_filter(sig):
    input_signal = np.array(sig)
    sos = signal.butter(10, [0.03,0.25], 'bandpass', fs=2, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, input_signal)
    return filtered_signal
