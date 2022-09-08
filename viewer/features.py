import os
import sys
import glob
import copy
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import pyns
from time import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

import antropy as ant
from scipy import stats
from scipy import integrate
from scipy.signal import welch, correlate
from scipy.stats import kruskal
from scipy.fft import dct, idct

import continuous
from MFDFA import MFDFA
import librosa
from ssqueezepy import ssq_cwt, ssq_stft


def get_MFCC(array,sampling_rate=2):
    """
    Calculates the Mel-Frequency Cepstral Coefficients (MFCCs) of the signal
    Parameters
    -----------------
    array: array 
    array of signal values to be evaluated
    sampling_rate:
    sampling rate of the audio signal
    Returns
    -----------------
    mfccs: list 
    flattened list of means of coefficients 
    """
    mfccs = librosa.feature.mfcc(y=array, sr=sampling_rate,n_mfcc=5)
    mfccs = np.mean(mfccs,axis=1)
    return mfccs.flatten()

def get_Freq(x, fs, nfft, low_freq, hi_freq):

    if ~np.isfinite(x).any():
        print("There is non-numeric value in input to get_Freq()")
        print(np.argwhere(~np.isfinite(x)))
        sys.exit("Exit the program because the output will not be valid.")

    nperseg = len(x)
    if nfft < nperseg:
        print(f'nfft is : {nfft}')
        print(f'nperseg is : {nperseg}')
        nfft = nperseg
    freqs, Pxx_den = welch(x,
                           fs,
                           nperseg=len(x),
                           noverlap=int(0.5 * nperseg),
                           nfft=nfft,
                           scaling="density",
                           detrend=False,
                           average="mean")
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low_freq, freqs <= hi_freq)

    return freqs[idx_band], Pxx_den[idx_band]


def bandpower(freqs, psd, band, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """

    if ~np.isfinite(psd).any():
        print("There is non-numeric value in input to bandpower()")
        print(np.argwhere(~np.isfinite(psd)))
        sys.exit("Exit the program because the output will not be valid.")

    band = np.asarray(band)
    low, high = band

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    #print(freq_res)

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = integrate.simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= (integrate.simpson(psd, dx=freq_res) + np.finfo(float).eps)

    if ~np.isfinite(bp).any():
        print("There is non-numeric value in output bandpower()")
        print(bp)
        print(np.argwhere(~np.isfinite(bp)))
        sys.exit("Exit the program because the output will not be valid.")

    return bp


def crest_factor(x):
    if ~np.isfinite(x).any():
        print("There is non-numeric value in input to crest_factor()")
        print(np.argwhere(~np.isfinite(x)))
        sys.exit("Exit the program because the output will not be valid.")

    return np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x)))

def closest(lst, K):
    lst = np.asarray(lst)
    if ~np.isfinite(lst).any():
        print("There is non-numeric value in input to closest()")
        print(np.argwhere(~np.isfinite(lst)))
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]
    
def get_PSD_Freq_percentiles(f, P):

    Area = integrate.cumulative_trapezoid(P, f, initial=0)
    Ptotal = Area[-1]
    mpf = integrate.cumulative_trapezoid(f * P,
                                         f) / Ptotal  # mean power frequency
    medianF = f[np.argwhere(Area == closest(Area, Area[-1] / 2))][0][0]

    return medianF, mpf[-1]


def get_MFDFA(x, plot=False):

    # Select a band of lags, which usually ranges from
    # very small segments of data, to very long ones, as
    lag = np.unique(np.logspace(3, 4, 100).astype(int))
    # Notice these must be ints, since these will segment
    # the data into chucks of lag size

    # Select the power q
    q = 2

    # The order of the polynomial fitting
    order = 1

    # Obtain the (MF)DFA as
    lag, dfa = MFDFA(x, lag=lag, q=q, order=order)
    # To uncover the Hurst index, lets get some log-log plots
    if plot:
        plt.loglog(lag, dfa, 'o', label='fOU: MFDFA q=2')

    # And now we need to fit the line to find the slope. Don't
    # forget that since you are plotting in a double logarithmic
    # scales, you need to fit the logs of the results
    H_hat = np.polyfit(np.log(lag), np.log(dfa), 1)[0]

    # Now what you should obtain is: slope = H + 1
    #print('Estimated H = ' + '{:.3f}'.format(H_hat[0]))

    return np.round(H_hat[0], 3)

def get_spectral_entropy(signal,sampling_rate,method,normalize=True,axis=-1):
    spectral_entropy = ant.spectral_entropy(signal,sampling_rate,method=method,nperseg=len(signal),
    normalize=normalize,axis=axis)
    return spectral_entropy






