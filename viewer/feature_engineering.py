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
#from mne_connectivity import spectral_connectivity_epochs
from scipy import stats
from scipy import integrate
from scipy.signal import welch, correlate
from scipy.stats import kruskal
from scipy.fft import dct, idct

import viewer.continuous as continuous
from MFDFA import MFDFA
import librosa
from viewer.segment_data import make_views
from viewer.denoising_data import denoise_signals
np.random.seed(0)


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


def signal_features(x, fs, nfft, low_freq, hi_freq):

    x = x.reshape(-1, )  # To make sure x is a 1D array
    x = x.astype(np.float32)

    if ~np.isfinite(x).any():
        print("There is non-numeric value in input to signal_features()")
        print(np.argwhere(~np.isfinite(x)))
        sys.exit("Exit the program because the output will not be valid.")

    #print(f"x shape is: {x.shape}")
    output = []
    '''
    Time Domain Features
    '''
    x_mean_value = np.mean(x)
    output.append(x_mean_value)  # Mean
    output.append(
        np.var(x))  # Variance or Activity parameter, the variance of the
    # time function, can indicate the surface of power spectrum in frequency domain.
    output.append(stats.mode(
        x,
        nan_policy='raise')[0][0])  # Mode: the most repeated value in signal
    output.append(np.median(x))  # Median
    output.append(
        stats.skew(x)
    )  # Skewness: For normally distributed data, the skewness should be about zero.
    #  For unimodal continuous distributions, a skewness value greater than zero means that there
    #  is more weight in the right tail of the distribution.
    output.append(
        stats.kurtosis(x)
    )  # Kurtosis: The fourth central moment divided by the square of the variance.0.0 for a normal distribution.
    output.append(
        np.sqrt(np.mean(x**2))
    )  # RMS. Note: The peak height in the power spectrum is an estimate of the RMS amplitude.
    output.append(np.sum(np.abs(
        np.diff(x))))  # Line Length:  the total vertical length of the signal
    output.append(
        np.round(continuous.get_h_mvn(x), 3)
    )  # compute the entropy from the determinant of the multivariate normal distribution
    output.append(ant.app_entropy(x, order=10,
                                  metric='chebyshev'))  # Approximate entropy
    output.append(ant.sample_entropy(x, order=10,
                                     metric='chebyshev'))  # Sample entropy
    print(ant.sample_entropy(x, order=10,
                                     metric='chebyshev'))
    output.append(ant.perm_entropy(x, order=10,
                                   normalize=True))  # Permutation entropy
    output.append(ant.svd_entropy(x, order=10, delay=1,
                                  normalize=True))  # SVD entropy
    x_binary = np.zeros_like(x)
    x_binary[np.where(x >= x_mean_value)] = 1
    output.append(ant.lziv_complexity(
        x_binary,
        normalize=True))  #Lempel-Ziv (LZ) complexity of (binary) sequence.
    output.append(ant.hjorth_params(x)[0])  # Hjorth Mobility
    output.append(ant.hjorth_params(x)[1])  # Hjorth Complexity
    '''
    Frequency Domain Features
    '''
    Freq, PSD = get_Freq(x, fs, nfft, low_freq, hi_freq)
    output.append(Freq[np.argmax(PSD)] * 60)  # Dominant frequency in CPM
    DP = np.max(PSD)
    output.append(
        DP)  # Dominant power, maximum magnitude of power spectrum density
    output.append(
        np.sum(np.where(PSD > (DP / 4))) /
        len(PSD))  # Percentage of PSD that has higher value than DP/4
    output.append(
        ant.spectral_entropy(x,
                             fs,
                             method='welch',
                             nperseg=len(x),
                             normalize=True))  # Spectral entropy
    output.append(bandpower(Freq, PSD, (3 / 60, 7 / 60),
                            relative=True))  # Bandpower between 3-7 cpm
    output.append(bandpower(Freq, PSD, (7 / 60, 11 / 60),
                            relative=True))  # Bandpower between 7-11 cpm
    output.append(bandpower(Freq, PSD, (11 / 60, 15 / 60),
                            relative=True))  # Bandpower between 11-15 cpm

    output.append(crest_factor(PSD))  # Crest factor of PSD
    output.append(
        get_PSD_Freq_percentiles(Freq, PSD)[0]
    )  # Median Frequency, Frequency where the area under PSD is 50% of total power.
    output.append(get_PSD_Freq_percentiles(
        Freq, PSD)[1])  # Mean power frequency (MPF)
    '''
    Fractal Features
    '''
    output.append(ant.petrosian_fd(x))  # Petrosian fractal dimension
    output.append(ant.katz_fd(x))  # Katz fractal dimension
    #output.append(get_MFDFA(x.flatten()))  # Hurst index
    output.extend(get_MFCC(x,fs)) #Top 5 MFCC 

    return np.asarray(output)


def feature_wrapper(x, fs, nfft, low_freq, hi_freq):

    return np.apply_along_axis(signal_features, 0, x, fs, nfft, low_freq,
                               hi_freq)


def get_feature_df_single_channel(df, win_size, step_size, fs, nfft, low_freq,
                                  hi_freq, feature_names, state, ch, ID):

    windows = make_views(df, win_size, step_size)
    features = feature_wrapper(windows[:, :, 0], fs, nfft, low_freq, hi_freq)
    features = pd.DataFrame(data=features.T, columns=feature_names)
    features["state"] = state
    features["channel"] = ch
    features["ID"] = ID

    return features

def build_features_dataframe(dataset_records,features_names,f_low,f_high,fs,save=False,path=None):

    data_features_df = pd.DataFrame(columns=features_names)
    clean_dataset = denoise_signals(dataset_records,'ibdct',f_low,f_high,fs)
    #fix this code to be generic and at the same time include the labels
    for i,record in enumerate(clean_dataset):
        data = record[0]
        label = record[1]
        for j in range(len(data)):
            ch_df = get_feature_df_single_channel(data[j],120,60,fs,0,f_low,f_high,features_names,label,(j+1),(i+1))
            data_features_df = pd.concat([data_features_df,ch_df])

    if save:
        if path:
            data_features_df.to_csv(path,index=False)
        else:
            data_features_df.to_csv('Features_data_df.csv',index=False)

    return data_features_df



