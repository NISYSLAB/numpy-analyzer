import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy import stats
from scipy.stats import scoreatpercentile as q
from collections import defaultdict


def resampled_signal_length(up, down, original_length):
    ratio = float(up) / float(down)
    return max(int(round(ratio * original_length)), 1)

def resample_signal(signal,number_of_samples=None,upsample=1.0,downsample=1.0,window=None,axis=-1):
    # first pad signal, then apply resampling, then return resampled signal
    signal = np.array(signal)
    new_length = number_of_samples
    if (upsample != 1.0) or (downsample != 1.0):
        new_length = resampled_signal_length(upsample,downsample,signal.shape[axis])
    new_signal = resample(signal,new_length,window=window,axis=axis)
    return new_signal


def describe(ch,name,cols_dict,id=None):
    cols_dict['name'].append(name)
    cols_dict['unit'].append(ch.dtype)
    cols_dict['length'].append()
    cols_dict['min'].append(np.min(ch))
    cols_dict['max'].append(np.max(ch))
    cols_dict['mean'].append(np.mean(ch))
    cols_dict['Q1'].append(q(ch,25))
    cols_dict['median'].append(np.median(ch))
    cols_dict['Q3'].append(q(ch,75))
    cols_dict['mode'].append(np.mode(ch))
    cols_dict['variance'].append(np.var(ch))
    cols_dict['skew'].append(stats.skew(ch))
    cols_dict['kurtosis'].append(stats.kurtosis(ch))
    cols_dict['DF'].append()
    cols_dict['DP'].append()
    if id:
        cols_dict['id'].append(id)

    return cols_dict


def describe_signal(signal,n_ch,ch_names=None,ch_structure='v',get_dataframe=True,id=None):
    cols = defaultdict(list)
    if ch_structure.lower() !='v' and ch_structure.lower()!='h':
        raise ValueError('Please provide a valid channel structure. Enter v for vertical or h for horizontal')

    for i in range(n_ch):
        if ch_structure.lower() =='v':
            ch = signal[:,i]
        elif ch_structure.lower() =='h':
            ch = signal[i,:]

        if ch_names:
            data = describe(ch,ch_names[i],cols)
        else:
            data = describe(ch,(i+1),cols)

        cols = data

    if get_dataframe:
        df = pd.DataFrame(cols)
        return df

    return cols



def describe_dataset(dataset,n_ch,ch_names=None,ch_structure='v',get_dataframe=True):
    cols = defaultdict(list)
    for j in range(len(dataset)):
        data = describe_signal(dataset[j],n_ch,ch_names=ch_names,ch_structure=ch_structure,get_dataframe=get_dataframe,id=(j+1))
        cols = data

    if get_dataframe:
        df = pd.DataFrame(cols)
        return df

    return cols
    

