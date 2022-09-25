from types import MethodType
import numpy as np
from scipy.fftpack import dct, idct
from scipy import signal

def ibdct(signal,f_low,f_high,fs):
    '''
    removes noise from signal using the index based discrete cosine transform method.
    Parameters
    ----------
    sig: numpy array of input signal.

    Returns
    -------
    filtered_signal: denoised signal after applying IBDCT to it.
    '''
    signal_length = len(signal)
    signal_dct = dct(signal,norm='ortho')
    k_low = int((2*signal_length*f_low)/fs)
    k_high = int((2*signal_length*f_high)/fs)
    signal_dct[:k_low] = 0
    signal_dct[k_high:] = 0
    filtered_signal = idct(signal_dct,norm='ortho')
    return filtered_signal

def butterworth_filter(signal,f_low,f_high,fs):
    sos = signal.butter(10, [f_low,f_high], 'bandpass', fs=fs, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, signal)
    return filtered_signal

def get_freq_response(filter,output_type='ba', type='iir',analog=False):
    if analog:
        if output_type == 'ba':
            b,a = filter
            w,h = signal.freqs(b, a)
        
        elif output_type == 'zpk':
            z,p,k = filter
            w,h = signal.freqs_zpk(z, p, k)
        else:
            raise ValueError('Analog Filter output type must be ba or zpk.')


    else:
        if type.lower() == 'iir':
            if output_type == 'ba':
                b,a = filter
                w,h = signal.freqz(b, a)
        
            elif output_type == 'zpk':
                z,p,k = filter
                w,h = signal.freqz_zpk(z, p, k)

            elif output_type == 'sos':
                w,h = signal.sosfreqz(filter)
        
        elif type.lower() == 'fir':
                w,h = signal.freqz(b=filter)
        
        else:
            raise ValueError('Invalid filter type. Please enter iir or fir')
    
    return w,h



def create_filter(fs,f_low=None,f_high=None,order=2,method='iir', 
                    rp=None, rs=None, btype='bandpass', analog=False, ftype='butter', 
                    window='hamming', scale=True,output='ba'):
    if f_low <= 0 or f_high<=0:
        raise ValueError('Frequency must be a positive number.')
    if f_high >= (fs/2) or f_low >= (fs/2):
        raise ValueError('Frequency cannot be greater than fs/2.')
    if f_low >= f_high:
        print('f_low must be less than f_high. Reversing the values now.')
        temp = f_low
        f_low = f_high
        f_high = temp
 
    if f_low is None:
        if btype.lower() == 'lowpass' or btype.lower() == 'highpass':
            Wn = f_high
        else:
            Wn = (0,f_high)

    elif f_high is None:
        if btype.lower() == 'lowpass':
            Wn = f_low
        elif btype.lower() == 'highpass':
            Wn = (fs/2)-0.00001
        else:
            Wn = (f_low,(fs/2)-0.00001)

    elif f_low is None and f_high is None:
        if btype.lower() == 'lowpass' or btype.lower() == 'highpass':
            Wn = (fs/2)-0.00001
        else:
            Wn = (0,(fs/2)-0.00001)
    else:
        if btype.lower() == 'lowpass':
            Wn = f_low
        elif btype.lower() == 'highpass':
            Wn = f_high
        else:
            Wn = (f_low,f_high)

    print(Wn)

    if method.lower() == 'iir':
        if output == 'ba':
            if analog:
                b,a = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, analog=analog, 
                                ftype=ftype, output=output)
                filter = (b,a)

            else:
                b,a = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, analog=analog, 
                                    ftype=ftype, output=output, fs=fs)
                filter = (b,a)
            w,h = get_freq_response(filter,output,analog=analog,type=method)
            


        elif output == 'zpk':
            if analog:
                z,p,k=signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, analog=analog, 
                                ftype=ftype, output=output)
                filter = (z,p,k)

            else:
                    
                z,p,k = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, analog=analog, 
                                    ftype=ftype, output=output, fs=fs)
                filter = (z,p,k)
            w,h = get_freq_response(filter,output,analog=analog,type=method)

        elif output == 'sos':
            if analog:
                filter = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, analog=analog, 
                                ftype=ftype, output=output)

            else:
                filter = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, analog=analog, 
                                    ftype=ftype, output=output, fs=fs)
            w,h = get_freq_response(filter,output,analog=analog,type=method)

        else:
            raise ValueError('Invalid output type. Please enter ba or zpk or sos')

    elif method.lower() == 'fir':

        filter = signal.firwin((order+1),Wn,window=window, pass_zero=btype, scale=scale,fs=fs)
        w,h = get_freq_response(filter,output,analog=False,type=method)
    
    else:
        raise ValueError('the method entered is not a valid method. Please enter iir or fir')
    return filter, w,h

