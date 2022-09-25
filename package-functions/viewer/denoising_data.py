import numpy as np
from viewer.segment_data import separate_data_chs
from viewer.filters import ibdct, butterworth_filter

def denoise_signals(dataset_records,denoising_method,f_low=None,f_high=None,fs=2,n_chs=3):
    
    denoised_dataset = []
    if denoising_method.lower() == 'ibdct':
        for i,record in enumerate(dataset_records):
            clean_chs = []
            record_chs = separate_data_chs(record,n_chs)
            for ch in record_chs:
                clean_chs.append(ibdct(ch,f_low=f_low,f_high=f_high,fs=fs))
            print(np.array(clean_chs).shape)
            denoised_dataset.append(np.array(clean_chs).T)
            print(len(denoised_dataset))
    elif denoising_method.lower() == 'butterworth':
        for i,record in enumerate(dataset_records):
            clean_chs = []
            record_chs = separate_data_chs(record,n_chs)
            for ch in record_chs:
                clean_chs.append(butterworth_filter(ch,f_low=f_low,f_high=f_high,fs=fs))
            denoised_dataset.append(np.array(clean_chs).T)
    else:
        raise ValueError('Denoising method does not exist. Please enter ibdct or butterworth')

    return denoised_dataset
        


