import numpy as np
from viewer.segment_data import separate_data_chs
from viewer.filters import ibdct, butterworth_filter

def denoise_signals(dataset_records,denoising_method,f_low,f_high,fs):
    
    denoised_dataset = np.empty(len(dataset_records))
    if denoising_method.lower() == 'ibdct':
        for i,record in enumerate(dataset_records):
            clean_chs = np.empty((record.shape[0],record.shape[1]))
            record_chs = separate_data_chs(record)
            for ch in record_chs:
                clean_chs.append(ibdct(ch,f_low,f_high,fs).reshape(-1,1),axis=1)
            denoised_dataset.append(clean_chs)
    elif denoising_method.lower() == 'butterworth':
        for i,record in enumerate(dataset_records):
            clean_chs = np.empty((record.shape[0],record.shape[1]))
            record_chs = separate_data_chs(record)
            for ch in record_chs:
                clean_chs.append(butterworth_filter(ch,f_low,f_high,fs).reshape(-1,1),axis=1)
            denoised_dataset.append(clean_chs)
    else:
        raise ValueError('Denoising method does not exist. Please enter ibdct or butterworth')

    return denoised_dataset
        


