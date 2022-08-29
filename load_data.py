import os
import numpy as np
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import numba
import glob
import re
import math
import pandas as pd
import pywt

from constants import Dirs, DatasetInfo


def build_dataset_directory(dataset_info):
    '''
    builds directory for a dataset and extracts all records inside it
    using the parameters from the dataset information dictionary.

    Return
    ------
    dataset_dir: the path to the folder where the dataset is extracted
    '''
    name, zip_path = dataset_info['NAME'], dataset_info["ZIP_PATH"]
    dataset_dir = f"{Dirs['DATASETS']}{os.sep}{name}"
    if(os.path.exists(dataset_dir)):
        shutil.rmtree(dataset_dir)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    return dataset_dir
    

def load_dataset_metadata(dataset_dir,i,label):
    '''
    gets a record name, path, and label.

    Return
    ------
    record_metadata: an array containing the record's name, path, and label.
    '''
    record_name = f'ID{i}_{label}'
    record_path = f'{dataset_dir}{os.sep}{DatasetInfo["ZIP_NAME"]}{os.sep}ID{i}_{label}.txt'
    record_metadata = []
    record_metadata.append([record_name,record_path,label])
    return record_metadata


def build_dataframe(dataset_dir):
    '''
    Builds a dataframe containing the records' name, path, and label.

    Return
    ------
    dataset_df: dataframe containing the records' name, path, and label.
    '''
    dataset_directory = dataset_dir
    records = []
    for i in range(1,21):
        records.extend(load_dataset_metadata(dataset_directory,i,'fasting'))
        records.extend(load_dataset_metadata(dataset_directory,i,'postprandial'))
    dataset_df = pd.DataFrame(records,columns=["name", "path", "label"])
    return  dataset_df


def load_record_array(record,label):
    '''
    convert a record from text file to an array

    Return
    ------
    record_array: array containing the record's channel data. 
    each channel is represented by a column.
    '''
    with open(record) as f:
        lines = f.readlines()
    record_array = []
    for index in range(len(lines)) :
        lines[index] = lines[index].split()
        for num in range(len(lines[index])):
            lines[index][num] = float(lines[index][num])
    record_array.append([lines,label])
    return record_array 


def load_records(df):
    ''''
    loads all records from their file path and saves them into an array.

    Return
    ------
    dataset: multi-dimensional array containing all the records' arrays. 
    '''
    dataset = []
    for i, j in df.iterrows():
        record  = load_record_array(j[1],j[2])
        dataset.extend(record)
    return dataset

