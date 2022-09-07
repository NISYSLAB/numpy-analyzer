import os
import numpy as np
import zipfile
import shutil
from pathlib import Path



def build_dataset_dir(dataset_info,target_dir):
    '''
    builds directory for a dataset and extracts all records inside it
    using the parameters from the dataset information dictionary.

    Parameters

    Return
    ------
    dataset_dir: the path to the folder where the dataset is extracted
    '''
    name, zip_path = dataset_info['NAME'], dataset_info["ZIP_PATH"]
    dataset_dir = f"{target_dir}{os.sep}{name}"
    if(os.path.exists(dataset_dir)):
        shutil.rmtree(dataset_dir)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    return dataset_dir
    

def structure_record(record,ch_structure='v',target_ch_structure='v'):
    '''
    re-structures the record channels according to the target channel structure

    Parameters
    ----------
    record: numpy array of record to be restructured.
    ch_structure: whether channels are in rows or columns. default is columns
    target_ch_structure: whether the output file channels should be in rows or columns. default is columns

    Return
    ------
    restructured record.
    '''

    if ch_structure.lower() != 'h' and ch_structure.lower() != 'v':
        raise ValueError("This channel structure doesn't exist. Please enter v for vertical or h for horizontal.")
    if target_ch_structure.lower() != 'h' and target_ch_structure.lower() != 'v':
        raise ValueError("This target channel structure doesn't exist. Please enter v for vertical or h for horizontal.")
    if ch_structure.lower() == 'h':
        
        if target_ch_structure.lower() == 'v':
            return record.T
        
        elif target_ch_structure.lower() == 'h':
            return record
    elif ch_structure.lower() == 'v':

        if target_ch_structure.lower() == 'v':
            return record
        
        elif target_ch_structure.lower() == 'h':
            return record.T


def load_record_txt(record_file,ch_structure='v',target_ch_structure='v'):
    '''
    read and convert a record from text file to a numpy array with the target channel format.
    Parameters
    ----------
    record: text file
    ch_structure: whether channels are in rows or columns. default is columns
    target_ch_structure: whether channels are in rows or columns. default is columns

    Return
    ------
    numpy array of the input file with the target channel format.
    '''
    with open(record_file) as f:
        lines = f.readlines()
    
    for index in range(len(lines)) :
        lines[index] = lines[index].split()
        for num in range(len(lines[index])):
            lines[index][num] = float(lines[index][num])
    record = np.array(lines)
    
    return structure_record(record,ch_structure=ch_structure,target_ch_structure=target_ch_structure)
            


def load_record_csv(record,delimiter=',',dtype=None,ch_structure='v',target_ch_structure='v'):
    '''
    convert a record from csv file to a numpy array with the target channel format.

    Parameters
    ----------
    record: csv file
    delimiter: delimiter used in file to separate entries.
    ch_structure: whether channels are in rows or columns. default is columns
    target_ch_structure: whether channels are in rows or columns. default is columns

    Return
    ------
    numpy array of the input file with the target channel format.
    '''
    if dtype:
        record = np.genfromtxt(record, delimiter=delimiter,dtype=dtype)
    else:
        record = np.genfromtxt(record, delimiter=delimiter)
    return structure_record(record,ch_structure=ch_structure,target_ch_structure=target_ch_structure)

def load_record_np(record,ch_structure='v',target_ch_structure='v'):
    '''
    reads numpy file and returns the record array in the target restructured format.

    Parameters
    ----------
    record: numpy file
    ch_structure: whether channels are in rows or columns. default is columns
    target_ch_structure: whether channels are in rows or columns. default is columns

    Return
    ------
    numpy array of the input file with the target channel format.
    '''
    record = np.load(record)
    return structure_record(record,ch_structure=ch_structure,target_ch_structure=target_ch_structure)


def load_record(file_path,delimiter=',',dtype=None,ch_structure='v',target_ch_structure='v'):
    '''
    takes record's path and reads the record depending on the record extension.

    Parameters
    ----------
    file_path: string containing the file name and extension.
    ch_structure: whether channels are in rows or columns. default is columns
    target_ch_structure: whether the output file channels should be in rows or columns. default is columns

    Return
    ------
    record: numpy array of the file contents.
    '''
    file_name,file_extension = os.path.splitext(file_path)
    if file_extension == '.txt':
        record = load_record_txt(file_path,ch_structure=ch_structure,target_ch_structure=target_ch_structure)
    elif file_extension == '.csv':
        record = load_record_csv(file_path,delimiter,dtype,ch_structure=ch_structure,target_ch_structure=target_ch_structure)
    elif file_extension == '.npy':
        record = load_record_np(file_path,ch_structure=ch_structure,target_ch_structure=target_ch_structure)
    return record



def load_dir(dir_path,ch_structure='v',target_ch_structure='v',delimiter=',',dtype=None):
    ''''
    loads all records from their file path and saves them into an array.

    Parameters
    ----------
    dir_path: path to directory containing the files.
    ch_structure: whether channels are in rows or columns. default is columns
    target_ch_structure: whether the output file channels should be in rows or columns. default is columns

    Return
    ------
    dataset: array containing all the records' arrays. 
    '''
    dataset = np.array(len(os.listdir(dir_path)))
    for path in os.listdir(dir_path):
        print(path)
        if os.path.isfile(os.path.join(dir_path, path)):
            print(os.path.join(dir_path, path))
            np.append(dataset,load_record(os.path.join(dir_path, path),ch_structure,target_ch_structure,dtype),axis=0)

    return dataset




