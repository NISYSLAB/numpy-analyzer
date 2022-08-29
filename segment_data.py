import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

## get each ch readings separated
def separate_data_chs(record):
    ch1_readings = []
    ch2_readings = []
    ch3_readings = []
    ## loop over the record and get each ch reading
    for i in range(len(record)):
        ch1_readings.append(record[i][0])
        ch2_readings.append(record[i][1])
        ch3_readings.append(record[i][2])
    return ch1_readings,ch2_readings,ch3_readings


def make_views(
    arr,
    win_size,
    step_size,
    writeable=False,
):
    """
  arr: any 2D array whose columns are distinct variables and
    rows are data records at some timestamp t
  win_size: size of data window (given in data points along record/time axis)
  step_size: size of window step (given in data point along record/time axis)
  writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  Note that step_size is related to window overlap (overlap = win_size - step_size), in
  case you think in overlaps.
  This function can work with C-like and F-like arrays, and with DataFrames.  Yay.
  """
    # If DataFrame, use only underlying NumPy array
    if type(arr) == type(pd.DataFrame()):
        arr = arr.values
    if ~np.isfinite(arr).any():
        print("There is non-numeric value in input to make_views()")
        print(np.argwhere(~np.isfinite(arr)))
    # Compute Shape Parameter for as_strided
    n_records = arr.shape[0]
    n_columns = arr.shape[1]
    print(n_records, type(n_records))
    print(win_size, type(win_size))
    print(step_size, type(step_size))
    remainder = (n_records - win_size) % step_size
    num_windows = 1 + int((n_records - win_size - remainder) / step_size)
    shape = (num_windows, win_size, n_columns)
    # Compute Strides Parameter for as_strided
    next_win = step_size * arr.strides[0]
    next_row, next_col = arr.strides
    strides = (next_win, next_row, next_col)
    new_view_structure = as_strided(
        arr,
        shape=shape,
        strides=strides,
        writeable=writeable,
    )
    return new_view_structure


