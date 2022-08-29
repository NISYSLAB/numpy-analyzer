import os

BASE_DIR = os.getcwd()
Dirs = {
    "BASE" : BASE_DIR,
    "DATASETS" : f"{BASE_DIR}{os.sep}datasets",
}

DatasetInfo = {
    "NAME" : "EGG",
    "ZIP_NAME" : "EGG-database",
    "ZIP_PATH" : f"{BASE_DIR}{os.sep}EGG-database.zip"
}


features_names = ['Mean','Variance','Mode','Median','Skewness','Kurtosis','RMS','Line_Length','Entropy_MVN','Approximate_Entropy','Sample_Entropy',
'Permutation entropy','SVD_Entropy','LZ_Complexity','Hjorth_Mobility','Hjorth_Complexity','DF_CPM',
'DP','P_PSD_G_DP','Spectral_Entropy','BandPower_3-7','BandPower_7-11','BandPower_11-15','Crest_Factor_PSD','Median_Frequency',
'MPF','Petrosian_Fractal_Dimension','Katz_Fractal_Dimension','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5']