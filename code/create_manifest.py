import os
import pandas as pd
import numpy as np

train_path = '/data/voxforge_sample_files/train/'
test_path = '/data/voxforge_sample_files/test/'

train_wav_files = os.listdir(os.getcwd() + train_path + 'wav/')
train_txt_files = os.listdir(os.getcwd() + train_path + 'txt/')

test_wav_files = os.listdir(os.getcwd() + test_path + 'wav/')
test_txt_files = os.listdir(os.getcwd() + test_path + 'txt/')

for i in range(len(train_wav_files)):
    train_wav_files[i] = os.getcwd() + train_path + 'wav/' + train_wav_files[i]

for i in range(len(train_txt_files)):
    train_txt_files[i] = os.getcwd() + train_path + 'wav/' + train_txt_files[i]

for i in range(len(test_wav_files)):
    test_wav_files[i] = os.getcwd() + test_path + 'wav/' + test_wav_files[i]

for i in range(len(train_txt_files)):
    test_txt_files[i] = os.getcwd() + test_path + 'wav/' + test_txt_files[i]

train_manifest = pd.DataFrame(np.column_stack((train_wav_files, train_txt_files)))
train_manifest.to_csv('train_manifest.csv')

test_manifest = pd.DataFrame(np.column_stack((test_wav_files, test_txt_files)))
test_manifest.to_csv('test_manifest.csv')