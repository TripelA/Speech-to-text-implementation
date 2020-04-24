#%%
import pandas as pd
import os
from shutil import copy
import random


vox_data = pd.read_csv('data/voxforge_train_manifest.csv')

newdir = os.getcwd() + '/data/voxforge_sample_files/'

train_dir = newdir + 'train/'
test_dir = newdir + 'test/'

# create directories
if not os.path.exists(newdir):
    os.mkdir(newdir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(test_dir + 'wav/'):
    os.mkdir(test_dir + 'wav/')

if not os.path.exists(test_dir + 'txt/'):
    os.mkdir(test_dir + 'txt/')

if not os.path.exists(train_dir + 'wav/'):
    os.mkdir(train_dir + 'wav/')

if not os.path.exists(train_dir + 'txt/'):
    os.mkdir(train_dir + 'txt/')

#%%
random.seed(123)
n = random.sample(range(0, 90000), 1000)

lens = []

wav_str = str(os.getcwd() + '/data/voxforge_dataset/wav/')
txt_str = str(os.getcwd() + '/data/voxforge_dataset/txt/')

for i in n:
    wav_name = vox_data.iloc[i, 0]
    wav_name_stripped = wav_name.replace(wav_str, '')

    txt_name = vox_data.iloc[i, 1]
    txt_name_stripped = txt_name.replace(txt_str, '')

    if len(lens) == int(len(n)/2):
        print('Training Files Created')

    if len(lens) < int(len(n)/2):

        copy(wav_name, train_dir + 'wav/' + wav_name_stripped)
        copy(txt_name, train_dir + 'txt/' + txt_name_stripped)

    else:
        copy(wav_name, test_dir + 'wav/' + wav_name_stripped)
        copy(txt_name, test_dir + 'txt/' + txt_name_stripped)

    lens.append(1)



