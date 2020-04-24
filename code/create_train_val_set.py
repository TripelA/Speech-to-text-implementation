#%%
import pandas as pd
from data_loader_stripped import load_audio
import os
import scipy.io.wavfile as wave
import random

vox_data = pd.read_csv('data/voxforge_train_manifest.csv')

newdir = 'data/transfer_set/'

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
    wav_file = load_audio(wav_name)

    txt_name = vox_data.iloc[i, 1]
    txt_name_stripped = txt_name.replace(txt_str, '')
    f = open(txt_name, 'r')
    txt = f.read()
    f.close()

    if len(lens) == int(len(n)/2):
        print('Training Files Created')

    if len(lens) < int(len(n)/2):

        wave.write(train_dir + 'wav/' + wav_name_stripped, len(wav_file), wav_file)
        f2 = open(train_dir + 'txt/' + txt_name_stripped, 'w')
        f2.write(txt)
        f2.close

    else:
        wave.write(test_dir + 'wav/' + wav_name_stripped, len(wav_file), wav_file)
        f2 = open(test_dir + 'txt/' + txt_name_stripped, 'w')
        f2.write(txt)
        f2.close

    lens.append(len(wav_file))



