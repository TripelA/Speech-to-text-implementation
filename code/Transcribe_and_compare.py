#%%
import argparse
import warnings
import sys
from tqdm import tqdm

from opts import add_decoder_args, add_inference_args
from utils import load_model
from transcribe import transcribe

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
import os.path
import json
import numpy as np
import random
import torch
import Levenshtein

if os.getcwd()[-18:] != 'deepspeech.pytorch':
    print('Please change your working directory to the cloned repo located at \n'
          'https://github.com/SeanNaren/deepspeech.pytorch.git \nto resolve potential filepath '
          'issues, then continue working')
else:
    print('Loaded into the correct working directory')

#%% Transcribe and compare function


def transcribe_and_compare(wav_dir, txt_dir, model_dir, n_files='All', verbose=False):

    # set random seed for sampling wav files
    random.seed(1)

    try:

        # set device as cuda if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameter needed for model loading
        half = False

        # load list of wav files that we will transcribe
        wav_dir = os.getcwd() + wav_dir
        wav_files = [wav_dir + f for f in os.listdir(wav_dir) if f[-4:] == '.wav']
        print('wav files found 1/3')

        # since the an4 model is built on the smallest dataset (130 test cases), downsample if needed to ensure
        # equivalent sizes
        if 'an4' not in wav_dir:
            wav_files = random.sample(wav_files, 130)  # 130 is the length of the an4 testing set

        # load list of txt files containing the actual transcriptions
        txt_dir = os.getcwd() + txt_dir
        txt_files = [txt_dir + f[len(wav_dir):][:-4] + '.txt' for f in wav_files]
        print('txt files found 2/3')

        # load the model that will be used to transcribe
        model_path = os.getcwd() + model_dir
        model = load_model(device, model_path, half)
        print('model found 3/3')

    # print error if loading fails
    except:
        print('Model and source not found, returning NaN')
        return np.nan, np.nan

    try:
        # specify decoder
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

        # specify spectrogram parser - turn wav files into spectrograms
        spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

        # set n_files to the max possible, unless the user specifies a number of files to use
        if n_files == 'All':
            n_files = len(wav_files)

        # initialize empty lists to store information
        all_distance = []
        transcribed = []
        actual = []

        print('Paths specified and model loaded. Transcribing')

    # raise error if parser and decoder have an issue loading
    except:
        print('Parser and decoder issue')
        return np.nan, np.nan

    try:
        # loop through each file in list of files
        for i in tqdm(range(n_files)):
            decoded_output, decoded_offsets = transcribe(audio_path=wav_files[i],        # name of wav file
                                                         spect_parser=spect_parser,      # spectrogram parser
                                                         model=model,                    # model
                                                         decoder=decoder,                # greedy or beam decoder
                                                         device=device,                  # cuda or cpu
                                                         use_half=half)                  # half precision or not

            # open associated txt file and get contents
            f = open(txt_files[i], 'r')
            act = f.read()
            f.close()

            # get the contents of the decoded output
            decode = decoded_output[0][0]

            # METRICS TO UPDATE! Currently, calculate Levenshtein distance between transcription and predicted
            ld = Levenshtein.distance(decode, act)
            all_distance = np.append(all_distance, ld)

            # option for user to print as the data as it is transcribed
            if verbose:
                print('Predicted: %s' % decoded_output[0][0])
                print('Actual: %s' % act)
                print('Levenshtein Distance: %i' % ld, end='\n\n')

            # append pred and actual to respective lists
            transcribed = np.append(transcribed, decoded_output[0][0])
            actual = np.append(actual, act)

        print('Completed Parsing')
        print('Mean Leveshtein distance: %f' % np.mean(all_distance))

        # return lists of predicted transcriptions and actual transcriptions
        return transcribed, actual

    except:
        print('Transcription Prediction failed')
        return np.nan, np.nan



#%% an4 model with an4 test data
transcribed, actual = transcribe_and_compare("/data/an4_dataset/test/an4/wav/",
                                             "/data/an4_dataset/test/an4/txt/",
                                             "/models/an4_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% LibriSpeech Model with LibriSpeech test data
transcribed, actual = transcribe_and_compare("/data/LibriSpeech_dataset/test_clean/wav/",
                                             "/data/LibriSpeech_dataset/test_clean/txt/",
                                             "/models/librispeech_pretrained_v2.pth")

try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% Tedlium model with tedlium test data
transcribed, actual = transcribe_and_compare("/data/TEDLIUM_dataset/TEDLIUM_release2/test/converted/wav/",
                                             "/data/TEDLIUM_dataset/TEDLIUM_release2/test/converted/txt/",
                                             "/models/ted_pretrained_v2.pth")

try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% an4 model with voxforge data
transcribed, actual = transcribe_and_compare("/data/voxforge_dataset/wav/",
                                             "/data/voxforge_dataset/txt/",
                                             "/models/an4_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% librispeech model with voxforge data
transcribed, actual = transcribe_and_compare("/data/voxforge_dataset/wav/",
                                             "/data/voxforge_dataset/txt/",
                                             "/models/librispeech_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% librispeech model with voxforge data
transcribed, actual = transcribe_and_compare("/data/voxforge_dataset/wav/",
                                             "/data/voxforge_dataset/txt/",
                                             "/models/ted_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')
except:
    print('Failed')
