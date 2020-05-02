#%%
from tqdm import tqdm
import os.path
import json
import numpy as np
import random
import Levenshtein
import torch

from transcribe_stripped import transcribe
from data_loader_stripped import SpectrogramParser
from utils_stripped import load_model
from decoder_stripped import GreedyDecoder


# temporary
if os.getcwd()[-4:] != 'code':
    print('Please change your working directory to "code" folder of this repository to resolve potential filepath '
          'issues, then continue working')
else:
    print('Loaded into the correct working directory')

#%% Transcribe and compare function


def transcribe_and_compare(wav_dir, txt_dir, model_dir, n_files=500, verbose=False):

    # set random seed for sampling wav files
    random.seed(1)

    try:

        # set device as cuda if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load list of wav files that we will transcribe
        wav_dir = os.getcwd() + wav_dir
        wav_files = [wav_dir + f for f in os.listdir(wav_dir) if f[-4:] == '.wav']
        print('wav files found 1/3')

        # since the an4 model is built on the smallest dataset (130 test cases), downsample if needed to ensure
        # equivalent sizes
        #     wav_files = random.sample(wav_files, 130)  # 130 is the length of the an4 testing set

        # load list of txt files containing the actual transcriptions
        txt_dir = os.getcwd() + txt_dir
        txt_files = [txt_dir + f[len(wav_dir):][:-4] + '.txt' for f in wav_files]
        print('txt files found 2/3')

        # load the model that will be used to transcribe - look into why half
        model_path = os.getcwd() + model_dir
        model = load_model(device, model_path, use_half=False)
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
        cer = []
        wer = []
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
                                                         device=device)                  # cuda or cpu

            # open associated txt file and get contents
            f = open(txt_files[i], 'r')
            act = f.read()
            f.close()

            # get the contents of the decoded output
            decode = decoded_output[0][0]

            # METRICS TO UPDATE! Currently, calculate Levenshtein distance between transcription and predicted
            ### CER ###

            # replace any spaces
            decode_lev, act_lev = decode.replace(' ', ''), act.replace(' ', '')

            # calculate distance without spaces
            ld_cer = Levenshtein.distance(decode_lev, act_lev)

            # append CER to running list
            cer = np.append(cer, ld_cer)

            ### WER ###

            # split output strings by spaces and create set of unique words
            uniquewords = set(decode.split() + act.split())

            # create dictionary of each word and a corresponding index
            word2char = dict(zip(uniquewords, range(len(uniquewords))))

            # map the words to a char array (Levenshtein packages only accepts
            # strings)

            # map words to index in the dictionary word2char
            w1 = [chr(word2char[w]) for w in decode.split()]
            w2 = [chr(word2char[w]) for w in act.split()]

            # calculate distance from word vectors and append to running total
            ld_wer = Levenshtein.distance(''.join(w1), ''.join(w2))
            wer = np.append(wer, ld_wer)

            # option for user to print as the data as it is transcribed
            if verbose:
                print('Predicted: %s' % decoded_output[0][0])
                print('Actual: %s' % act)
                print('Levenshtein Distance (CER): %i' % ld_cer, end='\n\n')

            # append pred and actual to respective lists
            transcribed = np.append(transcribed, decoded_output[0][0])
            actual = np.append(actual, act)

        print('Completed Parsing')
        print('Mean Levenshtein distance (CER): %f' % np.mean(cer))
        print('Mean Levenshtein distance (WER): %f' % np.mean(wer))

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

#%% an4 model with voxforge test data
transcribed, actual = transcribe_and_compare("/data/voxforge_sample_files/test/wav/",
                                             "/data/voxforge_sample_files/test/txt/",
                                             "/models/an4_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% librispeech model with test voxforge data
transcribed, actual = transcribe_and_compare("/data/voxforge_sample_files/test/wav/",
                                             "/data/voxforge_sample_files/test/txt/",
                                             "/models/librispeech_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')

except:
    print('Failed')

#%% tedlium model with test voxforge data
transcribed, actual = transcribe_and_compare("/data/voxforge_sample_files/test/wav/",
                                             "/data/voxforge_sample_files/test/txt/",
                                             "/models/ted_pretrained_v2.pth")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')
except:
    print('Failed')

#%% Transfer Learning model with test voxforge data
transcribed, actual = transcribe_and_compare("/data/voxforge_sample_files/test/wav/",
                                             "/data/voxforge_sample_files/test/txt/",
                                             "ENTER PATH TO NEW MODEL HERE")
try:
    for i in range(5):
        print('Transcribed: %s' % transcribed[i])
        print('Actual: %s' % actual[i], end='\n\n')
except:
    print('Failed')
