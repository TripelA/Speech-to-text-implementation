##### transcribe_and_compare.py
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
    print('Please change your working directory to the cloned repo code folder to resolve potential filepath '
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


##### create_train_val_set.py
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

##### data_loader_stripped.py
import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import torch
from scipy.io.wavfile import read
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from spec_augment import spec_augment

# define different windows and their associated signals - READ INTO
windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def load_audio(path):

    # read audio file using scipy.io.wavfile read, return sample rate (16000)
    # and sound (len(wav) x 1) - from voxforge test file 1 is len 102,400
    sample_rate, sound = read(path)

    # normalize by dividing by constant - max value possible for given encoding, so setting between [-1, 1]
    sound = sound.astype('float32') / 32767  # normalize audio

    # if sound is multidimensional, try to reshape to an n x 1 array
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average  <- original comment

    # return sound
    return sound

# audio parser class
# not sure what it does since it's not explicitly called,
# but it's used in the spectrogram parser when it's initialized so leaving it in
class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, speed_volume_perturb=False, spec_augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """

        # self initialize
        super(SpectrogramParser, self).__init__()

        # audio_conf is a dictionary of all these values saved with the model, tells us the settings when it was
        # trained. Can load directly by passing model.audio_conf to parser when initializing in transcribe file

        # window stride - get from model
        self.window_stride = audio_conf['window_stride']

        # window size - get from model
        self.window_size = audio_conf['window_size']

        # sample rate - get from model
        self.sample_rate = audio_conf['sample_rate']

        # window - get from model
        self.window = windows.get(audio_conf['window'], windows['hamming'])

        # normalize y/n, default False but in transcribe is True
        self.normalize = normalize

        # speed volume perturb, default is False
        self.speed_volume_perturb = speed_volume_perturb

        # whether or not to augment during training, default is False
        self.spec_augment = spec_augment

        # injecting noise, for robustness when training
        # NOTE: most models did not use noise injection, so will return None as noise_dir = None
        # can strip from files since none of the 3 pre-trained have noise directories specified
        # self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
        #                                     audio_conf['noise_levels']) if audio_conf.get(
        #     'noise_dir') is not None else None

        # probability of noise injection
        self.noise_prob = audio_conf.get('noise_prob')

    # function to parse audio - takes path to wav file as input and returns spectrogram
    def parse_audio(self, audio_path):

        # specific function to load audio with volume perturb, otherwise load audio
        y = load_audio(audio_path)

        # function to randomly inject noise, pulls from [0,1] with probability noise_prob, all None
        # if self.noiseInjector:
        #     add_noise = np.random.binomial(1, self.noise_prob)
        #     if add_noise:
        #         y = self.noiseInjector.inject_noise(y)

        # number of fft points (sample rate * window size = total points) per second
        # eg. voxforgetest[1] = 320
        n_fft = int(self.sample_rate * self.window_size)

        # set window length (320 hz)
        win_length = n_fft

        # size to hop through spectrogram window
        # eg. 160 for voxforgetest[1] and an4
        hop_length = int(self.sample_rate * self.window_stride)

        # STFT = computes discrete fourier transform, see
        # https://librosa.github.io/librosa/generated/librosa.core.stft.html
        # create an nxm sized array from y where n is the n_fft/2 - 1, is such that
        # (n-1)*(m-1) = len(y); n = 161 = hop_length + 1
        # 161 x 641
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)

        # magphase = separate a spectorgram into magnitude and phase, see more at
        # https://librosa.github.io/librosa/generated/librosa.core.magphase.html
        # spect is n x m (161x641)
        # phase is n x m (161x641)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)

        # turn to floattensor
        spect = torch.FloatTensor(spect)

        # normalize if specified, default is False but set to True in transcribe.py
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        # augment if needed, default is False
        #     spect = spec_augment(spect)
        # if self.spec_augment:

        # return spectrogram (which is magnitude component)
        # 161x641
        return spect


# function to load dataset from csv files with directory paths; part of the data loaders in train.py
class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, speed_volume_perturb=False, spec_augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """

        # read IDs and create strings with WAV_PATH, TXT_PATH
        with open(manifest_filepath) as f:
            ids = f.readlines()

        # split into list
        ids = [x.strip().split(',') for x in ids]

        # initialize properites
        self.ids = ids
        self.size = len(ids)

        # create the dictionary mapping characters to numerical values
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

        super(SpectrogramDataset, self).__init__(audio_conf, normalize, speed_volume_perturb, spec_augment)

    # function to get item when indexing class object
    def __getitem__(self, index):

        # subset to passed index
        sample = self.ids[index]

        # extract audio path and txt path from the manifest
        audio_path, transcript_path = sample[0], sample[1]

        # get the spectrogram of the WAV file at the filepath using the parse_audio method
        spect = self.parse_audio(audio_path)

        # get the text transcription of the txt file at the filepath
        transcript = self.parse_transcript(transcript_path)

        # return the spectrogram and the transcript
        return spect, transcript

    # function to get the contents of the transcript
    def parse_transcript(self, transcript_path):

        # read the txt contents at the given filepath
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')

        # convert the text to the numerical items in the dictionary
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))

        # return the numerical version of the transcript
        return transcript

    # create a length method
    def __len__(self):
        return self.size


# Semi-unclear exactly how this function works - looks like it's for the data loader to use to iterate through data
# and return normalized spectrogram values, targets, input_percentages, and target_sizes

# returns tuple with four objects:
# 1. input of [batchsize x 1 x 161 x len]
# 2. targets [len]
# 3. input_percentages [batchsize]
# 4. target_sizes [batchsize]

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    # initially sorts by transcription size
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)

    # gets the longest transcription
    longest_sample = max(batch, key=func)[0]

    # gets the number of frequency pieces (161)
    freq_size = longest_sample.size(0)

    # number of batches (len(data)/batchsize)
    minibatch_size = len(batch)

    # gets the number of 20ms bits
    max_seqlength = longest_sample.size(1)

    # create zero tensor
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)

    # create percentages [minibatch- 20]
    input_percentages = torch.FloatTensor(minibatch_size)

    # create target tensor of size [minibatch - 20]
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []

    # for each batch
    for x in range(minibatch_size):

        # get the samples
        sample = batch[x]

        # get the input
        tensor = sample[0]

        # get the targets
        target = sample[1]

        # get the transcription length(s)
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


# data loader class for the audio data
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


# data sampler to create batches (inheriting from pytorch Sampler)
class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)

        # set the data source as the passed data source (either train_dataset or test_dataset)
        self.data_source = data_source

        # create ids
        ids = list(range(0, len(data_source)))

        # create bins of size batch_size to use during sampling
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    # iter property to loop over values, returning IDs
    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    # length property
    def __len__(self):
        return len(self.bins)

    # function to shuffle bins
    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


# sampler for distributed data (NOT USED FOR OUR TRAINING BUT ADDED IN CASE)
class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]


def fft_plot(audio, rate):
    import matplotlib.pyplot as plt
    n = len(audio)
    T = 1/rate
    yf = scipy.fft(audio)
    xf = np.linspace(0, 1.0/(2.0*T), int(n/2))
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform')
    return plt.show()

##### train_stripped

import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
from warpctc_pytorch import CTCLoss
from logger import VisdomLogger, TensorBoardLogger


from data_loader_stripped import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
# from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler

from decoder_stripped import GreedyDecoder
from model import DeepSpeech, supported_rnns
from test import evaluate
from utils import reduce_tensor, check_loss, remove_parallel_wrapper

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.cpu().numpy()


class DefaultArguments():
    """ Use this class to change values of model, data and training parameters"""

    def __init__(self):
        self.id = 'Deepspeech training'

        # TODO check paths
        self.train_manifest = 'data/voxforge_train_manifest_v2.csv'
        self.val_manifest = 'data/voxforge_test_manifest.csv'

        # sampling_rate = 16k says that this audio was recorded(sampled) with a sampling frequency of 16k. In other
        # words, while recording this file we were capturing 16000 amplitudes every second.
        self.sample_rate = 16000

        self.batch_size = 20
        self.num_workers = 0
        self.labels_path = 'labels.json'
        self.window_size = .02  # 'Window size for spectrogram in seconds'
        self.window_stride = .01  # 'Window stride for spectrogram in seconds'
        self.window = 'hamming'  # 'Window type for spectrogram generation'
        self.hidden_size = 1000  # 'Hidden size of RNNs'
        self.hidden_layers = 5  # 'Number of RNN layers'
        self.rnn_type = 'lstm'  # 'Type of the RNN. rnn|gru|lstm are supported'
        self.epochs = 30  # Number of training epochs
        self.cuda = 'cuda'  # Use cuda to train model'
        self.lr = 3e-4  # 'initial learning rate'
        self.momentum = 0.9  # 'momentum'
        self.max_norm = 400  # 'Norm cutoff to prevent explosion of gradients'
        self.learning_anneal = 1.1  # 'Annealing applied to learning rate every epoch'
        self.silent = False  # 'Turn off progress tracking per iteration'
        self.checkpoint = False  # 'Enables checkpoint saving of model'
        self.checkpoint_per_batch = 0  # Save checkpoint per batch. 0 means never save'
        self.visdom = False  # Turn on visdom graphing'
        self.tensorboard = False  # 'Turn on tensorboard graphing'
        self.log_dir = 'visualize/deepspeech_final'  # 'Location of tensorboard log'
        self.log_params = False  # 'Log parameter values and gradients'
        self.id = 'Deepspeech training'  # 'Identifier for visdom/tensorboard run'
        self.save_folder = 'models/'  # 'Location to save epoch models'
        self.model_path = 'models/iteration5.pth'  # 'Location to save best validation model'
        # TODO check path
        self.continue_from = 'librispeech_pretrained_v2.pth'  # continue from checkpoint model
        self.finetune = True  # 'Finetune the model from checkpoint "continue_from"'
        self.speed_volume_perturb = False  # 'Use random tempo and gain perturbations.'
        self.spec_augment = False
        self.noise_dir = None  # 'Directory to inject noise into audio. If default, noise Inject not added'
        self.noise_prob = 0.4  # 'Probability of noise being added per sample'
        self.noise_min = 0.0  # 'Minimum noise level to sample from. (1.0 means all noise, not original signal)'
        self.noise_max = 0.5  # 'Maximum noise levels to sample from. Maximum 1.0'
        self.no_shuffle = False  # 'Turn off shuffling and sample from dataset based on sequence length (smallest to
        # largest)'
        self.no_sortaGrad = False  # 'Turn off ordering of dataset on sequence length for the first epoch.'
        self.bidirectional = True  # 'Turn off bi-directional RNNs, introduces lookahead convolution'
        self.dist_url = 'tcp://127.0.0.1:1550'  # 'url used to set up distributed training'
        self.dist_backend = 'nccl'  # distributed backend
        self.world_size = 1  # 'number of distributed processes'
        self.rank = 0  # 'The rank of this process'
        self.gpu_rank = None  # 'If using distributed parallel for multi-gpu, sets the GPU for the process'
        self.seed = 123456  # Seed to generators
        self.opt_level = 'O1'
        self.keep_batchnorm_fp32 = None
        self.loss_scale = 1  # Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of
        # gradients'
        self.distributed = False
        self.no_sorta_grad = False


class AverageMeter(object):
    """Computes and stores the average and current value, used for evaluation and epoch time computation and ctc loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':

    # load the default arguments
    args = DefaultArguments()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set device to cuda
    device = torch.device("cuda" if args.cuda else "cpu")
    os.system("export CUDA_VISIBLE_DEVICES=1")

    # if the number of distributed process is 1, set the value to True
    args.distributed = args.world_size > 1

    main_proc = True
    # device = torch.device("cuda" if args.cuda else "cpu")

    # if we want to use distributed programming
    # if args.distributed:
    #     if args.gpu_rank:
    #         torch.cuda.set_device(int(args.gpu_rank))
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)
    #     main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    # set up the variables for the number of epochs
    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None

    # visualization tool for check progress of model training
    # if main_proc and args.visdom:
    #     visdom_logger = VisdomLogger(args.id, args.epochs)
    # if main_proc and args.tensorboard:
    #     tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    avg_loss, start_epoch, start_iter, optim_state, amp_state = 0, 0, 0, None, None

    # start from the pretrained models
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)

        # Load all tensors onto the CPU, using a function ( refer to torch.serialization doc to know more)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        # load pretrained model
        model = DeepSpeech.load_model_package(package)

        # set labels A-Z, -, ' ', total 29
        labels = model.labels

        audio_conf = model.audio_conf

        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            amp_state = package['amp']
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1

            # get what was the last avg loss
            avg_loss = int(package.get('avg_loss', 0))

            # get evaluation metrics for ctc loss, wer, cer
            loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], \
                                                     package['wer_results']
            best_wer = wer_results[start_epoch]
            # if main_proc and args.visdom:  # Add previous scores to visdom graph
            #     visdom_logger.load_previous_values(start_epoch, package)
            # if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
            #     tensorboard_logger.load_previous_values(start_epoch, package)
    # train new model
    else:
        # read labels
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        # create audio configuration dictionary
        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        # rnn type either GRU or LSTM
        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

        # create network architecture
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)

    # choose the algorithm to decode the model output
    decoder = GreedyDecoder(labels)

    # read the train dataset
    # representation of frequencies of a given signal with time is called a spectrogram
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.speed_volume_perturb,
                                       spec_augment=args.spec_augment)

    # read the test dataset
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    # sample the train sampler depending on the batchsize
    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        # if we are using distributed programing on multiple GPUs
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)

    # data generator for train and test
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    # shuffle the batches after every epoch to improve the performance
    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    parameters = model.parameters()

    # Declare model and optimizer as usual, with default (FP32) precision
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    # amp is automatic mixed precision
    # Allow Amp to perform casts as required by the opt_level
    # Amp allows users to easily experiment with different pure and mixed precision modes.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    # load optimizer state
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    # load precision state
    if amp_state is not None:
        amp.load_state_dict(amp_state)

    if args.distributed:
        model = DistributedDataParallel(model)
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # create class objects
    criterion = CTCLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # start the training epoch
    for epoch in range(start_epoch, args.epochs):

        model.train()
        end = time.time()
        start_epoch_time = time.time()

        # load data using generator audio data loader
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break

            # input_percentages = sample seq len/ max seq len in the batch
            # target sizes = len of target in every seq
            inputs, targets, input_percentages, target_sizes = data

            # every input size input % * max seq length size(3)
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)

            # model outputs batch * max seq length (T) * 29(labels)
            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss

            # calculate ctc loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            # if distributed gather ctc loss
            if args.distributed:
                loss = loss.to(device)
                loss_value = reduce_tensor(loss, args.world_size).item()
            else:
                loss_value = loss.item()

            # Check to ensure valid loss was calculated, there is no inf or nan
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()
                # compute gradient

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            # add epoch loss
            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print the output on the console
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
            # if you want to save output after every batch, default set to 0
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(remove_parallel_wrapper(model),
                                                optimizer=optimizer,
                                                amp=amp,
                                                epoch=epoch,
                                                iteration=i,
                                                loss_results=loss_results,
                                                wer_results=wer_results,
                                                cer_results=cer_results,
                                                avg_loss=avg_loss),
                           file_path)
            del loss, out, float_out

        # average loss across all batches
        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        # evalulate results on test dataset
        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=decoder,
                                             target_decoder=decoder)
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        values = {
            'loss_results': loss_results,
            'cer_results': cer_results,
            'wer_results': wer_results
        }
        # if args.visdom and main_proc:
        #     visdom_logger.update(epoch, values)
        # if args.tensorboard and main_proc:
        #     tensorboard_logger.update(epoch, values, model.named_parameters())
        #     values = {
        #         'Avg Train Loss': avg_loss,
        #         'Avg WER': wer,
        #         'Avg CER': cer
        #     }

        # if you have to save file after every epoch
        if main_proc and args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(remove_parallel_wrapper(model),
                                            optimizer=optimizer,
                                            amp=amp,
                                            epoch=epoch,
                                            loss_results=loss_results,
                                            wer_results=wer_results,
                                            cer_results=cer_results),
                       file_path)
        # anneal lr Learning rate annealing is reducing the rate after every epoch in order to not miss the local
        # minimum and avoid oscillation
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        # if the best model is found than previous iteration, overwrite the model with better model
        if main_proc and (best_wer is None or best_wer > wer):
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(remove_parallel_wrapper(model),
                                            optimizer=optimizer,
                                            amp=amp, epoch=epoch,
                                            loss_results=loss_results,
                                            wer_results=wer_results,
                                            cer_results=cer_results)
                       , args.model_path)
            best_wer = wer
            avg_loss = 0

        # if you want to shuffle argument after every epoch
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

##### decoder_stripped.py

import Levenshtein as Lev
import torch
from six.moves import xrange


class Decoder(object):

    # initialize decoder
    def __init__(self, labels, blank_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"

        # initialize labels (basically alphabet)
        self.labels = labels

        # stores dictionary of labels and place in list of labels (ie. 0:'_' since it's the first label), passed from
        # the model with model.labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

        # where the underscore is located
        self.blank_index = blank_index

        #
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    # can use the decoder to calculate wer and cer, or just use lev distance on the outputs
    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError

# LEFT IN TO NOT POTENTIALLY BREAK OTHER CODE, BUT NOT USED DURING TRAINING
class BeamCTCDecoder(Decoder):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None

        # xrange used to save memory
        for x in xrange(len(sequences)):

            # specify sequence length
            seq_len = sizes[x] if sizes is not None else len(sequences[x])

            # process string
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)

            # append string to overall strings
            strings.append([string])  # We only return one path

            # if we want the offsets, append the offsets
            if return_offsets:
                offsets.append([string_offsets])

        # return values
        if return_offsets:
            return strings, offsets
        else:
            return strings

    # function to process string from predicted character indices
    def process_string(self, sequence, size, remove_repetitions=False):

        # initialize
        string = ''
        offsets = []

        # loop through each piece of the output within window
        for i in range(size):

            # turn from integer to character based on dictionary
            char = self.int_to_char[sequence[i].item()]

            # if the character is not the blank index:
            if char != self.int_to_char[self.blank_index]:

                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass

                # if the character is a space, add a space to the string and set this as an offset location
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)

                # append string and offsets
                else:
                    string = string + char
                    offsets.append(i)

        # return values
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """

        # get the argmax of dim 2 (3rd dimension), which is the probability of each character from the dictionary of
        # characters defined in the model

        # returns both the probability (saved as _) and the index (max_probs). Don't care about the probability
        _, max_probs = torch.max(probs, 2)

        # convert the list of predicted characters to strings -- see convert_to_strings and process_string above
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets

##### test_stripped

from tqdm import tqdm
import torch


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False):
    # set model to eval functionality
    model.eval()

    # initialize values at zero
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0

    # create empty list for storing output
    output_data = []

    # loop through the test_loader data loader
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):

        # unpack the data from the data loader
        inputs, targets, input_percentages, target_sizes = data

        # set input sizes and add to device
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)

        # if the data uses half precision, set the half method
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0

        # for the sizes
        for size in target_sizes:

            # append the offset values and increase the size for the loop
            split_targets.append(targets[offset:offset + size])
            offset += size

        # retrieve the model output
        out, output_sizes = model(inputs, input_sizes)

        # decode the output using the passed decoder and convert output to string format
        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)


        # if there is a location set for saving output
        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu(), output_sizes, target_strings))

        # loop through the length of the target strings
        for x in range(len(target_strings)):

            # get the decoded output and target strings and calculate the WER and CER
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst

            # calculate the number of words and characters
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))

            # if verbose (ie. print everything), show all output and predicted output
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")

    # divide WER and CER by total length of strings in number of words/characters
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, output_data

##### transcribe_stripped.py

import warnings
import torch
warnings.simplefilter('ignore')


def transcribe(audio_path, spect_parser, model, decoder, device):

    # convert the file in the audio path to a spectrogram - see data_loader/SpectrogramParser for more info
    spect = spect_parser.parse_audio(audio_path).contiguous()

    # nest the spectrogram within two arrays - why?? Look in model
    # think it might have to do with the first layer being a conv2d, so it needs channel values
    # but why 4d?
    # produces a 1x1x161xn
    # 1: 1 wav file/spectrogram
    # 1: 1 x value
    # n: number of windows from spectrogram (seemingly 161 for all files)
    # m: number of frequency bands (641 for voxforgesample/test[1]
    spect = spect.view(1, 1, spect.size(0), spect.size(1))

    # move the spectrogram to the device
    spect = spect.to(device)

    # empty tensor with number of inputs
    input_sizes = torch.IntTensor([spect.size(3)]).int()

    # model the spectrogram and produce the output and output sizes
    # out: 1 x len(data)/win_length x len(labels) of probabilities of each class for each piece of the spectrogram
    # output_sizes: number of pieces of the spectrogram
    out, output_sizes = model(spect, input_sizes)

    # decode the output sizes
    # decoded_output: estimated transcription
    # decoded_offsets: time step for each piece of the transcription (in the original wav file)
    # ie. before reducing will have x number of 'S' character estimations for each component of the wav file,
    # this tells you which position in the original probability matrix each character initially ends (so the first 36
    # are 'S', then 7 more ' ', which means ' ' is decoded_offset 43
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

    return decoded_output, decoded_offsets

##### utils_stripped.py

import torch
import torch.distributed as dist
from model import DeepSpeech

# function to consolidate tensor on a single processor


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


# function to check CTC loss is valid
def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """

    # initialize values
    loss_valid = True
    error = ''

    # If the loss grows to infinity, return error message
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"

    # if there are NaN losses, return error message
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'

    # if loss is negative, return error message
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"

    # return T/F of loss validity and potential error message
    return loss_valid, error


# function to load model from pth file
def load_model(device, model_path, use_half):

    # use load_model method from DeepSpeech class
    model = DeepSpeech.load_model(model_path)

    # set model to eval
    model.eval()

    # put model on device (GPU/CPU)
    model = model.to(device)

    # if the model is using half-precision sampling, use the half method of the model to indicate so
    if use_half:
        model = model.half()

    # return the model
    return model


# function to un-parallelize the model
def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    # Take care of distributed/data-parallel wrapper
    model_no_wrapper = model.module if hasattr(model, "module") else model
    return model_no_wrapper


