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
# from .spec_augment import spec_augment

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
    def __init__(self, audio_conf, normalize=False, speed_volume_perturb=False):
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
        # self.spec_augment = spec_augment

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