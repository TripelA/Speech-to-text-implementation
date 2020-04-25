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

        # number of fft points (sample rate * window size = total points)
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
        # (n-1)*(m-1) = len(y); n = 160 = hop_length
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

