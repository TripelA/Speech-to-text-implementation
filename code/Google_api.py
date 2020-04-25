# Import packages
import io
import os
import numpy as np

os.system("sudo -H pip install --upgrade python-Levenshtein")
import Levenshtein
from tqdm import tqdm
from google.cloud import speech_v1  # python wrapper for google cloud API calls

# Assign google cloud API credentials variable to personal credentials file
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = os.getcwd() + '/Machine-Learning-2-bdc5dbdf312d.json'  # add own credentials

# dummy API call to verify authorization
client = speech_v1.SpeechClient()


def sample_recognize(local_file_path):
    """
    Transcribe a short audio file using synchronous speech recognition

    :param
    local_file_path : Path to local audio file (works for flac and wav files)

    :returns
    Most probable transcription for given audio file
    """
    client = speech_v1.SpeechClient()

    language_code = "en-US"  # The language of the supplied audio

    sample_rate_hertz = 16000  # Sample rate in Hertz of the audio data sent

    # encoding = enums.RecognitionConfig.AudioEncoding.FLAC # add encoding type if not using flac or wav files

    config = {"language_code": language_code,
              "sample_rate_hertz": sample_rate_hertz,
              # "encoding" : encoding,
              "audio_channel_count": 1,
              }

    with io.open(local_file_path, "rb") as f:
        content = f.read()

    audio = {"content": content}
    response = client.recognize(config, audio)  # doing the transcription through Google Speech-to-Text API

    # some API responses return no result/transcription
    try:
        for result in response.results:
            # First alternative is the most probable result (highest confidence)
            alternative = result.alternatives[0]
        return alternative.transcript.upper()  # converting to upper for uniformity in evaluation
    except:
        return " "  # returns empty string if the API returns no transcription


path_txt = "./voxforge_sample_files/test/txt/"
path_wav = "./voxforge_sample_files/test/wav/"

texts = []
trans = []
WER = []
CER = []

for each in tqdm(os.listdir(path_txt)):
    # Read the original/gold-standard transcription
    with open(path_txt + each, "r") as file:
        text = file.read()
    texts.append(text)
    
    # Function call to obtain decoded transcription from API
    tran = sample_recognize(path_wav + each[:-4] + ".wav")
    trans.append(tran)

    # Calculate Character Error Rate (CER)
    cer = Levenshtein.distance(tran.replace(' ', ''), text.replace(' ', ''))
    CER = np.append(CER, cer)
    
    # Calculate Word Error Rate (WER)
    uniquewords = set(tran.split() + text.split())  # split output strings by spaces and create set of unique words
    word2char = dict(zip(uniquewords, range(len(uniquewords))))  # create dictionary of unique words

    # map words to the word2char dictionary which is converted to character array (Levenshtein package only accepts strings)
    w1 = [chr(word2char[w]) for w in tran.split()]
    w2 = [chr(word2char[w]) for w in text.split()]

    # calculate distance from word vectors and append to running total
    wer = Levenshtein.distance(''.join(w1), ''.join(w2))
    WER = np.append(WER, wer)

print('Completed Parsing')
print('Mean Levenshtein distance (CER): %f' % np.mean(CER))
print('Mean Levenshtein distance (WER): %f' % np.mean(WER))
