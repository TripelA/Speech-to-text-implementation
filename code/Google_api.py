from google.cloud import speech_v1
import io
import os
os.system("sudo -H pip install --upgrade python-Levenshtein")
import Levenshtein
from tqdm import tqdm
import numpy as np

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getcwd()+'/Machine-Learning-2-bdc5dbdf312d.json' # add own credentials
client = speech_v1.SpeechClient()

def sample_recognize(local_file_path):
    """
    Transcribe a short audio file using synchronous speech recognition

    :param
    local_file_path : Path to local audio file (works for flac and wav files)

    :returns
    Most probable transcription for given audio file.
    """

    client = speech_v1.SpeechClient()

    # The language of the supplied audio
    language_code = "en-US"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # encoding = enums.RecognitionConfig.AudioEncoding.FLAC # add encoding type if not using flac or wav files

    config = {"language_code": language_code,
              "sample_rate_hertz": sample_rate_hertz,
              # "encoding" : encoding,
              "audio_channel_count": 1,
              }

    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)  # doing the transcription through Google Speech-to-text API

    # some API responses return no result/transcription
    try:
        for result in response.results:
            # First alternative is the most probable result (highest confidence)
            alternative = result.alternatives[0]
        return alternative.transcript.upper() # converting to upper for uniformity in evaluation
    except:
        return " "  # returns empty string if the API returns no transcription

path_txt = os.getcwd()+"/voxforge_sample_files/test/txt"
path_wav = os.getcwd()+"/voxforge_sample_files/test/wav"

texts = []
trans = []
WER = []
CER = []

for each in tqdm(os.listdir(path_txt)):
    with open("./voxforge_sample_files/test/txt/" + each, "r") as file:
        text = file.read()
        texts.append(text)
    tran = sample_recognize("./voxforge_sample_files/test/wav/" + each[:-4] + ".wav")
    trans.append(tran)

    cer = Levenshtein.distance(tran.replace(' ', ''), text.replace(' ', ''))
    CER = np.append(CER, cer)
    uniquewords = set(tran.split() + text.split())  # split output strings by spaces and create set of unique words

    word2char = dict(zip(uniquewords, range(len(uniquewords))))  # create dictionary of unique word

    # map the words to a char array (Levenshtein packages only accepts strings)
    # map words to index in the dictionary word2char
    w1 = [chr(word2char[w]) for w in tran.split()]
    w2 = [chr(word2char[w]) for w in text.split()]

    # calculate distance from word vectors and append to running total
    wer = Levenshtein.distance(''.join(w1), ''.join(w2))
    WER = np.append(WER, wer)

print('Completed Parsing')
print('Mean Levenshtein distance (CER): %f' % np.mean(CER))
print('Mean Levenshtein distance (WER): %f' % np.mean(WER))

