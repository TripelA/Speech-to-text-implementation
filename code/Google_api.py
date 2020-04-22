from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/Machine-Learning-2-b1acfc61fd57.json' # change this
client = speech_v1.SpeechClient()

def sample_recognize(local_file_path):
    """
    Transcribe a short audio file using synchronous speech recognition

    Args:
      local_file_path Path to local audio file, e.g. /path/audio.wav
    """

    client = speech_v1.SpeechClient()

    # local_file_path = 'resources/brooklyn_bridge.raw'

    # The language of the supplied audio
    language_code = "en-US"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.FLAC
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }
    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript.upper()))
    return alternative.transcript.upper()

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int
    """
    # initialisation
    import numpy

    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

path = os.getcwd()+"/LibriSpeech/dev-clean/84/121123/" # change this

# Get file locations
audio_files = []
for f in os.listdir(path):
    if f[-4:]==".txt":
        transcript = f
    else:
        audio_files.append(f)

# Get audio files
locs = []
for f in audio_files:
    locs.append(f[:-5])

# Get original transcript
with open(transcript, "r") as f:
    trans = f.read().split("\n")

# Using google speech-to-text
# sample usage : sample_recognize("audio_file_path")
Texts = []
Trans = []
for name in locs:
    for file in trans:
        if name in file:
            Texts.append(file.replace(name,""))
            print("Original:",file.replace(name,""))
            Trans.append(sample_recognize(path+"/"+name+".flac"))

Scores = []
for i in range(len(Texts)):
    score = wer(Texts[i].split(),Trans[i].split())
    Scores.append(score)

print("Average WER score:",sum(Scores)/len(Scores))

