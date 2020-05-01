# Machine Learning 2 Final Project - Group 6

Repo for ML2 final project! :) 

Check the Wiki tab for data links and other resources.

## PROJECT OVERVIEW:

Languages are an integral part of cultures worldwide. Due to the sheer variety of languages as well as the infinite permutations of accents and dialects, speech recognition is a complex problem that is growing more relevant in our day-to-day lives. With the advent of technologies like Siri, the Amazon Echo, and Google Home, major technology companies are bringing these voice-to-text recognition systems into tasks that users perform every day. Each of these major tech companies has likely spent years of development time, thousands of hours of speech data, and billions of calculations to arrive at these final neural architectures used to recognize a voice when speaking into a cell phone or Alexa speaker, and translate it to a series of commands. These speech recognition systems are all highly advanced, and do not provide many with the ability to look under the hood and see what is actually going on when you ask Siri to pull up directions home. Deep Speech2, end-to-end Automatic Speech Recognition (ASR) engine, allows us the opportunity to study and understand one of these highly sophisticated models, and **see if transfer learning techniques can be used to compare the capabilities of this open-source model with that of Google’s Speech-To-Text API.** 

The Deep Speech 2 model is a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) model that transcribes language by converting spectrograms of audio data to an attempted series of alphabet characters that are believed to be present in the audio file. The original model was developed by Baidu, an internet and artificial intelligence company, and published in their paper Deep Speech 2: End-to-End Speech Recognition in English and Mandarin. An implementation of Deep Speech2 that we used for this project came courtesy of GitHub user SeanNaren, whose code and pre-trained models were used as the basis of our project. Our solution was to compare the performance of three pre-trained Deep Speech 2 implementations against Google’s Speech-to-Text API, select the best performing model, and use transfer learning techniques to see how close we can get an open-source model to that of a technology giant. 


## REQUIREMENTS:
Please visit the `code` tab for a guide on how to install and configure your machine in order to run the code for this project

## DATA: 
The data for this project can be found at the following locations:

- An4 : http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz

- LibriSpeech (test-clean) : http://www.openslr.org/resources/12/test-clean.tar.gz
- LibriSpeech (dev-clean) : http://www.openslr.org/resources/12/dev-clean.tar.gz
- LibriSpeech (train-clean-360) : http://www.openslr.org/resources/12/train-clean-360.tar.gz

- TED-LIUM : http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz

- Voxforge : http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/

Additionally, if you do not want to download and configure all the files yourself, please clone the original Deep Speech pytorch repository using the command `git clone https://github.com/SeanNaren/deepspeech.pytorch.git`, where you can then run the `data/an4.py`, `data/librispeech.py`, `data/ted.py` and/or `data/voxforge.py` scripts to retrieve and format the audio files for you.

Be aware that downloading all of the audio files and configuring the directories will take _2-3 hours_. As most of the models investigated are pre-trained, to shorten the download time simply download the LibriSpeech data, as the specific Voxforge files used for testing can be found in a zip file on Google Drive, linked in the Code directory. 

## CODE

The code can generally be divided into a few different categories:

#### Data Loading

`data_loader_stripped.py` - a file used to create data loader objects, parsers, and spectrogram classes

`create_train_val_set.py` - a file used to generate the Voxforge training and validation sets

`utils_stripped.py` - a file housing miscelaneous utility functions, including the `load_data()` function

#### Model Training

`model.py` - a file storing the Deep Speech 2 model class and associated properties

`train_stripped.py` - a file used to train new models and perform transfer learning on existing models

#### Transcription

`decoder_stripped.py` - a file used to create the Greedy Decoder class for decoding model output

`Google_API.py` - a file used to transcribe audio files with Google's Speech-to-Text API

`Transcribe_stripped.py` - a file used to house the transcribe functionality

`Transcribe_and_Compare.py` - a file used to transcribe data using a given model and calculate WER and CER measures
