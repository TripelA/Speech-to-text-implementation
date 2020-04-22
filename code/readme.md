# Code repository

**Models**

This analysis relies on some pretrained DeepSpeech models, which are unfortunately too large to upload directly to GitHub. In order to use them, please create a `models` directory within the code folder by running `mkdir models; cd models`. Once this folder is created and you are inside the folder, run the following commands:
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth` (this is the an4 model)
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth` (this is the LibriSpeech Model)
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth` (this is the Tedlium dataset)
These models must be downloaded in order to use the [`Transcribe_and_Compare.py`](https://github.com/TripelA/ML2_FinalProject/blob/master/code/Transcribe_and_compare.py) code
