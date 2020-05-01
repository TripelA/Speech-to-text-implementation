# Code repository
**Set up**

In order to install required packages to run this codebase, first be sure to have Ananconda and pytorch installed and please execute the below commands

`pip install -r requirements.txt`

**To get the CTC loss function, install this fork for Warp-CTC bindings:**

`git clone https://github.com/SeanNaren/warp-ctc.git`

`cd warp-ctc; mkdir build; cd build; cmake ..; make`

`export CUDA_HOME="/usr/local/cuda"`

`cd ../pytorch_binding && python setup.py install`


**Install NVIDIA apex:**

`git clone --recursive https://github.com/NVIDIA/apex.git`

`cd apex && pip install .`

**Finally clone this repo and run this within the repo:**

`pip install -r requirements.txt`

**Models**

This analysis relies on some pretrained DeepSpeech models, which are unfortunately too large to upload directly to GitHub. In order to use them, please create a `models` directory within the code folder by running `mkdir models; cd models`. Once this folder is created and you are inside the folder, run the following commands:

> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth` (this is the an4 model)
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth` (this is the LibriSpeech Model)
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth` (this is the Tedlium dataset)
These models must be downloaded in order to use the [`Transcribe_and_Compare.py`](https://github.com/TripelA/ML2_FinalProject/blob/master/code/Transcribe_and_compare.py) code

**Dataset**

To download the voxforge training and testing sets, please clone the following repo using the commands:

> `git clone https://github.com/chentinghao/download_google_drive.git`
>
> `cd download_google_drive`
>
> `python download_gdrive.py 1yqf-NFeGHOh2RtqISmNlgPCsHsNuM_h_ PATH/FOR/DOWNLOAD/NAME.zip` <- set whatever filepath and name you want
>
> `cd PATH/FOR/DOWNLOAD/`
>
> `unzip NAME.zip`

(The file can additionally be found [here](https://drive.google.com/file/d/1yqf-NFeGHOh2RtqISmNlgPCsHsNuM_h_/view?usp=sharing))

This will create a directory called `transfer_set`, with two nested directories, `train` and `test`. Keep the filepaths to the `train` and `test` directories available, as you will be able to use them in the `Transcribe_and_compare.py` script as the wav directory and the txt directories within the `test` folder. Additionally, the contents of the `train` folder will be available for any model tuning. 
