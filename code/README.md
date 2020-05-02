# Code repository
**Set up**

In order to install required packages to run this codebase, first be sure to have Ananconda and pytorch installed and please execute the below commands

`git clone https://github.com/TripelA/Final-Project-Group6.git`

`cd Final-Project-Group6/code/`

`sudo pip install -r requirements.txt`

**To get the CTC loss function, install this fork for Warp-CTC bindings:**

` cd ../..`

`git clone https://github.com/SeanNaren/warp-ctc.git`

`sudo apt install cmake`

`cd warp-ctc; mkdir build; cd build; cmake ..; make`

`export CUDA_HOME="/usr/local/cuda"`

`cd ../pytorch_binding`

`sudo python3 setup.py install`


**Install NVIDIA apex:**

`git clone --recursive https://github.com/NVIDIA/apex.git`

`cd apex && pip install .`

Finally cd again to main code repo

`cd Final-Project-Group6/code/`


**Models**

This analysis relies on some pretrained DeepSpeech models, which are unfortunately too large to upload directly to GitHub. In order to use them, please create a `models` directory within the code folder by running `mkdir models; cd models`. Once this folder is created and you are inside the folder, run the following commands:

> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth` (this is the an4 model)
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth` (this is the LibriSpeech Model)
> - `wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth` (this is the Tedlium dataset)
These models must be downloaded in order to use the [`Transcribe_and_Compare.py`](https://github.com/TripelA/ML2_FinalProject/blob/master/code/Transcribe_and_compare.py) code

**Dataset**

To download the voxforge training and testing sets, please use the following commands within the data folder (will make redefining later filepaths easier)

> `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ldkgRaV0bIn-Qlw1KW_BAjjUsqt36wCG' -O voxforge_train_test2.7z`
>
> `sudo apt-get install p7zip-full`
>
> `7z  x voxforge_train_test2.7z`
>
> `cd voxforge_train_test2`
> 
> `mv voxforge_sample_files ..; cd ..;`


(The file can additionally be found [here](https://drive.google.com/open?id=1ldkgRaV0bIn-Qlw1KW_BAjjUsqt36wCG))

This will create a directory called `transfer_set`, with two nested directories, `train` and `test`. Keep the filepaths to the `train` and `test` directories available, as you will be able to use them in the `Transcribe_and_compare.py` script as the wav directory and the txt directories within the `test` folder. Additionally, the contents of the `train` folder will be available for any model tuning. 

Once the dataset is load to create csv file like  /path/to/audio.wav,/path/to/text.txt /path/to/audio2.wav,/path/to/text2.txt  ... run following command

`python voxforge.py`
