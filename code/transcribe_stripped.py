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
