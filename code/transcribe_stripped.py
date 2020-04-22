import warnings
import torch
warnings.simplefilter('ignore')


def transcribe(audio_path, spect_parser, model, decoder, device):

    # convert the file in the audio path to a spectrogram - see data_loader/SpectrogramParser for more info
    spect = spect_parser.parse_audio(audio_path).contiguous()

    # nest the spectrogram within two arrays - why?? Look in model
    # think it might have to do with the first layer being a conv2d, so it needs channel values
    # but why 4d?
    # produces a 1x1x161x391
    # 1: 1 wav file/spectrogram
    # 1: 1 x value
    # 161: 161 y values in spectrogram?
    # 391: # channels?
    spect = spect.view(1, 1, spect.size(0), spect.size(1))

    # move the spectrogram to the device
    spect = spect.to(device)

    # empty tensor with number of inputs
    input_sizes = torch.IntTensor([spect.size(3)]).int()

    # model the spectrogram and produce the output and output sizes
    out, output_sizes = model(spect, input_sizes)

    # decode the output sizes
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets
