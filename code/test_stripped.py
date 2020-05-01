from tqdm import tqdm
import torch


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False):
    # set model to eval functionality
    model.eval()

    # initialize values at zero
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0

    # create empty list for storing output
    output_data = []

    # loop through the test_loader data loader
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):

        # unpack the data from the data loader
        inputs, targets, input_percentages, target_sizes = data

        # set input sizes and add to device
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)

        # if the data uses half precision, set the half method
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0

        # for the sizes
        for size in target_sizes:

            # append the offset values and increase the size for the loop
            split_targets.append(targets[offset:offset + size])
            offset += size

        # retrieve the model output
        out, output_sizes = model(inputs, input_sizes)

        # decode the output using the passed decoder and convert output to string format
        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)


        # if there is a location set for saving output
        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu(), output_sizes, target_strings))

        # loop through the length of the target strings
        for x in range(len(target_strings)):

            # get the decoded output and target strings and calculate the WER and CER
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst

            # calculate the number of words and characters
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))

            # if verbose (ie. print everything), show all output and predicted output
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")

    # divide WER and CER by total length of strings in number of words/characters
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, output_data
