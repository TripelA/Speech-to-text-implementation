import Levenshtein as Lev
import torch
from six.moves import xrange


class Decoder(object):

    # initialize decoder
    def __init__(self, labels, blank_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"

        # initialize labels (basically alphabet)
        self.labels = labels

        # stores dictionary of labels and place in list of labels (ie. 0:'_' since it's the first label), passed from
        # the model with model.labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

        # where the underscore is located
        self.blank_index = blank_index

        #
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    # can use the decoder to calculate wer and cer, or just use lev distance on the outputs
    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError

# LEFT IN TO NOT POTENTIALLY BREAK OTHER CODE, BUT NOT USED DURING TRAINING
class BeamCTCDecoder(Decoder):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None

        # xrange used to save memory
        for x in xrange(len(sequences)):

            # specify sequence length
            seq_len = sizes[x] if sizes is not None else len(sequences[x])

            # process string
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)

            # append string to overall strings
            strings.append([string])  # We only return one path

            # if we want the offsets, append the offsets
            if return_offsets:
                offsets.append([string_offsets])

        # return values
        if return_offsets:
            return strings, offsets
        else:
            return strings

    # function to process string from predicted character indices
    def process_string(self, sequence, size, remove_repetitions=False):

        # initialize
        string = ''
        offsets = []

        # loop through each piece of the output within window
        for i in range(size):

            # turn from integer to character based on dictionary
            char = self.int_to_char[sequence[i].item()]

            # if the character is not the blank index:
            if char != self.int_to_char[self.blank_index]:

                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass

                # if the character is a space, add a space to the string and set this as an offset location
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)

                # append string and offsets
                else:
                    string = string + char
                    offsets.append(i)

        # return values
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """

        # get the argmax of dim 2 (3rd dimension), which is the probability of each character from the dictionary of
        # characters defined in the model

        # returns both the probability (saved as _) and the index (max_probs). Don't care about the probability
        _, max_probs = torch.max(probs, 2)

        # convert the list of predicted characters to strings -- see convert_to_strings and process_string above
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets
