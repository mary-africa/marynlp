"""
NOTE: exists for the sake of the 
"""
import torch
from .character import CharacterEncoder


class GreedyEncoder(object):
    def __init__(self, text_encoder: CharacterEncoder):
        self.text_encoder = text_encoder

    def decode_target(self,
                      output: torch.Tensor,
                      labels: torch.Tensor,
                      label_lengths,
                      collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        blank_label = self.text_encoder.BLANK_INDEX

        for i, args in enumerate(arg_maxes):
            decoded = []
            decode_text = self.text_encoder.decode(labels[i][:label_lengths[i]].tolist())
            targets.append(decode_text)
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j -1]:
                        continue

                    decoded.append(index.item())

            decodes.append(self.text_encoder.decode(decoded))

        return decodes, targets

    def decode_test(self,
                    output,
                    collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        blank_label = self.text_encoder.BLANK_INDEX

        for i, args in enumerate(arg_maxes):
            decoded = []
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decoded.append(index.item())

            decodes.append(self.text_encoder.decode(decoded))

        return decodes
