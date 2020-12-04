"""
NOTE: This was built for the voice project
"""

import re
from typing import List


class CharacterEncoder(object):
    """Maps characters in a string as sequence of characters
    the space character should be <SPACE>
    Args:
        file_name (str): The path of the csv file that contains
            the characters to map
        data (`List[str]`): The list of characters that are
            to be used for mapping
    """

    # to indicate space
    _SPACE_ = '<SPACE>'

    def __init__(self, file_name: str = None, data: List[str] = None):
        # TODO: This currently assumes data only.
        #  add feature to support file_name
        self.char2ix = dict(zip(data, range(len(data) + 1)))
        self.ix2char = {v: k for k, v in self.char2ix.items()}

    def encode(self, text: str) -> List[int]:
        """
        Use a character map and convert text to an integer sequence.
        Notice that spaces in the text are also encoded with 1.

        args: text - text to be encoded
        """
        try:
            text = text.strip()
        except:
            print("ERR Text:", text)
            assert False

        characters = [c if not re.match(r'\s', c) else self._SPACE_ for c in list(text)]
        return [self.char2ix[c] for c in characters]

    def decode(self, indices: List[int]) -> str:
        """
        Use a character map and convert integer labels to a text sequence.
        It converts each integer into its corresponding char and joins the chars to form strings.
        Notice that the strings will be separated wherever the integer 1 appears.

        args: labels - integer values to be converted to texts(chars)
        """
        characters = [self.ix2char[ix] for ix in indices]
        return "".join([c if c != self._SPACE_ else ' ' for c in characters])

    @property
    def BLANK_INDEX(self):
        return self.count

    @property
    def count(self):
        """Returns the number of characters"""
        return len(self.char2ix)
