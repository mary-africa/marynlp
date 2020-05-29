from unicodedata import normalize
from overrides import overrides
from .reference_vocabulary import LowerCaseSwahiliRV
from nltk.tokenize import RegexpTokenizer
from typing import List

from .reference_vocabulary import UNK_TOKEN, UNK_CHAR, NUM_TOKEN, \
    REGEX_UNK_TOKEN, REGEX_UNK_CHAR, REGEX_NUM_TOKEN


class DataTextTransformer(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def extra_repr(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """This calls the transform()"""
        return self.transform(*args, **kwargs)

    def __repr__(self):
        return f'{type(self).__name__}(self.extra_repr())'


class StackedTextTransformer(DataTextTransformer):
    def __init__(self, transformers: List[DataTextTransformer]):
        assert len(transformers) > 0, 'You must include atleast one DataTextTransformer'

        for ix, trn in enumerate(transformers):
            assert isinstance(trn, DataTextTransformer) \
                , 'Expected transformer at index {} to be \'{}\', but got \'{}\'' \
                .format(ix, DataTextTransformer.__name__, type(trn).__name__)

        self.transformers = transformers

    @overrides
    def transform(self, text: str):
        t_text = text
        for trn in self.transformers:
            t_text = trn(t_text)

        return t_text