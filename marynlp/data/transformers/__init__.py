from overrides import overrides
from typing import List
from nltk.tokenize.regexp import RegexpTokenizer
from marynlp.data.reference_vocabulary import LowerCaseSwahiliRV


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


class StackedDataTextTransformer(DataTextTransformer):
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


class TagDataTransformer(DataTextTransformer):
    def __init__(self, ref_vocab: LowerCaseSwahiliRV):
        self.ref_vocab = ref_vocab
        # initiate regex
        self._init_regex()

        self.tokenizer = RegexpTokenizer(self.token_regex)

    def _init_regex(self):
        word_regex, indi_word_regex = self._get_word_regex()
        self.word_regex = word_regex
        self.indi_word_regex = indi_word_regex

    @property
    def word_tag_pair_regex(self):
        return self.get_word_tag_regex(self.regex_word, self.regex_tag)

    @property
    def regex_word(self):
        return '[{self.word_regex}(\s){self.ref_vocab.base_word_non_letter_chars}]+'

    @property
    def regex_tag(self):
        raise NotImplementedError()

    def get_word_tag_regex(self, regex_word, regex_tag):
        raise NotImplementedError()

    @property
    def token_regex(self):
        net_regex = f'({self.indi_word_regex})|({self.word_tag_pair_regex})'
        return net_regex

    def _get_regex_punctuations(self):
        filtered_punctuations = ''.join([c for c in self.ref_vocab.base_punctuations if c not in list(',[]')])
        regex_punctuation = ''.join([f'\\{c}' for c in filtered_punctuations])

        return regex_punctuation

    def _get_word_regex(self):
        # word token with space
        # for sentence matching
        word_regex_ls = [
            f'{self.ref_vocab.regex_word}',
            self._get_regex_punctuations(),
        ]

        indi_word_regex_ls = [
            f'[{self._get_regex_punctuations()}]',
            f'[{self.ref_vocab.regex_word}]+',
        ]

        return "".join(f"({rgx})" for rgx in (self.ref_vocab.regex_special_tokens + word_regex_ls)), \
               "|".join(self.ref_vocab.regex_special_tokens + indi_word_regex_ls)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
