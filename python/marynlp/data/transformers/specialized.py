import re
from unicodedata import normalize
from overrides import overrides

from nltk.tokenize.regexp import RegexpTokenizer
from ..reference_vocabulary import LowercaseSwahiliReferenceVocabulary as LowerCaseSwahiliRV, (
    UNK_TOKEN, UNK_CHAR, NUM_TOKEN, \
    REGEX_UNK_TOKEN, REGEX_UNK_CHAR, REGEX_NUM_TOKEN
)
from ..transformers import DataTextTransformer, StackedDataTextTransformer

"""
Text Transformer that are used in modifying tests
"""
class TextNormalizer(DataTextTransformer):
    NORMAL_FORMS = ['NFC', 'NFKC', 'NFKD', 'NFD']

    def __init__(self, normal_form: str = 'NFKC'):
        # Check the normal form
        assert normal_form in self.NORMAL_FORMS, \
            "The '{}' normal is not among the options '{}'".format(normal_form,
                                                                   ','.join(self.NORMAL_FORMS))
        self.form = normal_form

    @overrides
    def transform(self, text: str) -> str:
        """Normalizes the text according to some normal form """
        return normalize(self.form, text)

    @overrides
    def extra_repr(self):
        return 'form={}'.format(self.form)


class SwahiliTextTransformer(DataTextTransformer):
    def __init__(self):
        # binding a lower case swahili reference vocabulary
        self.ref_vocab = LowerCaseSwahiliRV()
        self.special_chars = self.ref_vocab.special_tokens

        self._regex_for_extraction = (
            r'[{}]+'.format(self.ref_vocab.regex_word),
            r'[{}]'.format(self.ref_vocab.regex_non_word),
        )
        self._regex_rule = "|".join(self.ref_vocab.regex_special_tokens + list(self._regex_for_extraction))
        self._regex_tokenizer = RegexpTokenizer(self._regex_rule)

    @property
    def reference_vocabulary(self):
        return self.ref_vocab

    def valid_text_replace(self, text: str):
        # replace all words that dont follow swahili with '[UNK]'
        text = re.sub(self.ref_vocab.inverse_regex_word, UNK_TOKEN, text)

        # replace all non-allowed characters with [UNKC]
        text = re.sub(self.ref_vocab.inverse_regex_non_word, UNK_CHAR, text)

        # replace all number like patters with [NUM]
        text = re.sub(self.ref_vocab.regex_for_numbers, NUM_TOKEN, text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return self._regex_tokenizer.tokenize(text)

    @overrides
    def transform(self, text: str) -> str:
        # convert to lower case
        text = text.lower()

        # replace swahili invalid words
        text = self.valid_text_replace(text)

        # form the string with model ready structure
        text = ' '.join(self.tokenize(text))

        return text

class TagDataTransformer(DataTextTransformer):
    def __init__(self, ref_vocab: LowerCaseSwahiliRV):
        self.ref_vocab = ref_vocab
        # initiate regex
        self._init_regex()

        # TODO: remove `RegexpTokenizer` since its the only reason to 
        #  depend on the `nltk` package
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

"""
TODO: Add docs
IOB-Scheme NER Tagging
"""

class NERDataTransformer(TagDataTransformer):
    @overrides
    def get_word_tag_regex(self):
        return f'\[({self.regex_word}),([{self.regex_tag}]+)\]'

    @property
    def regex_tag(self):
        return r'A-Za-z\s'

    def iob_encode(self, text, main_tag) -> List[Tuple[str, str]]:
        tag = B_TAG
        for en_, word in enumerate(re.split('\s+', text)):
            if en_ != 0:
                tag = I_TAG

            # remove spaces in tags like WORK OF ART, and sub them with '-'
            #   so it becomes WORK-OF-ART
            main_tag = re.sub(r'\s+', '_', main_tag)
            yield word, f'{tag}-{main_tag}'.upper()

    @overrides
    def transform(self, text: str, lower: bool = True) -> Iterator[Tuple[str, str]]:
        if lower:
            text = text.lower()

        for o_word, tagged_word, *tag_content in self.tokenize(text):
            if o_word.strip():
                # this is a word
                # O-tag it
                yield o_word, O_TAG
            elif tagged_word.strip():
                # bio tag
                word, tag = tag_content
                for word_biotag_pair in self.iob_encode(word.strip(), tag.strip()):
                    yield word_biotag_pair

"""
TODO: Add docs
Custom Togging:
POS Tags
"""

# punctuation tag
PUNCT = 'PUNCT'

# number tag
NUM = 'NUM'

# unknown word tag
X = 'X'


class POSDataTransformer(TagDataTransformer):
    @overrides
    def get_word_tag_regex(self, regex_word, regex_tag):
        return r'({0})\[({1})\]'.format(regex_word, regex_tag)

    @property
    def token_regex(self):
        net_regex = f'({self.ref_vocab.regex_for_numbers})|{self.word_tag_pair_regex}|({self.indi_word_regex})'
        return net_regex

    @property
    def regex_word(self):
        return r'[{}]+|[{}]'.format(self.ref_vocab.regex_word, self.ref_vocab.regex_non_word)

    @property
    def regex_tag(self):
        return r'[A-Za-z]+'

    @overrides
    def transform(self, text: str, lower: bool = True) -> Iterator[Tuple[str, str]]:
        if lower:
            text = text.lower()
            text = re.sub(f'\[{PUNCT}\]|\[{NUM}\]'.lower(), '', text)
        else:
            text = re.sub(f'\[{PUNCT}\]|\[{NUM}\]', '', text)

        for number, *num_grp, word, tag, other in self.tokenize(text):
            if number.strip():
                yield number, NUM
            elif other.strip():
                # check if word is punctuation
                if re.match(f'[{self.ref_vocab.regex_non_word}]', other):
                    yield other, PUNCT

                else:
                    yield other, X
            elif word.strip():
                if tag.upper() == 'N':
                    tag = 'noun'

                yield word, tag.upper()

"""
Flair related transformers
"""
class FlairSupportTransformer(DataTextTransformer):
    """To support how flair embedds the words

    Ensure the sub-pairs replacement tokens aren't in the set of allowed characters
    """
    sub_pair = [
        (REGEX_UNK_TOKEN, '^'),
        (REGEX_UNK_CHAR, '#'),
        (REGEX_NUM_TOKEN, '$'),
    ]

    def transform(self, text: str):
        transformed = text
        for regex_val, sub_token in self.sub_pair:
            transformed = re.sub(regex_val, sub_token, transformed)

        return transformed

    def get_replacement_chars(self):
        """
        Returns the values
        """
        return [x[-1] for x in self.sub_pair if x[0] != REGEX_UNK_TOKEN]



"""
Miscellaneous Transformers
"""
class FtSupportTextTransformer(DataTextTransformer):
    """
    Helper transformer that helps with dealing with fasttext related 
    transformation
    """
    def transform(self, text: str):
        text = re.sub(r'\'\-', '', text)
        return text

class NormSwahiliTextTransformer(StackedDataTextTransformer):
    def __init__(self):
        super().__init__([
            TextNormalizer(),
            SwahiliTextTransformer()
        ])

class FlairDataTextTransformer(StackedDataTextTransformer):
    def __init__(self):
        super().__init__([
            NormSwahiliTextTransformer(),
            FlairSupportTransformer()
        ])
