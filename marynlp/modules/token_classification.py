from pathlib import Path
from flair.data import Sentence
from flair.models import SequenceTagger
from marynlp.data.transformers.specialized import (DataTextTransformer,
                                                   SwahiliTextTransformer,
                                                   TextNormalizer,
                                                   FlairSupportTransformer)

DEFAULT_TEXT_TRANSFORMERS_LIST = [SwahiliTextTransformer, TextNormalizer, FlairSupportTransformer]


class TokenClassifier(object):
    KNOWN_MODEL_SOURCES = {
        'sw-ner-base': ('resources/taggers/ner-general-base-model/', 'label')
    }

    DATA_TEXT_TRANSFORMERS = {
        'sw-ner-base': DEFAULT_TEXT_TRANSFORMERS_LIST
    }

    def __init__(self,
                 tagger: SequenceTagger,
                 token_label_type: str,
                 stacked_transformers: list = None):

        self.tagger = tagger
        self.token_label_type = token_label_type
        self._st_transforms = stacked_transformers

    def transform(self, text: str):
        if self._st_transforms is None:
            return text

        assert isinstance(self._st_transforms, list), 'Transforms must be a list'

        for ix, trnsf in enumerate(self._st_transforms):
            assert isinstance(trnsf,
                              DataTextTransformer), 'Expected object at index {} to be a {}, instead received a {}' \
                .format(ix, DataTextTransformer.__name__, type(trnsf).__name__)

            text = trnsf(text)

        return text

    def classify(self, text: str):
        sentence = Sentence(self.transform(text), language_code='sw')
        self.tagger.predict(sentence)

        return sentence

    @classmethod
    def from_pretrained(cls, src: str, token_label_type: str = None):
        model_src = src.lower()
        if model_src in cls.KNOWN_MODEL_SOURCES:
            # might want to download the model first
            # and store it in this path
            actual_path, label_type = cls.KNOWN_MODEL_SOURCES[model_src]

            tagger = SequenceTagger.load('{}/best-model.pt'.format(actual_path))

            # load the contents
            if model_src in cls.DATA_TEXT_TRANSFORMERS:
                _trans_class_list = cls.DATA_TEXT_TRANSFORMERS[model_src]
                trns_stack = [_cls() for _cls in _trans_class_list]

                # load with text transformers
                return cls(tagger, label_type, trns_stack)

            # load without text transformers
            return cls(tagger, label_type)

            # checks if the path exists
        assert Path(src).exists(), 'The model path doesn\'t exist'

        # otherwise, require token_label_type
        assert token_label_type is not None, \
            'If \'{}\' is not from an established source, you must indicate the \'{}\''.format(
                'src', 'token_label_type'
            )

        return cls(SequenceTagger.load(src), token_label_type)
