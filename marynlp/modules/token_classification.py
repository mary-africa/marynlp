from flair.data import Sentence
from flair.models import SequenceTagger as FlairSequenceTagger

from ..data.transformers import DataTextTransformer
from .module import Module, pretrained_models_class
from marynlp import SWAHILI_LANGUAGE_CODE


class TokenClassifier(Module):
    pretrained_models = pretrained_models_class['taggers']

    def __init__(self,
                 tagger: FlairSequenceTagger,
                 token_label_type: str,
                 transformer: DataTextTransformer):

        self.tagger = tagger
        self.token_label_type = token_label_type
        self.transform = transformer

    def classify(self, text: str):
        sentence = Sentence(self.transform(text), language_code=SWAHILI_LANGUAGE_CODE)
        self.tagger.predict(sentence)

        return sentence

    @classmethod
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **clf_kwargs):
        model_full_path = cls.get_full_model_path(src=src,
                                                  credentials_json_path=credentials_json_path)

        return cls(FlairSequenceTagger.load(model_full_path), **clf_kwargs)
