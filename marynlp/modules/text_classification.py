from marynlp.data.transformers import DataTextTransformer
from typing import List
from pathlib import Path
from overrides import overrides

from .module import Module, pretrained_models_class
from marynlp import SWAHILI_LANGUAGE_CODE

from flair.models import TextClassifier as FlairTextClassifier
from flair.data import Sentence


class TextClassifier(Module):
    @property
    def pretrained_models(self):
        return pretrained_models_class['classifier']

    def __init__(self, text_clf: FlairTextClassifier, transformer: DataTextTransformer):
        self.text_clf = text_clf
        self.transform = transformer

    def classify(self, text: List[str]):
        sentence = Sentence(self.transform(text), language_code=SWAHILI_LANGUAGE_CODE)
        self.text_clf.predict(sentence)

        return sentence

    @classmethod
    @overrides
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **clf_kwargs):
        src = src.lower()
        model_option = 'best-model.pt'

        if src in cls.pretrained_models:
            # check if the credentials key exists
            assert credentials_json_path, 'If using pre-trained model, you need to include the credentials key file'
            assert Path(credentials_json_path).exists(), "Credentials key file doesn't exist"

            model_dir_path = cls._get_pretrained_model_path(src, credentials_json_path)
        else:
            model_dir_path = src

        # use path model
        model_dir_path = Path(model_dir_path)

        assert model_dir_path.exists(), 'Model path doesn\'t exist'

        # setting the contents to load the data
        model_full_path = model_dir_path.joinpath(model_option)

        return cls(FlairTextClassifier.load(model_full_path), **clf_kwargs)
