from pathlib import Path
from flair.data import Sentence
from flair.models import SequenceTagger as FlairSequenceTagger

from marynlp.data.transformers import DataTextTransformer
from .module import Module, pretrained_models_class
from marynlp import SWAHILI_LANGUAGE_CODE


class TokenClassifier(Module):
    @property
    def pretrained_models(self):
        return pretrained_models_class['taggers']

    def __init__(self,
                 tagger: FlairSequenceTagger,
                 token_label_type: str,
                 transformers: DataTextTransformer = None):

        self.tagger = tagger
        self.token_label_type = token_label_type

        if transformers:
            self.transform = transformers

    def classify(self, text: str):
        sentence = Sentence(self.transform(text), language_code=SWAHILI_LANGUAGE_CODE)
        self.tagger.predict(sentence)

        return sentence

    @classmethod
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

        # use model path
        model_dir_path = Path(model_dir_path)

        assert model_dir_path.exists(), 'Model path doesn\'t exist'

        # setting the contents to load the data
        model_full_path = model_dir_path.joinpath(model_option)

        return cls(FlairSequenceTagger.load(model_full_path), **clf_kwargs)
