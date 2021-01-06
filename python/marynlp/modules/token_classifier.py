from .module import Module, FlairModule
from pathlib import Path
from overrides import overrides

from ..utils import flair
from flair.models import SequenceTagger as FlairSequenceTagger

class TokenClassifier(Module):
    @classmethod
    @overrides
    def from_pretrained(src: str, credentials_json_path: str, **model_kwargs):
        pass

class FlairTokenClassifier(TokenClassifier, FlairModule):
    @classmethod
    @overrides
    def from_pretrained(cls, src: str, inner_model_file: str, credentials_json_path: str=None):
        # load the model
        model_dir_path = cls.prepare_pretrained_model_path(src, credentials_json_path=credentials_json_path)

        # construct the full model path
        # according to flair
        model_full_path = flair.build_model_full_path(model_dir_path, inner_model_file)

        return FlairSequenceTagger.load(model_full_path)
