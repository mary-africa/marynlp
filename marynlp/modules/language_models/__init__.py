from overrides import overrides
from pathlib import Path
from ..module import Module, pretrained_models_class


class PretrainedLanguageModel(Module):
    pretrained_models = pretrained_models_class['language_model']

    @classmethod
    @overrides
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **clf_kwargs):

        return cls(FlairTextClassifier.load(model_full_path), **clf_kwargs)