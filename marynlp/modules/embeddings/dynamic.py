"""
This is to contain the module for static embeddings
such as Flair, ELMo, TransformerEmbeddings, ...

"""
from ..module import Module, pretrained_models_class
from marynlp.data.transformers import DataTextTransformer, StackedDataTextTransformer
from marynlp.data.transformers.specialized import TextNormalizer, FlairSupportTransformer, SwahiliTextTransformer

from flair.embeddings import FlairEmbeddings
from typing import Union, List

from flair.data import Sentence

from marynlp import SWAHILI_LANGUAGE_CODE


class DynamicEmbeddings(Module):
    def embed(self, text: Union[str, List[str]]):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **module_kwargs):
        raise NotImplementedError()


class SwahiliFlairEmbeddings(DynamicEmbeddings):
    # this filters all the language models that have `flair-` in them
    pretrained_models = { k: v for k, v in pretrained_models_class['language-model'].items() if 'flair' in k }

    def __init__(self, embeddings: FlairEmbeddings, transformer: DataTextTransformer=None):
        self.embeddings = embeddings

        if transformer is None:
            self.transform = StackedDataTextTransformer([TextNormalizer(),
                                                         SwahiliTextTransformer(),
                                                         FlairSupportTransformer()
                                                        ])
        else:
            self.transform = transformer

    def embed(self, texts: Union[str, List[str]]) -> List[Sentence]:
        """removing the abstraction for Sentences

        """
        if type(texts) == str:
            sentences = Sentence(self.transform(texts), language_code=SWAHILI_LANGUAGE_CODE)
        elif type(texts) == list:
            sentences = [Sentence(self.transform(text), language_code=SWAHILI_LANGUAGE_CODE) for text in texts]
        else:
            print(texts)
            raise RuntimeError('Texts must be \'{}\' or \'{}\', instead got \'{}\''.format('str', 'List[str]', type(texts).__name__))

        self.embeddings.embed(sentences)
        return sentences

    @classmethod
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **module_kwargs):
        lang_model_path = cls.get_full_model_path(src=src,
                                                  credentials_json_path=credentials_json_path,
                                                  model_option='best-lm.pt')

        return cls(FlairEmbeddings(lang_model_path))
