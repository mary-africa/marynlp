"""
This is to contain the module for static embeddings
such as GloVe, FastText, BPEmb, CharLevelEmbeddings ..

For the sake of the swahili language. Preferably go with `FastText` implementation
"""
from flair.embeddings import FastTextEmbeddings as FlairFastTextEmbeddings
from flair.embeddings.token import TokenEmbeddings as FlairTokenEmbeddings
from abc import abstractmethod, abstractproperty
from typing import Union, List
from ..module import Module

from flair.data import Sentence
from marynlp.data.transformers.specialized import FlairDataTextTransformer, FtSupportTextTransformer
from marynlp.data.transformers import DataTextTransformer, StackedDataTextTransformer

from overrides import overrides
from pathlib import Path

# These are the embedding models types that are (to be) supported
EMB_MODEL_TYPES = ['fasttext', 'bpemb', 'glove']


class TokenEmbeddings(object):
    @abstractmethod
    def embed(self, text: Union[str, List[str]]):
        """Takes in a sentences, or list of sentences so that it can be converted into embeddings"""
        raise NotImplementedError()

    @abstractmethod
    def embedding_length(self):
        """To return the length of the embeddings"""
        raise NotImplementedError()


class WordEmbeddings(TokenEmbeddings, Module):
    def __init__(self, model: FlairTokenEmbeddings, data_transformers: List[DataTextTransformer]):
        self._emb = model
        self._dtf = StackedDataTextTransformer(data_transformers)

    def text_embed(self, texts: Union[str, List[str]]):
        """Embeds the string in a natural way"""
        # Embeddings for the texts
        txt_embeds = []

        if isinstance(texts, str):
            # working with the string
            text = self._dtf(texts)
            return self._indi_text_embed(text)

        else:
            assert isinstance(texts, list), "Input must be a str or List[str]"
            texts = (map(texts, self._dtf))
            return list(map(texts, self._indi_text_embed))

    def _indi_text_embed(self, text: str):
        """Individually embed the texts"""
        sent = Sentence(text)
        self._emb.embed(sent)

        return [(token.text, token.get_embedding()) for token in sent]

    def embed(self, sentences: Union[Sentence, List[Sentence]]):
        """This does the typical embedding for the flair embeddings"""
        return self._emb.embed(sentences=sentences)

    def embedding_length(self) -> int:
        return self._emb.embedding_length

    @classmethod
    @overrides
    def from_pretrained(cls,
                        src: str,
                        model_option=None,
                        credentials_json_path: str = None,
                        model_type: str = 'fasttext',
                        model_name='sw-fasttext-100',
                        **emb_kwargs):
        src = src.lower()

        if src in cls.pretrained_models:
            # check if the credentials key exists
            assert credentials_json_path, 'If using pre-trained model, you need to include the credentials key file'
            assert Path(credentials_json_path).exists(), "Credentials key file doesn't exist"

            model_dir_path = cls._get_pretrained_model_path(src, credentials_json_path)
        else:
            model_dir_path = src

        # use path model
        model_dir_path = Path(model_dir_path)

        assert model_dir_path.exists(), 'model directory \'{}\' doesn\'t exist'.format(model_dir_path)

        if model_option is None:
            # setting model_option as folder_name.bin
            model_option = "%s.bin" (model_dir_path.name)

        # setting the contents to load the data
        model_path = model_dir_path.joinpath(model_option)

        assert model_type in EMB_MODEL_TYPES, "The model '%s' is not supported, " + \
                                                "choose among %r " % (model_type, EMB_MODEL_TYPES)

        # TODO: remove the line below after supporting the models
        assert model_type == 'fasttext', "We only currently support fasttext model"

        if model_type == 'fasttext':
            model = FlairFastTextEmbeddings(model_path)
            model.name = model_name

            data_transfs = [FlairDataTextTransformer(), FtSupportTextTransformer()]
            return cls(model, data_transformers=data_transfs)
