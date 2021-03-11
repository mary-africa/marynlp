"""
The embedding produced by the modules here are supposed to generate a single embedding for a token regardless of the context.
You think of them as deterministic embeddings are being produced regardless of the sentence that its being used in
"""

from ..module import FlairModule
from .base import TokenEmbeddings
from typing import Union, List


from overrides import overrides

from ...utils import flair
from flair.embeddings.token import FastTextEmbeddings as BaseFlairFastTextEmbeddings
from flair.data import Sentence


class WordEmbeddings(TokenEmbeddings):
    @classmethod
    @overrides
    def from_pretrained(src: str, credentials_json_path: str, **model_kwargs):
        pass


class FastTextWordEmbeddings(TokenEmbeddings):
    def __init__(self, embedder) -> None:
        self._emb = embedder

    def single_text_embed(self, text: str):
        """Individually embed the texts"""
        sent = Sentence(text, use_tokenizer=False)
        self._emb.embed(sent)

        return [token.get_embedding() for token in sent][0]

    def embed(self, texts: Union[str, List[str]]):
        """This does the typical embedding for the flair embeddings"""
        if isinstance(texts, str):
            return self.single_text_embed(texts)
        else:
            assert isinstance(texts, list), "Input must be a str or List[str]"
            return list(map(self.single_text_embed, texts))

    def embedding_length(self) -> int:
        return self._emb.embedding_length

    @classmethod
    @overrides
    def from_pretrained(cls, src: str, inner_model_file: str, credentials_json_path: str=None):
        # load the model
        model_dir_path = cls.prepare_pretrained_model_path(src, credentials_json_path=credentials_json_path)

        # construct the full model path
        # according to flair
        model_full_path = flair.build_model_full_path(model_dir_path, inner_model_file)

        return cls(BaseFlairFastTextEmbeddings(model_full_path, use_local=True))
