from abc import abstractmethod
from typing import List, Union

from marynlp.modules.module import Module

class BaseEmbeddingModule(Module):
    @classmethod
    def pretrained_categories(self)
        return 'embeddings'

    """This is responsible for loading the embedding model from GCP or local repository"""
    @classmethod
    def from_pretrained(cls, src: str, credentials_json_path: str):
        # This gets the models folder as reference in the PRETRAINED_MODEL
        cls.prepare_pretrained_model_path(src: str, credentials_json_path: str)

class TokenEmbeddings(BaseEmbeddingModule):
    """Adds the interfaces that is required to use the embedding components"""
    @abstractmethod
    def embed(self, text: Union[str, List[str]]):
        """Takes in a sentences, or list of sentences so that it can be converted into embeddings"""
        raise NotImplementedError()

    @abstractmethod
    def embedding_length(self):
        """To return the length of the embeddings"""
        raise NotImplementedError()
