from abc import abstractmethod
from typing import List, Union

from marynlp.modules.module import Module

class BaseEmbeddingModule(Module):
    @classmethod
    def pretrained_categories(cls):
        return 'embeddings'

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


