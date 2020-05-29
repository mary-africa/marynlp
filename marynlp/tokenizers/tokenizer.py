from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs):
        pass
