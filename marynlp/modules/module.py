from abc import ABC, abstractmethod, abstractproperty


class Module(ABC):
    def __init__(self):
        pass

    @abstractmethod
    @property
    def module_name(self) -> str:
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        """Build the module"""
        pass

    @classmethod
    def get_full_model_path(cls, model_dir: str) -> str:
        return f'{model_dir}/{cls.module_name}'

    @classmethod
    def get_embedding_model_path(cls, model_dir: str, model_version: str) -> str:
        """Get the location where the trained embedder is stored"""
        return f'{cls.get_full_model_path(model_dir)}/{model_version}'

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # without including the name of the model (which depends of the type trained on)
        # simply retrieve the directory

        raise NotImplementedError()
