from os import path

from typing import Iterable
from marynlp.modules.module import Module

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

from flair.embeddings import TokenEmbeddings, FlairEmbeddings


class KiEmbeddings(Module):
    module_name = 'flair-h_128-nl_1-f-char_lvl'

    @staticmethod
    def _prebuild(swahili_corpus: list, swahili_corpus_path: str, embedding_path: str):
        # Prepare the corpus and stores the contents the path
        prepare(swahili_corpus, main_dir=swahili_corpus_path)

        # are you training a forward or backward LM?
        is_forward_lm = True

        # load the default character dictionary
        dictionary: Dictionary = Dictionary.load('chars')

        # get your corpus, process forward and at the character level
        corpus = TextCorpus(swahili_corpus_path,
                            dictionary,
                            is_forward_lm,
                            character_level=True)

        # instantiate your language model, set hidden size and number of layers
        language_model = LanguageModel(dictionary,
                                    is_forward_lm,
                                    hidden_size=128,
                                    nlayers=1)

        # train your language model
        trainer = LanguageModelTrainer(language_model, corpus)

        trainer.train(embedding_path,
                    learning_rate=0.1,
                    sequence_length=10,
                    mini_batch_size=10,
                    max_epochs=2)

    @classmethod
    def build(cls, model_dir_path: str = None, model_version: str = 'best-lm.pt', swahili_corpus: Iterable[str] = None, swahili_corpus_path: str = None) -> TokenEmbeddings:
        # Check if the embedder exists
        model_dir = model_dir_path

        # If it doesn't exist, set up the path
        if model_dir is None:
            model_dir = EMBEDDING_PATH

        full_model_path = cls.get_full_model_path(model_dir)
        embedding_model_path = cls.get_embedding_model_path(model_dir, model_version)

        if not path.exists(embedding_model_path):
            if swahili_corpus is None:
                raise ValueError(f"'swahili_corpus' args must contain value if model doesn't exist")

            # first retrains the model since it doesnt exist
            cls._prebuild(swahili_corpus, swahili_corpus_path, embedding_path=full_model_path)

        # -----------------------------------
        # ACTUAL STRUCTURE OF THE EMBEDDING
        # -----------------------------------
        return FlairEmbeddings(embedding_model_path)

    @classmethod
    def from_pretrained(cls, model_dir_path: str, model_type: str = 'best-lm.pt'):
        # without including the name of the model (which depends of the type trained on)
        # simply retrieve the directory

        return FlairEmbeddings(cls.get_embedding_model_path(model_dir_path, model_type))
