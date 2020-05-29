import pandas as pd
from os import path
from flair.embeddings import DocumentLSTMEmbeddings, CharacterEmbeddings,     TokenEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from flair.data import Corpus
from flair.datasets import ClassificationCorpus

from marynlp.modules.module import Module
from marynlp.utils.train import pd_split_train_val_test
from marynlp.utils.file import save_to_file, make_dir

RESOURCES_PATH = 'resources'
INTENT_CLASSIFIER_PATH = f'{RESOURCES_PATH}/intent_classifer/'


# NOTE: here, the validation path is not `valid.*` but `dev.*`
def prepare(corpus_dataframe: pd.DataFrame, store_dir_path: str):
    corpus_dataframe = corpus_dataframe[['Question', 'Intent']].rename(columns={"Intent": "label", "Question": "text"})
    corpus_dataframe['label'] = '__label__' + (corpus_dataframe['label'].astype(str))
    corpus_dataframe = corpus_dataframe.reindex(columns=['label', 'text'])

    tr, v, ts = pd_split_train_val_test(corpus_dataframe, 0.7, 0.6)
    make_dir(store_dir_path)

    # options to deal with the text
    opts = dict(sep='\t', index=False, header=False)

    tr.to_csv(f'{store_dir_path}/train.txt', **opts)
    v.to_csv(f'{store_dir_path}/dev.txt', **opts)
    ts.to_csv(f'{store_dir_path}/test.txt', **opts)


class IntentClassifier(Module):
    module_name = 'clf-lstm-h_256-wdim_128'

    @classmethod
    def _prebuild(cls, corpus_dataframe_path: str, prepared_corpus_store_path: str, token_embedding: TokenEmbeddings,
                  classifier_path: str):
        """ TODO: Add docs """
        corpus_dataframe = pd.read_csv(corpus_dataframe_path).sample(frac=1).reset_index(drop=True)
        prepare(corpus_dataframe, store_dir_path=prepared_corpus_store_path)

        corpus: Corpus = ClassificationCorpus(prepared_corpus_store_path,
                                              test_file='test.txt',
                                              dev_file='dev.txt',
                                              train_file='train.txt')

        document_embeddings = DocumentLSTMEmbeddings([CharacterEmbeddings(), token_embedding],
                                                     hidden_size=256,
                                                     reproject_words=True,
                                                     reproject_words_dimension=128)

        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=True)

        trainer = ModelTrainer(classifier, corpus)

        trainer.train(classifier_path,
                      learning_rate=0.01,
                      max_epochs=100)

    @classmethod
    def build(cls, embeddings: TokenEmbeddings, model_dir_path: str = None, model_version: str = 'final-model.pt',
              swahili_text_labeled_corpus_df_path: str = None, prepared_corpus_store_path: str = None):
        """ TODO: Add docs """

        model_dir = model_dir_path

        # If it doesn't exist, set up the path
        if model_dir is None:
            model_dir = INTENT_CLASSIFIER_PATH

        full_model_path = cls.get_full_model_path(model_dir)

        # change the name to get_module_model_path
        intent_clf_model_path = cls.get_embedding_model_path(model_dir, model_version)

        if not path.exists(intent_clf_model_path):
            if swahili_text_labeled_corpus_df_path is None:
                raise ValueError(f"'swahili_text_labeled_corpus_df' args must contain value if model doesn't exist")

            assert path.exists(
                swahili_text_labeled_corpus_df_path), "'swahili_text_labeled_corpus_df' path doesn't exist"

            # first retrains the model since it doesnt exist
            cls._prebuild(swahili_text_labeled_corpus_df_path, prepared_corpus_store_path, token_embedding=embeddings,
                          classifier_path=full_model_path)

        # -----------------------------------
        # ACTUAL STRUCTURE OF THE EMBEDDING
        # -----------------------------------
        return TextClassifier.load(intent_clf_model_path)
