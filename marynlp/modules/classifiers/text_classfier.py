from typing import Union, List
import numpy as np
import lightgbm as lgb

from pathlib import Path
from overrides import overrides

from marynlp.text_encoders.label_encoder import LabelEncoder
from marynlp.data.transformers.vectorized import TextTfidfVectorizer

from marynlp.modules.module import Module

_classifier = {
    'early-tfidf-lgb-afhs': 'models/early-tfidf-lgb-afhs.zip'
}

# Building the classe for Sentimental Analyis
class _TextClassifier(Module):
    """Make prediction of what class a text belongs to."""

    
    def classify(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Predicts one of multiple labels"""
        raise NotImplementedError()
        
    def classify_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Predicts probabilities of the classes the text(s) belong"""
        raise NotImplementedError()
    
    def get_classes(self) -> List[str]:
        """Outputs of the classes in the order with 
            which the prediction is made
        """
        raise NotImplementedError()


class EmotionSentAnalyClassifier(_TextClassifier):
    def __init__(self, 
                 classes: List[str],
                 models: List[lgb.Booster], 
                 word_vector_pkl_path: str, 
                 char_vector_pkl_path: str):
        
        super().__init__()     
        self.classes = classes # ['angry', 'fear', 'happy', 'sad']
        self.models = models
        self.transformer = TextTfidfVectorizer(word_vector_pkl_path, char_vector_pkl_path)
        self.le = LabelEncoder(self.classes)
        
    @classmethod
    # @overrides
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **kwargs):
        folder_path = cls.get_full_model_path(src, credentials_json_path=credentials_json_path, model_option='')

        _classes = ['angry', 'fear', 'happy', 'sad']
        _models = [lgb.Booster(model_file=str(folder_path.joinpath(f'{class_name}_lgb.txt'))) for class_name in _classes] 
        
        word_vectorizer_path = folder_path.joinpath("word_vectorizer.pickle")
        char_vectorizer_path = folder_path.joinpath("char_vectorizer.pickle")

        assert word_vectorizer_path.exists(), "Word vectorizer pickle file doesn't exist ('%s')" % (str(word_vectorizer_path))        
        assert word_vectorizer_path.exists(), "Char vectorizer pickle file doesn't exist ('%s')" % (str(char_vectorizer_path))

        return cls(classes=_classes, 
                   models=_models, 
                   word_vector_pkl_path=str(word_vectorizer_path),
                   char_vector_pkl_path=str(char_vectorizer_path))
        
    @overrides
    def classify(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        write code to load models from wherever and however you'll save them and store them in a dict called models
        """
        if isinstance(texts, str):
            return self._mono_classify(texts)
        
        return np.array([self._mono_classify(text) for text in texts])

    def _mono_classify(self, text: str) -> np.ndarray:
        return np.argmax(self._mono_classify_proba(text))
    
    @overrides
    def classify_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self._mono_classify_proba(texts)
        
        return np.array([self._mono_classify_proba(text) for text in texts])
    
    def _mono_classify_proba(self, text: str):
        vector = self.transformer.transform(text)
        preds = []
        
        for model in self.models:
            preds.append(model.predict(vector, num_iteration=model.best_iteration)[0])

        return np.array(preds)
    
    @overrides
    def get_classes(self):
        print(self.classes)
