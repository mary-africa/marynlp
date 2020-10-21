import re
import pickle
import numpy as np
import string
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from marynlp.data.transformers import DataTextTransformer


class TextTfidfVectorizer(DataTextTransformer):
    """
    TODO: 
    """
    def __init__(self, word_vector_pkl_path: str, char_vector_pkl_path: str):
        self.word_vectorizer = pickle.load(open(word_vector_pkl_path, "rb"))
        self.char_vectorizer = pickle.load(open(char_vector_pkl_path, "rb"))

    def count_regexp_occ(self, regexp="", text=None):
        """ Simple way to get the number of occurence of a regex"""
        return len(re.findall(regexp, text))
    
    def prepare_for_char_n_gram(self, text: str):
        """ Simple text clean up process"""
        # 1. Go to lower case (only good for english)
        # Go to bytes_strings as I had issues removing all \n in r""
        clean = bytes(text.lower(), encoding="utf-8")
        # 2. Drop \n and  \t
        clean = clean.replace(b"\n", b" ")
        clean = clean.replace(b"\t", b" ")
        clean = clean.replace(b"\b", b" ")
        clean = clean.replace(b"\r", b" ")
        # 4. Drop puntuation
        # I could have used regex package with regex.sub(b"\p{P}", " ")
        exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
        clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
        # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
        clean = re.sub(b"\d+", b" ", clean)
        # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
        clean = re.sub(b'\s+', b' ', clean)
        # Remove ending space if any
        clean = re.sub(b'\s+$', b'', clean)
        # 7. Now replace words by words surrounded by # signs
        # e.g. my name is bond would become #my# #name# #is# #bond#
        # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
        clean = re.sub(b" ", b"# #", clean)  # Replace space
        clean = b"#" + clean + b"#"  # add leading and trailing # 
        return str(clean, 'utf-8')
    
    def get_features(self, text: str):
        """
        Check all sorts of content and extract features such as number of words and chars in 
        a sentence
        """
        word_len = len(text.split())
        char_len = len(text)
        num_upper = self.count_regexp_occ(r"[A-Z]", text)
#         txt_lower = text.lower()
        txt_lower = self.prepare_for_char_n_gram(text)
        clean_chars = len(set(txt_lower))
        char_ratio = len(set(txt_lower)) / (1 + min(99, len(txt_lower)))
        
        return csr_matrix(np.array([word_len, char_len, num_upper, clean_chars, char_ratio]))
                          
    def vectorize(self, text: str):                                                    
        return self.char_vectorizer.transform([text]), self.word_vectorizer.transform([text])
                          
    def transform(self, text: str):     
        char_vector, word_vector = self.vectorize(text)
        return hstack([char_vector, word_vector, self.get_features(text)]).tocsr()


__all__ = ['TextTfidfVectorizer']
