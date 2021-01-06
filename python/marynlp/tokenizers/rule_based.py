import re
from ..data.reference_vocabulary import LowercaseSwahiliReferenceVocabulary

UNK_LABEL = 'unk'

class RuleBasedTokenizer(object):
    def __init__(self):
        self.srv = LowercaseSwahiliReferenceVocabulary()

        # dont change the order here [num must start]
        self._map = {
            'num': self.srv.regex_for_numbers,
            'abc': r'[%s]+' % (self.srv.regex_word),
            'sym': r'[%s]+' % (self.srv.regex_non_word)
        }
        
        self._t2i = None
    
    @property
    def token_labels(self):
        return list(self._map.keys())

    @property
    def label2ix(self):
        if self._t2i is None:
            self._t2i = dict(zip(self.token_labels, range(len(self.token_labels))))
            
        return self._t2i
    
    @property
    def all_token_labels(self):
        return self.token_labels + [UNK_LABEL]
    
    def _word_token(self, word: str):
        for label, lbl_regex in self._map.items():
            if re.match(r'^(%s)$' % (lbl_regex), word.lower()):
                return self.label2ix[label]
            
        return -1
        
    def tokenize(self, text: str):
        txl = re.split("\s+", text)
        ix_ls = [self._word_token(word) for word in txl]
                
        return list(zip(txl, ix_ls))