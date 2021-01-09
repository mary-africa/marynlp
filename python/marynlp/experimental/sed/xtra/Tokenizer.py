#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
from sed.xtra.Heuristic import MorphologyAnalyzer

class LabelEncoder:
    def __init__(self, items: list, sorted = False):
        self.items = items if not sorted else sorted(items)

    def encode(self, item):
        return float(self.items.index(item))

    def __call__(self, input):
        return self.encode(input)

    def decode(self, ix):
        return self.items[int(ix)]
    
class Tokenizer(MorphologyAnalyzer):
    '''
    Tokenizer for Swahili adapted from the SED Heuristic Morphology analyzer:
    https://www.researchgate.net/publication/228566510_Refining_the_SED_heuristic_for_morpheme_discovery_Another_look_at_Swahili
    '''
    def __init__(self,  morph_path=None):
        super().__init__()  
        self.encoder = LabelEncoder
        
        if morph_path is not None:
            self.morphemes.update(self.load_morphs(morph_path))

    def break_word(self, word, stems):
        '''
        find initial break of words by locating the stem, if exists in the collection of stems from analyzer, and
        breaking the word off at these points

        Args:
            word - string to be tokenized
            stems - collection of stems extrated using the morphology analyzer

        Returns:
            list containing the stem as the initial break and a list of substrings containing the rest of the character
            sequences from the word
        '''
        breaks = []
        morphs = None
        
        for stem in stems:
            #partition the word at the stem and include stem as first morpheme
            if stem in word:
                breaks.append(stem)
                init_break = word.partition(stem)
                morphs = [br for br in init_break if stem not in br and br not in '']
                break

        return breaks, morphs
        
    def get_morphemes(self, word, stems):
        '''
        break words down to stems and associated morphemes. searches for stem as initial break of word
        and uses the collection of other morphemes extracted using the morphology analyzer to search and
        find other possible breaks in the word. 

        Args:
            word - string to be tokenized
            stems - collection of stems extrated using the morphology analyzer

        Returns:
            list of morphemes composing the word 
        '''
        
        breaks, morphs = self.break_word(word, stems)
        
        if breaks:
            #get remaining partitions based on morphemes associated with the stem found in the word
            for morp in self.morphemes[breaks[0]]['morphemes']+[pre+vow for pre in self.noun_prefix for vow in self.vowels if len(pre)>1 and pre[1] not in self.vowels]:
                if len(morp)==2 and morp[0] not in self.vowels or len(morp)==3 and morp[:2] in self.noun_prefix:
                    
                    for i,br in enumerate(morphs):
                        brk = br.partition(morp)
                        interm_break = [ib for ib in brk if ib not in '' and ib not in [mor for mor in self.morphemes[breaks[0]]['morphemes'] if len(mor)>2] and len(ib)<len(''.join(brk)) or ib in self.vowels and (word.find(morp)==0 or word.rfind(morp)==-1)]
                        
                        if interm_break:
                            breaks.append(max(interm_break))
                            morphs[i] = min(interm_break)
                            
            breaks.extend(np.setdiff1d(morphs,breaks).tolist())
            tokens = list(set(breaks))
            
            return tokens

        elif not breaks and len(word)>3:
            subs = sorted(list(set([morp for stem in self.morphemes.keys() for morp in self.morphemes[stem]['morphemes']+[pre+vow for pre in self.noun_prefix for vow in self.vowels if len(pre)>1 and pre[1] not in self.vowels] if len(morp)==2 and morp[0] not in self.vowels or len(morp)==3 and morp[:2] in self.noun_prefix])), key=len, reverse=True)
            morphs = [s for s in subs if s in word]
            
            if morphs:
                breaks = [gp for gp in re.sub(r'|'.join(map(re.escape, morphs)), ' ', word).split(' ') if gp not in '' and len(gp)>1 or (len(gp)==1 and gp in self.vowels and word.find(gp)==0 or word.rfind(gp)==len(word)-1)]
                
            return breaks+morphs
        
        return breaks     
    
    def break_noun(self, word, stems):
        '''
        Uses predefined noun classes to find initial breaks in nouns or noun-like words and proceeds to further break the words
        down to basic morphemes

        Args:
            word  - string to be tokenized
            stems - list of stems obtained from data using the morphology analyzer

        Returns:
            list containing the corresponding noun class prefix and a list of leftover subwords
        '''

        #search for prefix denoting the noun class
        for pre in sorted(self.noun_prefix+[np+vow for np in self.noun_prefix for vow in self.vowels if np[-1] not in self.vowels], key=len, reverse=True):
            if word.find(pre)==0:
                breaks = [pre]
                break
        
        #any subword longer than for characters likely contains several
        #morphemes and is therefore further broken down
        if any(len(part)>4 for part in word.partition(breaks[0])):
            init_break = []

            for part in word.partition(breaks[0]):
                if len(part)>4:
                    morp = self.get_morphemes(part, stems)

                    if morp and len(''.join(morp))>=len(part):
                        init_break.extend(morp)
                        
                    elif morp and len(''.join(morp))<len(part):
                        init_break.extend(part.partition(min(morp)))
                        
                    else:
                        init_break.append(part)

                else:
                    init_break.extend(part)
        else:
            init_break = word.partition(breaks[0])
            
        return init_break, breaks
    
    def get_breaks(self, word, stems):
        '''
        experimental quick fix for nouns. since nouns have less morphemes and forms than verbs, the algorithm is
        less effective at analyzing them.
        '''        
        init_break, breaks = self.break_noun(word, stems)
        morphs = []
        
        for br in init_break:
            if br.find('ni')==(len(br)-2): 
                interm_break = list(br.partition('ni'))
                morphs += [br for br in interm_break if br not in '' and len(br)>2 or (len(br)==1 and br in self.vowels and (word.find(br)==0 or word.rfind(br)==len(word)-1)) or (len(br)==2 and br[0] not in self.vowels)]
            else:
                interm_break = []
                morphs += [br for br in init_break if br.find(breaks[0])!=0 and br not in '' and len(br)>2 or (len(br)==1 and br in self.vowels and word.find(br)==0 or word.rfind(br)==len(word)-1) or (len(br)==2 and br[0] not in self.vowels)]+interm_break

        return list(set(breaks+morphs))
                
    def align_tokens(self, word, tokens):
        '''
        list morphemes according to their order of appearance in the original word. This is essential for sequence based
        models and embedding algorithms based on composition functions such as CNN or LSTMs
        '''
        tok_idx = {c: [m.start() for m in re.finditer(c, word)] for c in tokens}
        idx_tup = sorted([tup for tup in [(k,idx) for k,v in tok_idx.items() for idx in v]], key= lambda x: x[1])
        tokens = [str(tok[0]) for i,tok in enumerate(idx_tup) if len(tok[0])>1 or len(tok[0])==1 and i==0 or len(tok[0])==1 and tok[0] not in idx_tup[i-1][0]]                

        return tokens
    
    def read_from_path(self, text_path):
        '''
        read text from path on disk
        '''
        with open(text_path) as file_object:
            text = [l for line in file_object for l in line.rstrip('\n').split(' ') if l not in '']
            
        return text
    
    def fit(self, corpus = None, corpus_path = None, morph_path=None):
        '''
        Function to tokenize words based on morphemes collected from morphology analysis. 
        Uses the extracted morphemes as well as predefined rules to break words according to the 
        swahili syntax.

        Args:
            corpus      - text data to be tokeninzed
            corpus_path - path to text data on disk
            morph_path  - path on disk to morphemes extracted using the morphology analyzer 

        Returns:
            dictionary of each unique word in corpus and its corresponding list of morphemes
        '''
            
        if not self.morphemes:
            print('warning:','no morpheme templates found, please load template from path or extract templates from data for optimal tokenization')
        
        if corpus_path is not None:
            corpus = self.read_from_path(corpus_path)
            
        if isinstance(corpus, str):
            corpus = corpus.split(' ')       
            
        corpus = [corp for corp in corpus if corp not in '']

        stems = sorted(list(self.morphemes.keys()), key=len, reverse=True)
        break_dict = {}
        
        for word in corpus:
            tokens = None
            
            if not any([word.find(pre)==0 for pre in self.noun_prefix]):#ensure the word is not a noun
                tokens = self.get_morphemes(word, stems)
                
                if not tokens:
                    break_dict[word] = word

            else:
                tokens = self.get_breaks(word, stems)
            
            if tokens is not None and tokens:
                break_dict[word] = self.align_tokens(word, tokens)
                
        return break_dict
    
    def initiate_encoder(self, break_dict):
        '''
        fit label encoder on training data
        '''
        tokens = []
        
        for v in break_dict.values():
            breaks = [v]
            tokens.extend(self.get_tokens(breaks))
            
        self.encoder = self.encoder(list(set(tokens)))
        
    def fit_transform(self, corpus = None, corpus_path = None):
        '''
        get morphemes and corresponding label encodings

        Returns:
            list of arrays with integer representations of the tokens from the corpus
        '''
        if corpus_path is not None:
            corpus = self.read_from_path(corpus_path)
            
        assert corpus is not None, 'please pass in corpus or path to corpus'
        
        if isinstance(corpus, str):
            corpus = corpus.split(' ') 
                 
        self.encoder = LabelEncoder
        
        break_dict = self.fit(corpus=list(set(corpus)))
        self.initiate_encoder(break_dict)
        
        return self.get_encodings(corpus)
        
    def get_tokens(self, breaks):
        '''
        extract every token from list of list
        '''
        if isinstance(breaks[0], str):
            tokens = breaks
            
        else:
            tokens = [tok for br in breaks for tok in br]
        
        return tokens
        
    def encode(self, string):   
        '''
        label encoding of individual string
        '''
        break_dict = self.fit(string)
        breaks = list(break_dict.values())
        tokens = self.get_tokens(breaks)
        
        return np.stack([np.array(self.encoder.encode(tok)) for tok in tokens])   
    
    def get_encodings(self, strings):
        ohe_dict = {}
        
        for string in list(set(strings)):
            if string not in '':                
                ohe_dict[string] = self.encode(string)

        return [ohe_dict[string]  for string in strings]
    
    def transform(self, strings = None, string_path = None, break_dict = None):
        '''
        label encoding of strings to be passed to the model
        '''
        if string_path is not None:
            strings = self.read_from_path(string_path)
            
        assert strings is not None, 'please pass strings or path to strings to be encoded'
        
        if isinstance(strings, str):
            strings = strings.split(' ')  
                    
        if isinstance(self.encoder, LabelEncoder):
            return self.get_encodings(strings)
        
        else:
            assert break_dict is not None, 'tokenizer must be fit on data before transforming'
            self.initiate_encoder(break_dict)
            
            return self.get_encodings(strings)
            
            
    def inverse_transform(self, arr):
        '''
        retrieve string from label-encoded tokens
        '''
        assert isinstance(self.encoder, LabelEncoder), 'encoder not initiated'
        
        return ''.join([self.encoder.decode(arr[i]) for i in range(len(arr))])