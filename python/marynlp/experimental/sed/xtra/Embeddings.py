# -*- coding: utf-8 -*-

import torch
import json
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm

from sed.config import EmbeddingsConfig
from sed.tokenizer import Tokenizer

def load_embeddings(store_path):
    '''
    load embeddings from path on disk
    '''
    
    f = open(str(store_path), 'r')
    data = f.read()
    data = data.replace('array', 'np.array')
    return eval(data)

class Embeddings(nn.Module):
    def __init__(self, token_vocab_size, embedding_dim, hidden_dim, composition_fn='non', batched_input=False, tokenizer=None, embed_path=None, morph_path=None):
        super(Embeddings, self).__init__()
        
        self.embeddings = {}
        self.batched = batched_input
        self.comp_fn = composition_fn
        self.morph_path = morph_path
        
        if embed_path is not None:
            self.embeddings.update(load_embeddings(embed_path))
            token_vocab_size = len(self.embeddings.keys())
            
        self.emb_mod = nn.Embedding(token_vocab_size, embedding_dim)
        
        if self.comp_fn == 'rnn':
            self.compose = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.comp_linear = nn.Sequential(
                nn.Linear(hidden_dim*2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
                )

        else:
            self.compose = None

    def padded_sum(self, data, input_len = None, dim=0):
        """
        summing over padded sequence data
        Args:
            data: of dim (batch, seq_len, hidden_size)
            input_lens: Optional long tensor of dim (batch,) that represents the
                original lengths without padding. Tokens past these lengths will not
                be included in the sum.

        Returns:
            Tensor (batch, hidden_size)

        """
        if input_len is not None:
            # print(input_len)
            return torch.stack([
                torch.sum(data[:, :input_len, :], dim=dim)
            ])
        else:
            return torch.stack([torch.sum(data, dim=dim)])

    def pack_padded_seq(self, emb_in, in_len):
        """
        Helper function to remove padded components in sequence before passing to the RNN
        """

        in_len = [torch.as_tensor(len(emb_in[k])).long().reshape(1,) if in_len[k] is None else in_len[k] for k in range(len(in_len))]
        
        return [pack_padded_sequence(emb_in[k], in_len[k], batch_first=True, enforce_sorted=False) for k in range(len(emb_in))]

    def rnn_compose(self, emb_in, in_len):
        """
        RNN composition of morpheme vectors into word embeddings
        """
        if not self.batched:
            emb_in = [torch.stack(emb_in)]
        
        # emb_in = self.pack_padded_seq(emb_in, in_len)
        rnn_out = [self.compose(emb)[0] for emb in emb_in]
        return rnn_out#[pad_packed_sequence(comp, batch_first=True)[0] for comp in rnn_out]

    def get_composition(self, emb_in, in_len, dim, requires_grad):
        """
        Helper function to get word embeddings from morpheme vectors. Uses additive function by default
        if composition function is not specified
        """

        if self.compose is not None:
            self.compose.requires_grad, self.comp_linear.requires_grad = requires_grad,requires_grad

            if self.comp_fn == 'rnn':
                rnn_out = self.rnn_compose(emb_in, in_len)
                return torch.sum(torch.stack([torch.mean(self.comp_linear(comp[:, :l, :]), 1) for comp in rnn_out for l in in_len]), dim=0)

            return [self.compose(emb) for emb in emb_in] 

        return [self.padded_sum(emb_in[j], in_len[j], dim=dim) for j in range(len(emb_in))][0]

    def check_input(self, x_in, in_len):
        """
        Helper function to ensure inputs are tensors and in the appropriate device
        """
        if not all(torch.is_tensor(x) for x in x_in):
            x_in = [torch.as_tensor(x).to(EmbeddingsConfig.device).long() for x in x_in]

        if not all(torch.is_tensor(il) for il in in_len):
            in_len = [torch.as_tensor(il).to(EmbeddingsConfig.device).long().reshape(1,) if il is not None else il for il in in_len]
       
        return x_in, in_len

    def compose_embeddings(self, x_in, in_len, requires_grad, dim=1):
        """
        Get embeddings from morpheme vectors after passing label-encoded vectors through the embedding layer
        Args:
            x_in   - label-encoded vector inputs
            in_len - original lengths of vectors if were padded, else is None

        Returns:
            vector representation (embeddings) of the text sequence 
        """  
        x_in, in_len = self.check_input(x_in, in_len)

        if self.batched:
            emb_in = [torch.stack([self.emb_mod(x)]) for x in x_in]
            return self.get_composition(emb_in, in_len, dim, requires_grad)

        dim = 0
        emb_in = [self.emb_mod(torch.stack(x_in))]
    
        return self.get_composition(emb_in, in_len, dim, requires_grad)
            
    def check_batched(self, inputs, requires_grad):
        """
        check whether data is passed in batches(for models like the language and sentiment analysis)
        or as single inputs and get embeddings accordingly.

        Args:
            inputs - collection containing the label-encoded words as well as their corresponding original
                     lengths if were padded

        Returns:
            tensor of the vector representation of the input sequence
        """

        emb_in = []
        if self.batched:
            for x_in, in_len in zip(inputs[0], inputs[1]):
                emb_in.append(self.compose_embeddings(x_in, in_len, requires_grad))
                
            return torch.stack(emb_in) if self.compose is not None else torch.cat(emb_in, dim=0)

        x_in, in_len = inputs, [None for i in range(len(inputs))]

        return torch.as_tensor(self.compose_embeddings(x_in, in_len, requires_grad))

    def get_embeddings(self, inputs, return_array=True, requires_grad=True):
        """
        get vector representation of the input data

        Args:
            inputs - collection of the label-encoded words as well as their corresponding original lengths if padded

        Returns:
            vector representation of the input sequence
        """

        emb_in = self.check_batched(inputs, requires_grad)

        if return_array:
            return np.array(emb_in.clone().detach().cpu().tolist())

        return emb_in

    def embed(self, tokens, strings, store_path=None):
        """
        get embeddings and store dict of each word from corpus and its corresponding embedding vector

        Args:
            tokens     - collection of label-encoded inputs
            strings    - original strings corresponding to their encodings
            store_path - location on disk to save the embeddings

        """

        emb_dict = {string : self.get_embeddings(token) for token, string in tqdm(zip(tokens, strings))}
        self.embeddings.update(emb_dict)

        if store_path is not None:
            self.save_embeddings(emb_dict, store_path)
    
    def save_embeddings(self, embeddings, store_path):
        '''
        store embeddings to file on disk

        Args:
            embeddings - dictionary of embedding vectors
            store_path - location on disk to store morphemes
        '''
        with open(str(store_path), 'w') as file:
            file.write(repr(embeddings))
   
    def check_embeddings(self, embeddings_dict):
        """
        check for existing embeddings and if none are found checks whether embeddings were passed in as an agument
        """

        if not self.embeddings:
            assert embeddings_dict is not None, 'please pass in embeddings'

            return embeddings_dict

        return self.embeddings

    def get_word_embedding(self, string, embeddings=None):
        """
        get word embedding of given string. checks whether word exists in stored embeddings and 
        returns its vector otherwise breaks the word into its morphological units and acquires 
        an embedding vector using the compositional function

        Args:
            string - string whose vector representation is to be obtained

        Returns:
            Embedding vector of given string
        """
        if embeddings is None:
            embeddings = self.check_embeddings(embeddings)

        if string in list(embeddings.keys()):
            return embeddings[string]
        
        tokenizer = Tokenizer(self.morph_path)
        tokenizer.fit(list(self.embeddings.keys()))

        return self.get_embeddings(tokenizer.transform(string)[0], requires_grad=False)

    def get_most_similar(self, string, sim_dict, threshold):
        """
        get most similar word(s) from collection of related words using the cosine similarity measure

        Args:
            string    - string whose most siilar words are to be obtained
            sim_dict  - dictionary of similar words
            threshold - minimum cosine similarity value for words to be considered most similar to string
                        if None then only word with highest cosine similarity is returned
        
        Returns:
            collection of most similar words as determined by their cosine similarity to the string being considered
        """

        cos_sim = [sim[1] for sim in sim_dict[string]]
        max_sim = max(cos_sim)

        if threshold is not None:
            assert max_sim>=threshold, 'threshold set too high, no similar words found'

            return [v for v in sim_dict[string] if v[1]>=threshold]

        return [v for v in sim_dict[string] if v[1]==max_sim]


    def get_similar_words(self, string, embeddings_dict=None, threshold=None):
        """
        get collection of closely related words usnig the cosine similarity of their embedding vectors

        Args:
            string          - string whose related words are to be obtained
            embeddings_dict - dictionary of word embeddings. If embedder already trained uses existing embeddings.
            threshold       - minimum cosine similarity for word to be considered similar to given word

        Returns:
            dictionary of similar words and their similarity as measure by the cosine similarity between their embedding vectors
            and that of the string
        """

        embeddings = self.check_embeddings(embeddings_dict)
        val = self.get_word_embedding(string, embeddings, tokenizer)

        sim_dict = {}
        sim_dict[string] = [(txt, cosine_similarity(val.reshape(1,-1), vec.reshape(1,-1)).reshape(1)[0]) for txt,vec in embeddings.items() if txt!=string or not (vec==val).all()]

        most_similar = self.get_most_similar(string, sim_dict, threshold)
        sim_dict[string] = sorted(most_similar,key=lambda x: x[1], reverse=True)

        return sim_dict

    def get_best_analogy(self, sim_list, string_b, return_cos_similarity):
        """
        get most relevant analogy from collection of analogous words. uses cosine similarity measure to determine 
        the best analogy

        Args:
            sim_list - list of words similar to the given word
            string_b - word whose analogy is to be determined
            return_cosine_similarity - whether or not output should include the analogy's cosine similarity

        Returns:
            analogy of the given word
        """
        sorted_sim = sorted([sim for sim in sim_list if sim[1]>0], key=lambda x:x[1], reverse=True)
        max_sim = sorted([sim for sim in sim_list if sim[1]>0], key=lambda x:x[1], reverse=True)[0][0]

        if not return_cos_similarity:
            sorted_sim = [sim[0] for sim in sorted_sim]

        if max_sim == string_b:
            return sorted_sim[1]
        
        return sorted_sim[0]

    def _3_cos_add(self, a, _a, b, string_b, embeddings, return_cos_similarity):
        """
        determine the analogy of the given word based on an additive function of cosine similarities

        Args:
            a,_a     - vector representation of the example of a word and its corresponding analogy
            b        - vecor representation of the string whose analogy is to be determined
            string_b - string whose analogy is to be determined

        Returns:
            analogy of the string based on given example and determined using cosine similarity
        """
        _b = b - a + _a

        sim_list = [(txt, cosine_similarity(vec.reshape(1,-1),_b).reshape(1)[0]) for txt,vec in embeddings.items()]
  
        return self.get_best_analogy(sim_list, string_b, return_cos_similarity)

    def _3_cos_mul(self, a, _a, b, string_b, embeddings, return_cos_similarity, eps=0.001):
        """
        determine the analogy of the given word based on a multiplicative function of cosine similarities

        Args:
            a,_a     - vector representation of the example of a word and its corresponding analogy
            b        - vecor representation of the string whose analogy is to be determined
            string_b - string whose analogy is to be determined

        Returns:
            analogy of the string based on given example and determined using cosine similarity
        """

        sim_list = [(txt, (cosine_similarity(vec.reshape(1,-1),b).reshape(1)[0]*cosine_similarity(vec.reshape(1,-1),_a).reshape(1)[0])/(cosine_similarity(vec.reshape(1,-1),a).reshape(1)[0]+eps))\
                    for txt,vec in embeddings.items()]

        return self.get_best_analogy(sim_list, string_b, return_cos_similarity)

    def pair_direction(self, a, _a, b, string_b, embeddings, return_cos_similarity):
        """
        determine the analogy of the given word based on an additive function of cosine similarities that maintains
        the ...

        Args:
            a,_a     - vector representation of the example of a word and its corresponding analogy
            b        - vecor representation of the string whose analogy is to be determined
            string_b - string whose analogy is to be determined

        Returns:
            analogy of given string based on given example and determined using cosine similarity
        """
        _b = _a - a

        sim_list = [(txt, cosine_similarity(vec.reshape(1,-1)-b,_b).reshape(1)[0]) for txt,vec in embeddings.items()]

        return self.get_best_analogy(sim_list, string_b, return_cos_similarity)

    def get_analogy(self, string_a, analogy_a, string_b, embeddings_dict=None, return_cos_similarity=False):
        """
        get analogous words using 3COSADD, PAIRDIRECTION, or 3COSMUL which make use of the cosine similarity of the embedding vectors.        
        adapted from: https://www.aclweb.org/anthology/W14-1618

        Args:
            string_a, analogy_a - example of a string and its analogy
            string_b - string whose analogy is to be determined
            embeddings_dict - dictionary of embeddings. uses existing embeddings if was pretrained
            return_cosine_similarity - whether or not output should include the analogy's cosine similarity
        
        Returns:
            analogy of given string based on given example and determined using cosine similarity
        """
        embeddings = self.check_embeddings(embeddings_dict)
        a, _a, b = (self.get_word_embedding(string, embeddings).reshape(1,-1) for string in [string_a, analogy_a, string_b])
        
        return self._3_cos_add(a, _a, b, string_b, embeddings, return_cos_similarity) 
        