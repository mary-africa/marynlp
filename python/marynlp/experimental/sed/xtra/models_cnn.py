
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.optim import AdamW

from sed.embeddings import Embeddings
from sed.config import EmbeddingsConfig

class SentimentAnalysisModel(Embeddings):
    def __init__(self, token_vocab_size, num_emotions, num_sentiments, rnn_dim):
        super(SentimentAnalysisModel, self).__init__(token_vocab_size, **EmbeddingsConfig.embedder_params)
        
        self.num_classes = num_emotions
        self.embedding_dim = EmbeddingsConfig.embedder_params['embedding_dim']

        self.birnn = nn.GRU(input_size=int(self.embedding_dim/2), hidden_size=rnn_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.cnn = nn.Conv2d(1, 64, 3, 2, padding=3//2)
        self.linear = nn.Sequential(
            nn.Linear(rnn_dim*2, 256), 
            nn.ReLU(),
            nn.Dropout(0.4)
            )

        self.classifier1 = nn.Linear(256, num_emotions)
        self.classifier2 = nn.Linear(256, num_sentiments)
      
    def forward(self, inputs):
        
        emb_in = self.get_embeddings(inputs, return_array=False, return_sequence=True)
        out = self.cnn(emb_in)
        # out = F.avg_pool2d(out, kernel_size=2)
        sizes = out.size()
        out = out.view(sizes[0], sizes[1] * sizes[2], sizes[3]) 
        out,_ = self.birnn(out)
        out = self.linear(out)
                
        return torch.mean(self.classifier1(out), 1), torch.mean(self.classifier2(out), 1)

# class LanguageModel(Embeddings):
#     def __init__(self, token_vocab_size, word_vocab_size, embedding_dim, composition_fn='non', batched_input=True, tokenizer=None):
#         super(LanguageModel, self).__init__(token_vocab_size, embedding_dim, composition_fn='non', batched_input=True, tokenizer=None)
        
#         # self.birnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True, bidirectional=True)
#         self.cnn = nn.Conv2d(1, 64, 3, 2, padding=3//2)
#         self.linear = nn.Sequential(
#             nn.Linear(embedding_dim*2, 128), 
#             nn.ReLU(),
#             nn.Linear(128, word_vocab_size)
#             )
      
#     def forward(self, inputs):
      
#         emb_in = self.get_embeddings(inputs, return_array=True).transpose(0,1)
#         out = self.cnn(emb_in)
#         print(out.shape)
#         assert False
#         return torch.mean(self.linear(out.transpose(0,1)), 1)
