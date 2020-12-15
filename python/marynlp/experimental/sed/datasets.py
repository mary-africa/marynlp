
import torch
import string
from typing import List, Tuple

from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import Dataset, DataLoader

def pad_collate(batch, pretrained=False):
    if not pretrained:
        x, y1, y2, x_len = [], [], [], []
        for x_batch, y1_batch, y2_batch in batch:
            xx, yy, yz = [torch.from_numpy(x) for x in x_batch], y1_batch, y2_batch
            x_len.append([torch.as_tensor(len(x)).long().reshape(1,) for x in xx])
            
            xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).long()

            x.append(xx_pad)
            y1.append(torch.as_tensor(yy).long().reshape(1,))
            y2.append(torch.as_tensor(yz).long().reshape(1,))

        return x, x_len, torch.stack(y1), torch.stack(y2)  

    x, y1, y2 = [], [], []
    for x_batch, y1_batch, y2_batch in batch:
        xx, yy, yz = torch.from_numpy(x_batch).float(), y1_batch, y2_batch
        x_len = None
        
        x.append(xx)
        y1.append(torch.as_tensor(yy).long().reshape(1,))
        y2.append(torch.as_tensor(yz).long().reshape(1,))

    return torch.stack(x), x_len, torch.stack(y1), torch.stack(y2)  

class SAMDataset(Dataset):
    def __init__(self, data, emotion_col, sentiment_col, tokens=None):
        self.data = data
        self.emotion_col = emotion_col
        self.sentiment_col = sentiment_col
        self.tokens = tokens

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx, :]
        emotion, sentiment = sample[self.emotion_col], sample[self.sentiment_col]

        token = self.tokens[idx]
        return token, emotion, sentiment

# class LMDataset(Dataset):
#     def __init__(self, text, tokens: list, config, target_encoder=None):
#         self.config = config        
#         if isinstance(text, str):
#             text = text.split(' ') 
            
#         self.encoder = None

#         if target_encoder is not None:
#             self.encoder = target_encoder

#         self.tokens = [[tokens[i+(j-1)] for j in range(self.config.context_size)] for i in range(1,len(tokens) - self.config.context_size)]
#         self.ngrams = [text[i + self.config.context_size] for i in range(len(text) - self.config.context_size)]
        
#     def __len__(self):
        
#         return len(self.tokens)-self.config.context_size
    
#     def __getitem__(self, idx):
        
#         if self.encoder is not None:
#             return self.tokens[idx], self.encoder.encode(self.ngrams[idx])
        
#         return self.tokens[idx]
