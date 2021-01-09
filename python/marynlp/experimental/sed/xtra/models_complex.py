
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.optim import AdamW
from pytorch_lightning.metrics.functional.classification import f1_score

from sed.embeddings import Embeddings
from sed.config import ModelsConfig

class LanguageModel(Embeddings):
    def __init__(self, token_vocab_size, word_vocab_size, embedding_dim, composition_fn='non', batched_input=True, tokenizer=None):
        super(LanguageModel, self).__init__(token_vocab_size, embedding_dim, composition_fn='non', batched_input=True, tokenizer=None)
        
        self.birnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim*2, 128), 
            nn.ReLU(),
            nn.Linear(128, word_vocab_size)
            )
      
    def forward(self, inputs):
      
        emb_in = self.get_embeddings(inputs, return_array=False).transpose(0,1)
        out,_ = self.birnn(emb_in)
        
        return torch.mean(self.linear(out.transpose(0,1)), 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y.reshape(y.shape[0],))
        return loss

class SentimentAnalysisModel(Embeddings):
    def __init__(self, num_emotions, num_sentiments, embedding_dim, composition_fn, token_vocab_size=0, pretrained=False):
        super(SentimentAnalysisModel, self).__init__(token_vocab_size, embedding_dim, composition_fn, batched_input=True)
        
        self.config = ModelsConfig
        self.metric = f1_score
        self.pretrained = pretrained
        self.num_classes = num_emotions

        if not self.pretrained:
            assert token_vocab_size>0, 'vocab size cannot be 0'

        self.birnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim*2, 128), 
            nn.ReLU()
            )

        self.classifier1 = nn.Linear(128, num_emotions)
        self.classifier2 = nn.Linear(128, num_sentiments)
      
    def forward(self, inputs):
        
        if not self.pretrained:
            emb_in = self.get_embeddings((inputs[0], inputs[1]), return_array=False)
        else:
            emb_in = inputs[0]

        x_packed = pack_padded_sequence(*self.check_input(emb_in, inputs[2]), batch_first=True, enforce_sorted=False)
        output_packed,_ = self.birnn(x_packed)
        
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
        output = self.linear(output_padded)
        
        return torch.mean(self.classifier1(output), 1), torch.mean(self.classifier2(output), 1)

    def training_step(self, batch, batch_idx):
        x, xx_len, x_len, y1, y2 = batch
        
        y = [y.reshape(y.shape[0],) for y in [y1,y2]]

        output = self((x, xx_len, x_len))
        loss = torch.sum(torch.stack([F.cross_entropy(out, y) for out,y in zip(output, y)]))

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss' : loss, 'preds' : output[0], 'target' : y[0]}

    def training_step_end(self, outputs):
        #update and log
        metric = self.metric(F.log_softmax(outputs['preds'], dim=1), outputs['target'], self.num_classes)
        self.log('metric', metric)

    def validation_step(self, batch, batch_idx):
        x, xx_len, x_len, y1, y2 = batch
        
        y = [y.reshape(y.shape[0],) for y in [y1,y2]]

        output = self((x, xx_len, x_len))
        loss = torch.sum(torch.stack([F.cross_entropy(out, y) for out,y in zip(output, y)]))

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss' : loss, 'preds' : output[0], 'target' : y[0]}

    def validation_step_end(self, outputs):
        #update and log
        metric = self.metric(F.log_softmax(outputs['preds'], dim=1), outputs['target'], self.num_classes)
        self.log('metric', metric)

    def test_step(self, batch, batch_idx):
        x, xx_len, x_len, y1, y2 = batch
        
        y = [y.reshape(y.shape[0],) for y in [y1,y2]]

        output = self((x, xx_len, x_len))
        loss = torch.sum(torch.stack([F.cross_entropy(out, y) for out,y in zip(output, y)]))

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.config.learning_rate)