 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sed.config import ModelsConfig
from sed.models import LanguageModel

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

class LMpipeline(object):
    def __init__(self, train_data, test_data):
        self.config = LMConfig

        self.train_loader = DataLoader(train_data, 
                                batch_size=self.config.BATCH_SIZE,
                                collate_fn=pad_collate,
                                shuffle=False,
                                num_workers=8)

        self.test_loader = DataLoader(test_data, 
                                batch_size=self.config.BATCH_SIZE,
                                collate_fn=pad_collate,
                                shuffle=False,
                                num_workers=8)
        
        self.model = LanguageModel(**self.config.model_params).to(self.config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.iter_meter = IterMeter()

    def train(self, epoch):
        self.model.train()
        data_len = len(self.train_loader.dataset)
        for batch_idx, _data in enumerate(self.train_loader):
            tokens, token_len, targets = _data 
            tokens, targets = [token.to(self.config.device) for token in tokens], torch.stack(targets).to(self.config.device)

            self.optimizer.zero_grad()

            output = self.model((tokens, token_len))
            loss = F.cross_entropy(output, targets.reshape(targets.shape[0],))
            loss.backward()

            self.optimizer.step()
            # scheduler.step()
            # self.iter_meter.step()
            if batch_idx % 1000 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(tokens), data_len,
                    100. * batch_idx / len(self.train_loader), loss.item()))
                
    def test(self, epoch, accum_loss):
        print('\nevaluating…')
        self.model.eval()
        test_loss = 0
        test_cer, test_wer = [], []
        with torch.no_grad():
            for I, _data in enumerate(self.test_loader):
                tokens, token_len, targets = _data 
                tokens, targets = [token.to(self.config.device) for token in tokens], torch.stack(targets).to(self.config.device)

                output = self.model((tokens, token_len))
                loss = F.cross_entropy(output, targets.reshape(targets.shape[0],))
                test_loss += loss.item() / len(self.test_loader)

        print('Test set: Average loss: {:.4f}\n'.format(test_loss))

        if test_loss < accum_loss:
            accum_loss = test_loss

        return test_loss
        
    def train_lm(self, early_stop=3):
        print(self.model)
        #train and evaluate model
        accum_loss = torch.tensor(float('inf'))
        stop_eps = 0 

        for epoch in range(1, self.config.EPOCHS + 1):
            self.train(epoch)             
            test_loss = self.test(epoch, accum_loss)

            if test_loss < accum_loss:
                accum_loss = test_loss
                stop_eps = 0
            else:
                stop_eps += 1

            if stop_eps >= early_stop:
                break
