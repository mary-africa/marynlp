 
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from sed.config import DataConfig, ModelsConfig

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

class Pipeline(object):
    def __init__(self, train_data, test_data, model, save_path=None):

        self.train_loader = DataLoader(train_data, **DataConfig.dataloader_params)

        self.test_loader = DataLoader(test_data, **DataConfig.dataloader_params)
        
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=ModelsConfig.learning_rate)
        self.iter_meter = IterMeter()
        self.save_path = save_path

    def train(self, epoch):
        self.model.train()
        data_len = len(self.train_loader.dataset)
        for batch_idx, _data in enumerate(self.train_loader):
            x, x_len, y1, y2 = _data 
            x, y1, y2 = [xx.to(ModelsConfig.device) for xx in x], y1.to(ModelsConfig.device), y2.to(ModelsConfig.device)
            y = [y.reshape(y.shape[0],) for y in [y1,y2]]

            self.optimizer.zero_grad()

            output = self.model((x, x_len))
            loss = torch.sum(torch.stack([F.cross_entropy(out, y) for out,y in zip(output, y)]))
            loss.backward()

            self.optimizer.step()
            if batch_idx % 10 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), data_len,
                    100. * batch_idx / len(self.train_loader), loss.item()))
                
    def test(self, epoch, accum_loss):
        print('\nevaluating…')
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for I, _data in enumerate(self.test_loader):
                x, x_len, y1, y2 = _data 
                x, y1, y2 = [xx.to(ModelsConfig.device) for xx in x], y1.to(ModelsConfig.device), y2.to(ModelsConfig.device)
                y = [y.reshape(y.shape[0],) for y in [y1,y2]]

                output = self.model((x, x_len))
                loss = torch.sum(torch.stack([F.cross_entropy(out, y) for out,y in zip(output, y)]))
                test_loss += loss.item() / len(self.test_loader)

        print('Test set: Average loss: {:.4f}\n'.format(test_loss))

        if test_loss < accum_loss:
            accum_loss = test_loss

        return test_loss, self.model.state_dict()
        
    def train_model(self, early_stop=3):
        # print(self.model)
        #train and evaluate model
        accum_loss = torch.tensor(float('inf'))
        stop_eps = 0 
        weights = []
        for epoch in range(1, ModelsConfig.EPOCHS + 1):
            self.train(epoch)             
            test_loss, w8 = self.test(epoch, accum_loss)

            if test_loss < accum_loss:
                weights.append(w8)
                accum_loss = test_loss
                stop_eps = 0
            else:
                stop_eps += 1

            if stop_eps >= early_stop:

                if self.save_path is not None:
                    torch.save(weights[-1], Path(self.save_path).joinpath(f'{round(accum_loss, 4)}_SAM.pth'))

                break
