
import torch
from sed.datasets import pad_collate

class DataConfig():
    dataloader_params = dict(        
    batch_size=256,
    collate_fn= lambda x: pad_collate(x, pretrained=False),
    shuffle=False,
    num_workers=0
    )

class ModelsConfig():

    # setting random seed
    torch.manual_seed(7)

    # checks if gpu is available
    use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda:0" if use_cuda else "cpu")

    learning_rate = 1e-4
    context_size = 3 #for the Language Model
    
    # Configurations
    # model_params = dict(
    # embedding_dim = 300,
    # rnn_dim = 256,
    # hidden_dim = 256,
    # composition_fn = 'rnn',
    # batched_input= DataConfig.dataloader_params['batch_size']>1
    # )

    EPOCHS = 100

class EmbeddingsConfig():
    # setting random seed
    torch.manual_seed(7)

    # checks if gpu is available
    use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    batched_input = DataConfig.dataloader_params['batch_size']>1

    # Configurations
    embedder_params = dict(
    embedding_dim = 512,
    hidden_dim = 512,
    dropout=0.2,
    num_attn_layers=6, 
    d_ff=2048,
    hidden=8,
    composition_fn = 'rnn'
    )
    