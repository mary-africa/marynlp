import pytorch_lightning as pl
from .base_transformer import BaseTransformer


class KiDistillBERTi(BaseTransformer):
    def __init__(self):
        super(KiDistillBERTi, self).__init__()
