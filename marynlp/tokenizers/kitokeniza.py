from .tokenizer import BaseTokenizer
from tokenizers import ByteLevelBPETokenizer
from typing import List

import logging
logger = logging.getLogger(__name__)

tokenizer_opts = dict(add_prefic_space=True)
tokenizer_train_opts = dict(vocab_size=30000,
                            min_frequency=2,
                            show_progress=True)


class KiTokeniza(ByteLevelBPETokenizer, BaseTokenizer):
    vocab_file = 'kitokeniza-vocab.json'
    merge_file = 'kitokeniza-merge.txt'

    def __init__(self, *bpetokenizer_opts):
        super().__init__(**bpetokenizer_opts, **tokenizer_opts)

    @classmethod
    def from_pretrained(cls, storage_dir: str):
        vocab_file = f"{storage_dir}/{cls.vocab_file}"
        merge_file = f"{storage_dir}/{cls.merge_file}"

        return cls(vocab_file, merge_file)

    def train(self, data_files: List[str], *args, **kwargs):
        self.train(data_files, **tokenizer_train_opts)


