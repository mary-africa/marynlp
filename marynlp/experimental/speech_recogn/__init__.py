from .model import SpeechRecognitionModel


class Config(object):
    @staticmethod
    def model_params(self, num_class: int):
        return dict(
            n_cnn_layers=3,
            n_rnn_layers=5,
            rnn_dim=512,
            n_class=num_class,  # the plus 1 is for the blank index
            n_feats=128
        )
