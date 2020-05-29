from overrides import overrides
from typing import List


class DataTextTransformer(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def extra_repr(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """This calls the transform()"""
        return self.transform(*args, **kwargs)

    def __repr__(self):
        return f'{type(self).__name__}(self.extra_repr())'


class StackedDataTextTransformer(DataTextTransformer):
    def __init__(self, transformers: List[DataTextTransformer]):
        assert len(transformers) > 0, 'You must include atleast one DataTextTransformer'

        for ix, trn in enumerate(transformers):
            assert isinstance(trn, DataTextTransformer) \
                , 'Expected transformer at index {} to be \'{}\', but got \'{}\'' \
                .format(ix, DataTextTransformer.__name__, type(trn).__name__)

        self.transformers = transformers

    @overrides
    def transform(self, text: str):
        t_text = text
        for trn in self.transformers:
            t_text = trn(t_text)

        return t_text