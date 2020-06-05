from overrides import overrides
from typing import List, Union, Iterable, Iterator
from marynlp.data.transformers.specialized import DataTextTransformer, NERDataTransformer, POSDataTransformer

import re
import os
import logging
import json

logger = logging.getLogger(__name__)


class DataTextReader(object):
    def __init__(self, transformers: List[DataTextTransformer] = None):
        self.stacked_transforms = transformers

        if not (transformers is None or len(transformers) == 0):
            logger.info('Included transformers')
            for ic, transfn in enumerate(transformers):
                assert isinstance(transfn, DataTextTransformer), \
                    "Object in the transformer must implement the {}, instead got {}".format(
                        DataTextTransformer.__name__, type(transfn).__name__)

                logger.info('[Tf .{:03d}] {}'.format(ic + 1, str(transfn)))

    def read(self, file_path: Union[str, os.PathLike]):
        logger.info('Reading the file \'{}\''.format(file_path))

        # perform checks here and there

        # TODO: think of wrapping this in a _read, when implemented by someone
        with open(file_path, 'r', encoding='utf-8') as rfl:
            return [self.transform(line) for line in rfl.readlines()]

    def transform(self, text: str) -> str:
        transformed_text = text

        if not (self.stacked_transforms is None or len(self.stacked_transforms) == 0):
            for transfn in self.stacked_transforms:
                transformed_text = transfn(transformed_text)

        return transformed_text


# ----------------------------------
# For JSON Data
# ----------------------------------
class DataJsonReader(DataTextReader):
    @overrides
    def read(self, file_path: Union[str, os.PathLike]):
        logger.info('Reading the file \'{}\''.format(file_path))
        # reads the json file
        with open(file_path, 'r', encoding='utf-8') as jfl:
            data = json.load(jfl)

        return data


class CorpusJsonReader(DataJsonReader):
    """Special Json Reader for the corpus that we have
    'swwiki.json'
    """

    @overrides
    def read(self, file_path: Union[str, os.PathLike]):
        json_data = super().read(file_path=file_path)

        # take only the text and words counts
        text_wc_dict = self._dictify(json_data.values())

        # perform transformations on the text
        transformed = [self.transform(text) for text in text_wc_dict['texts']]

        return transformed

    def _dictify(self, texts_dicts: Union[list, Iterable]):
        main_dict = dict(texts=[], wordcounts=[])

        # collecting the text and number of text
        for _d in texts_dicts:
            # keys
            main_dict['texts'].append(_d['text'])
            main_dict['wordcounts'].append(int(_d['wordcount']))

        return main_dict


# ------------------------------------------
# For NER txt data
# -----------------------------------------

class NERDataTextReader(DataTextReader):
    def __init__(self,
                 ner_transformer: NERDataTransformer = None,
                 other_stack_transformers: List[DataTextTransformer] = None):

        super().__init__(other_stack_transformers)
        # NOTE: the other_transformer mustn't be a NerDataTransformer
        self.ner_transform = ner_transformer

    def read(self, file_path: str) -> Iterator:
        logger.info('Reading the file \'{}\''.format(file_path))

        # perform checks here and there

        # TODO: think of wrapping this in a _read, when implemented by someone
        with open(file_path, 'r', encoding='utf-8') as rfl:
            for line in rfl.readlines():
                if self.ner_transform is not None:
                    for word, tag in self.ner_transform(self.transform(line)):
                        yield word, tag
                    yield None
                else:
                    # Assumed that the page is already in the needed format
                    if line.strip():
                        word, tag = re.split(r'\s+', self.transform(line))
                        yield word, tag
                    else:
                        yield None

# ------------------------------------------
# For POS txt data
# -----------------------------------------


class POSDataTextReader(DataTextReader):
    def __init__(self,
                 pos_transformer: POSDataTransformer = None,
                 other_stack_transformers: List[DataTextTransformer] = None):

        super().__init__(other_stack_transformers)
        # NOTE: the other_transformer mustn't be a NerDataTransformer
        self.pos_transformer = pos_transformer

    def read(self, file_path: str) -> Iterator:
        logger.info('Reading the file \'{}\''.format(file_path))

        # perform checks here and there

        # TODO: think of wrapping this in a _read, when implemented by someone
        with open(file_path, 'r', encoding='utf-8') as rfl:
            for line in rfl.readlines():
                if self.pos_transformer is not None:
                    for word, tag in self.pos_transformer(line, lower=True):
                        yield word, tag
                    yield None
                else:
                    # Assumed that the page is already in the needed format
                    if line.strip():
                        word, tag = re.split(r'\s+', self.transform(line))
                        yield word, tag
                    else:
                        yield None
