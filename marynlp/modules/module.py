from marynlp.utils.file import storage_path, cache_from_google_bucket
import itertools as it
import shutil
import os

from pathlib import Path
from zipfile import ZipFile


def model_path_fn(support_type: str, model_type: str, model: str):
    return '{}/{}/{}'.format(support_type, model_type, model)


LANG_CODE = 'sw'
_MARYNLP_BUCKET = 'marynlp-private'

support_type, dict_model = ((
                                'flair',  # When using flair trained models
                                'hf'  # When using HuggingFace trained models
                            ),
                            {
                                'classifier': ('wpc-small'),
                                'language-model': ('sw_lm-base'),
                                'taggers': ('sw-ner-gen1f-base')
                            })

# this are used
pretrained_models_class = {}
for stp in support_type:
    for mcl, model in dict_model.items():
        if mcl not in pretrained_models_class:
            pretrained_models_class[mcl] = {}

        # models classes
        pretrained_models_class[mcl]['{}-{}'.format(stp, model)] = model_path_fn(stp, mcl, model)


class Module(object):
    @property
    def pretrained_models(self):
        raise NotImplementedError()

    @classmethod
    def _get_pretrained_model_path(cls, src: str, credentials_json_path):
        assert src in cls.pretrained_models, "Model isn't in the pretrained models set"

        pt_model_path = cls.pretrained_models[src]

        # check if the model exists locally
        local_model_path = storage_path(pt_model_path, make_dir=False)

        if not Path(local_model_path).exists():  # Empty directory

            # make directory
            Path(local_model_path).mkdir(parents=True, exist_ok=True)

            # Cache the model data
            model_zip = f'{pt_model_path}.zip'
            local_model_zip = f'{local_model_path}.zip'

            # temporarily storage
            temp_local_model_path = storage_path(f'tmp/{pt_model_path}')
            temp_local_model_zip = f'{temp_local_model_path}.zip'

            print("Doesn't seem to exist in the local store ({}).\n".format(local_model_path) + \
                  "Downloading from [{}:{}]".format(_MARYNLP_BUCKET, model_zip))

            # fetch from google bucket
            cache_from_google_bucket(model_zip,
                                     to_store_path=temp_local_model_zip,
                                     credentials_json_path=credentials_json_path,
                                     bucket_name=_MARYNLP_BUCKET)

            print('Unzipping: {}'.format(local_model_zip))

            # Unzip the file
            with ZipFile(temp_local_model_zip, 'r') as zpf:
                zpf.extractall(path=Path(local_model_path).parent)

            print('-' * 50)
            print('Deleting temp files')
            print('-' * 50)
            print('Deleting \'{}\''.format(temp_local_model_path))
            shutil.rmtree(temp_local_model_path)
            print('Deleting \'{}\''.format(temp_local_model_zip))
            os.remove(temp_local_model_zip)
            print('-' * 50)

        return local_model_path

    @classmethod
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **module_kwargs):
        raise NotImplementedError()
