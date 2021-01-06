from abc import abstractmethod
from pathlib import Path
from . import _PRETRAINED_MODELS

from zipfile import ZipFile

from ..utils.storage import local as util_local
from ..utils.storage.google import cache_from_google_bucket

import os
import shutil

import logging
logger = logging.getLogger('marynlp')

def get_from_remote_store(remote_model_path: str, credentials_json_path: str, temporary: bool=True, store_path: str = None):
    """Downloads the model from remote source (in GCP bucket) and 
        stores as temporary waiting to be extracted

        Returns:
            The path of the zip file after being downloaded. If `temporary` is true, returns the path as located 
                in the machines temp location, otherwise, it returns the zip as stored in the `store_path + remote_model_path`
    """    

    if temporary:
        store_path = remote_model_path
        store_path = util_local.get_temp_path(store_path)


    # Cache the contents from GCP to locally chosen storage storage
    cache_from_google_bucket(remote_model_path, 
                            store_path,
                            credentials_json_path,
                            overwrite=True)

    # Returns the path
    return str(store_path)


def extract_zipped_model(zip_model_path: str, to_save_path: str):
    """Extract the zip file `zip_model_path` and stores it at `to_save_path`"""
    try:
        from zipfile import ZipFile

        with ZipFile(zip_model_path, mode='r') as zpf:
            zpf.extractall(path=to_save_path)
    except :
        logger.info('Unable to extract the path: %s' % (zip_model_path))
        return False

    return True


class Module(object): 
    @classmethod
    def pretrained_categories(cls):
        """Get the category that belongs to the pretrained model"""
        raise NotImplementedError()

    @classmethod
    def prepare_pretrained_model_path(cls, src: str, credentials_json_path: str = None, overwrite=True):
        # check if the entered category exist
        model_categ = cls.pretrained_categories()
        assert model_categ in _PRETRAINED_MODELS, "The pretrained category '%s' doesn't exist" % (model_categ)

        # check if the model exists from the prepared categories
        if src in _PRETRAINED_MODELS[model_categ]:
            # check if the credentials path exist
            assert credentials_json_path, 'If using pre-trained model, you need to include the credentials key file'
            assert Path(credentials_json_path).exists(), "Credentials key file doesn't exist"

            # it exists begin downloading the model
            get_reference_remote_zip_file = _PRETRAINED_MODELS[model_categ][src]

            # constuct the destination folder
            get_reference_remote_zip_file_name = ".".join(get_reference_remote_zip_file.split(".")[:-1])    # remove the '.zip'
            dest_folder_path = util_local.get_storage_path(get_reference_remote_zip_file_name)
            dest_folder_path = Path(dest_folder_path)

            if dest_folder_path.exists():
                if not overwrite:
                    # returns the folder
                    return str(dest_folder_path)
                else:
                    # delete the folder:
                    shutil.rmtree(str(dest_folder_path))
                    logger.info("The model at '%s' exist. Overwriting" % (dest_folder_path))
                    

            # get the zip path for the model as downloaded from GCP
            zip_path = get_from_remote_store(get_reference_remote_zip_file, credentials_json_path=credentials_json_path, temporary=True)

            logger.info("Unpacking the model in '%s'" % get_reference_remote_zip_file)
            extract_zipped_model(zip_path, str(dest_folder_path))
            logger.info('Unpacking completed')

            # remove zip version
            os.remove(zip_path)

            return str(dest_folder_path)           
            
        else:
            # it doesnt exist, load from local storage
            src = Path(src)
            assert src.exists(), "The path '%s' entered doesn't exist" % (str(src))

            return str(src)

    @classmethod
    @abstractmethod
    def from_pretrained(self, src: str, credentials_json_path: str, **model_kwargs):
        """Method for loading the pretrained model. These pretrained models are either taken
            locally, or from GCP Bucket that are referenced from a `self.pretrained_categories`"""
        raise NotImplementedError()
        
