import os
from pathlib import Path
from typing import Union

import logging
logger = logging.getLogger('marynlp')

__doc__ = """
This package is to store all file or folder 
related utility functions and variables
"""

# TODO: find a way to change the storage directory
#  for windows (C:/Users/<your-username>/.marynlp/store) or unix (~/.marynlp/store)
_STORAGE_PATH = str(Path.home().joinpath('./.marynlp/store'))


def storage_path(path: Union[str, Path], absolute: bool = False, make_dir: bool=True):
    """Building the path to store the contents like models, training data

        Args:
            path (`Union[str, Path]`): ,
            absolute (`bool`): ,
            make_dir (`bool`): ,
        Note:
            At the moment it is assumed to be WIN system
    """

    root_store_path = Path(_STORAGE_PATH)
    root_store_path = root_store_path.joinpath(f'./{path}')

    if absolute:
        root_store_path = Path(f'./{path}')

    if not root_store_path.exists():
        if make_dir:
            logger.info('Directory \'{}\' doesn\'t exist. Creating the directory'.format(root_store_path))
            # make the directory if it doesnt exist
            root_store_path.mkdir(parents=True, exist_ok=True)

    return str(root_store_path)


def save_to_file(text_list: list, file_path: str):
    with open(file_path, "w", encoding='utf-8') as fw:
        fw.write(str(text_list))

    logger.info(f'Saved to: {file_path}')
