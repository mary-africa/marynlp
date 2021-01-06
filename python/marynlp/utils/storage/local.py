import os
import tempfile
from pathlib import Path
from typing import Union

import logging
logger = logging.getLogger('marynlp')

__STORAGE_HOME_NAME = './.marynlp/store'
storage_path: Path = str(Path.home().joinpath(__STORAGE_HOME_NAME))

def get_storage_path(path: Union[str, os.PathLike]):
    store_path = Path(storage_path).joinpath(f'./{path}')

    return str(store_path)

def get_temp_path(path: Union[str, os.PathLike]):
    """Get the temporary path."""
    temp_path = Path(tempfile.TemporaryDirectory().name)
    temp_path = temp_path.joinpath(f'./{path}')
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    return str(temp_path)
