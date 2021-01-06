from typing import Union
from pathlib import Path

def build_model_full_path(flair_model_folder: Union[str, Path], model_version_file_pt: str):
    """This constructs the model path according to how it works with flair created models"""
    flair_model_folder = Path(flair_model_folder)

    return flair_model_folder.joinpath(model_version_file_pt)
