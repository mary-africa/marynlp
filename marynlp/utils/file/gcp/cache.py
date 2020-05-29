import os
import re
import requests

from contextlib import AbstractContextManager
# TODO: Find to add the tqdm to cache processing


class Cache(AbstractContextManager):
    """Context manager object for a cached content
    default => 'wb'
    """

    def __init__(self, file_name, method='wb'):
        """
        # TODO: NOTICE MEEE
        # TODO: Update Documentation
        Args:
            file_name: This is the file name from the **cached** path
            method: ...
        """
        # TODO: FIX THE TO main_path PROPERTY
        self.main_path = file_name
        self.file_obj = open(file_name, method)

    def __enter__(self):
        return self.file_obj

    def __exit__(self, type, value, traceback):
        """TODO: Fix the lazy way of handling exception"""
        print("Closed cached file")
        self.file_obj.close()
        return True

    def __repr__(self):
        """
        Should set the cached path object
        Returns:
        """
        raise NotImplementedError()


def _validate_url(url_string: str) -> bool:
    """Validates if a string is a valid URL
    # TODO: This might not belong here
    """

    # Regular expression for URL [RFC 1808]
    rURL = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.match(rURL, url_string) is not None


def cache_url(url_path: str, content_type: str = None):
    """Caches the contents of a single url"""

    if not _validate_url(url_path):
        raise RuntimeError(f'{url_path} is not a valid URL')

    rc = requests.get(url_path)

    # Checks if the content type are the same when set
    if content_type is not None:
        if rc.headers['content-type'] != content_type:
            raise RuntimeError(f'Request content-type mismatch. Target content-type = {content_type},'
                               f' instead got {rc.headers["content-type"]}')

    # extract data and store
    rc.content()

    #
    #
    #
    raise NotImplementedError()


def cache_bytes(bytes_to_cache: bytes, cache_path: str, overwrite: bool = False):
    """Stores the bytes in a defined cache directory"""

    # Check if there is something of the same cache_path already stored
    if not overwrite:
        if os.path.exists(cache_path):
            raise FileExistsError(f'The cache path \'{cache_path}\' already exists')

    with Cache(cache_path, 'wb') as cb:
        cb.write(bytes_to_cache)