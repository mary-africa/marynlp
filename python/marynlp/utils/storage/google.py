from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage

from pathlib import Path

import logging
logger = logging.getLogger('marynlp')

STORAGE_BUCKET = 'marynlp-private'

def cache_from_google_bucket(online_path: str, save_path: str, credentials_json_path: str, bucket_name: str = STORAGE_BUCKET, overwrite: bool = True, **bucket_blob_args):
    
    # Gets the google client associated with the service account
    # NOTE: a valid service account needs to have the marynlp-private storage
    google_client = None
    
    try:
        google_client = storage.Client.from_service_account_json(str(credentials_json_path))
    except DefaultCredentialsError:
        logger.error('Unable to get client from credentials path')
        raise
    
    save_path = Path(save_path)
    
    if save_path.exists():
        logger.info('Data already exists in \'%s\'' % (str(save_path)))
        
        # if overwrite is false
        if not overwrite:
            logger.info('')
            return False
        
        # ...
        # overrite is true
        logger.info('Overwriting contents')
        
    bucket = google_client.get_bucket(bucket_name)
    
    blob = bucket.blob(online_path, **bucket_blob_args)
    blob.download_to_filename(str(save_path))
    
    logger.info('Contents in [%s:%s] have been cached to \'%s\'' % (bucket_name, str(online_path), str(save_path)))
