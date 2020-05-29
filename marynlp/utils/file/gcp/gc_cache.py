import os

from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage

from .cache import Cache

# Google cloud bucket name
GCLOUD_DEFAULT_BUCKET_NAME: str = 'marynlp-private'


def _get_gcloud_client(credentials_json_path: str):
    # global GCLOUD_CLIENT
    """Setts"""
    # Initialize and connect to the gcloud client
    try:
        # Create path
        return storage.Client.from_service_account_json(credentials_json_path)
    except DefaultCredentialsError:
        print("Make sure you have run the `gcloud_cred_connect.sh` script in the main project folder"
              ". Run the script and try again")
        raise


def cache_from_google_bucket(bucket_content_path: str,
                             to_store_path: str,
                             credentials_json_path: str,
                             bucket_name: str = GCLOUD_DEFAULT_BUCKET_NAME,
                             encryption_key: bytes = None):
    """Cache contents from the bucket"""

    google_client = _get_gcloud_client(credentials_json_path)

    if not os.path.exists(to_store_path):
        bucket = google_client.get_bucket(bucket_name)

        blob = bucket.blob(bucket_content_path,
                               encryption_key=encryption_key)

        # Stores the file to the cache file
        blob.download_to_filename(to_store_path)

        # with Cache(to_store_path) as cb:
        #     blob.download_to_file(cb)

        print(f"Cached contents from '{bucket_content_path}' to '{to_store_path}'")
    else:
        print('The path -> {}, already exists'.format(to_store_path))