# Import the models used to deal with local
#  file storage
from .local import storage_path, save_to_file

# To deal with imports from the cloud storage
from .gcp.gc_cache import cache_from_google_bucket
