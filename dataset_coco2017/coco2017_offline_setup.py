
import os
import sys
import io
import collections
import urllib.parse
import urllib.request
import tempfile


# Script to install the COCO 2017 dataset offline in tensorflow-datasets.

# For the operation of this script, approximately 75 GB of storage are required:
#  - 25 GB for the downloaded files (can be manually deleted after installation)
#  - 25 GB for the extracted files (will be deleted automatically after installation)
#  - 25 GB for the installed dataset (permanently occupied)
#
# The default directories are subdirectories in the directory of this script:
#   "./dlcache" as the directory of the downloaded files (can be changed in the code)
#   "./extract_tmp_XXXXXXX" as the temporary extraction directory (can be modified using the "dir" parameter of "TemporaryDirectory()")
#   "./tf_datasets" as the root directory of installed datasets (can be defined using "TFDS_DATA_DIR")


# The environment variable "TFDS_DATA_DIR" controls where tensorflow-datasets'
# dataset data is stored. (Must be defined before importing "tensorflow_datasets".)
os.environ["TFDS_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_datasets")
import tensorflow_datasets as tfds
      
      
class UrlretrieveProgressIndicator():
  def __init__(self):
    self.last_percentage = None

  def __call__(self, block_num, block_size, total_size):
    curr_percentage = round( (100.0 * block_num * block_size) / total_size )
    
    if curr_percentage != self.last_percentage:
      print(f"\rProgress: {curr_percentage}%", end="")
      sys.stdout.flush()
      self.last_percentage = curr_percentage


def download_coco2017(dlcache_dir):
      
  # Files of the COCO 2017 dataset.
  # Can be downloaded automatically by this script, or downloaded manually
  # and placed into the "./dlcache" subdirectory.
  urls = [
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/image_info_test2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip"]

  # Download files (if needed, i.e., if not provided manually)
  if not os.path.exists(dlcache_dir):
    os.makedirs(dlcache_dir)
  
  for url in urls:
    url_parsed = urllib.parse.urlparse(url)
    file_local = os.path.basename(url_parsed.path)
  
    file_local_full = os.path.join(dlcache_dir, file_local)
    
    if os.path.exists(file_local_full):
      print(f"Already downloaded: '{url}'.")
    else:
      print(f"Downloading '{url}'...")
      urllib.request.urlretrieve(url, file_local_full, UrlretrieveProgressIndicator())
      print("\nDone!")

      
def install_coco2017(dlcache_dir):

  with tempfile.TemporaryDirectory(prefix = "extract_tmp_", 
    dir = os.path.dirname(os.path.abspath(__file__))) as temp_extract_dir:

    download_config = tfds.download.DownloadConfig(
      download_mode = tfds.GenerateMode.REUSE_DATASET_IF_EXISTS,
      manual_dir    = dlcache_dir,
      extract_dir   = temp_extract_dir
      )
      
    builder = tfds.builder("coco/2017")
    
    builder.download_and_prepare(
      download_config = download_config,
      download_dir    = None
      )
      
    print("Done installing dataset.")
    print("Deleting temporary extraction directory, please wait...")
    
  print("Installation completed!")

  
def main():

  try:
    ds_train      = tfds.load("coco/2017", split="train",      download = False)
    ds_validation = tfds.load("coco/2017", split="validation", download = False)
    ds_test       = tfds.load("coco/2017", split="test",       download = False)

    print("COCO 2017 dataset is already installed:")
    print(f"  #samples (train):      {ds_train.cardinality().numpy()}")
    print(f"  #samples (validation): {ds_validation.cardinality().numpy()}")
    print(f"  #samples (test):       {ds_test.cardinality().numpy()}")

  except AssertionError:
    
    # When tfds.load("...", download = False) fails with an AssertionError,
    # it usually means that the dataset is not installed yet. Install it now.
    
    print("COCO 2017 dataset is not installed yet, installing it now...")
    
    dlcache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlcache")

    download_coco2017(dlcache_dir)
    install_coco2017(dlcache_dir)
    

if __name__ == "__main__":
  main()
