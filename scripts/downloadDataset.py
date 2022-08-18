"""This script downloads the google speech commands dataset, which takes up to 30 minutes to download and prepare depending on network connection and processor.
If this script is not run first, then the download will be performed when running any other script that uses the dataset, this script is just to have the dataset ready before running the other scripts.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

from modules.kws import dataset
dataset.get_dataset_speech_commands({})
print("Download finished")