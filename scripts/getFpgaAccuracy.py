"""This script performs inference on the FPGA using the test set of the speech commands dataset and reports the accuracy and time taken
"""

import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import tensorflow as tf
import numpy as np

import time

from modules.kws import model
from modules.kws import dataset
from modules.quant import qInteger
from modules.utils import pad
from modules import fpgaInference as fpga

datasetConfig = model.GRU_L_datasetConfig
integerBits = 3 # Value of m in Qm.n, found with the script findQmn.py
memwidth = 64 # Should be a power of 2 and a multiple of 8



# Load dataset
splits, splits_length, in_shape, out_shape = dataset.get_dataset_speech_commands(datasetConfig)

_, _, test_ds = splits
_, _, ntest_ds = splits_length

# Save all the quantized input data in a numpy array for fast access
specs = list(test_ds.map(lambda x, y: x).as_numpy_iterator())
specs = [qInteger(x, integerBits) for x in specs]
elemwidth = 8
m = int(memwidth/elemwidth)
specs = [pad(x, m, supressOutput=True)[:in_shape[0]] for x in specs] # Pad input based on the memwidth
specs = np.array(specs)
ground_truth = np.array(list(test_ds.map(lambda x, y: y).as_numpy_iterator()))

print(f"Number of examples: {ntest_ds}")

# Run inference
print("Started inference")
startTime_ns = time.time_ns()
fpga_predictions = np.array([fpga.inference(x) for x in specs])
timeDiff_ns = time.time_ns() - startTime_ns

# Calculate accuracy
accuracy = (100*np.sum(ground_truth == fpga_predictions)/fpga_predictions.shape)[0]
print(f"Accuracy: {accuracy}%")
print(f"Time taken: {timeDiff_ns*1e-9}s")