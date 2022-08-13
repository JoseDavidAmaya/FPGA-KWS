"""This script tests the model quantized with different values of 'm' and 'n' to check which format is the most appropiate to use.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import tensorflow as tf
import keras

from aconnect.layers import GRUCell_AConnect
from modules.quant import quantSigm, quantTanh
from modules.kws import dataset
from modules.kws import model
from modules.kws import qmodel

modelPath = "models/GRU_SC"
datasetConfig = model.GRU_SC_datasetConfig



# Load model
custom_objects = {"quantTanh": quantTanh, "quantSigm": quantSigm, "GRUCell_AConnect": GRUCell_AConnect}
model = keras.models.load_model(modelPath, custom_objects=custom_objects)

# Load dataset
splits, splits_length, in_shape, out_shape = dataset.get_dataset_speech_commands(datasetConfig)

_, _, test_ds = splits
_, _, ntest_ds = splits_length

test_ds = test_ds.batch(ntest_ds)

print("Base accuracy: ")
model.evaluate(test_ds)

print("Accuracy after quantization")
maxAcc = 0
mForMaxAcc = 0
for m in range(0, 8):
    n = 7-m
    modelQuant = qmodel.getFullQuantModel(model, m)
    modelQuant.run_eagerly = True

    print("Accuracy in Q{0}.{1} format:".format(m,n))
    _, acc = modelQuant.evaluate(test_ds)
    if acc > maxAcc:
        maxAcc = acc
        mForMaxAcc = m

print("Best accuracy ({0}) achieved quantizing in Q{1}.{2} format".format(maxAcc, mForMaxAcc, 7-mForMaxAcc))