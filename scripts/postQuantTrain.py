"""This script perform post quantization training on a model and save the model with the updated weights
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import tensorflow as tf
import keras

from aconnect.layers import GRUCell_AConnect
from modules.quant import q, quantSigm, quantTanh
from modules.kws import dataset
from modules.kws import model
from modules.kws import qmodel


modelPath = "models/GRU_L"
datasetConfig = model.GRU_L_datasetConfig
integerBits = 3 # Value of m in Qm.n, found with the script findQmn.py



# Load model
custom_objects = {"quantTanh": quantTanh, "quantSigm": quantSigm, "GRUCell_AConnect": GRUCell_AConnect}
model = keras.models.load_model(modelPath, custom_objects=custom_objects)

# Load dataset
splits, splits_length, in_shape, out_shape = dataset.get_dataset_speech_commands(datasetConfig)

train_ds, val_ds, test_ds = splits
ntrain_ds, nval_cs, ntest_ds = splits_length

batch_size = 100
#train_ds = train_ds.shuffle(ntrain_ds).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#val_ds = val_ds.batch(nval_cs).prefetch(tf.data.AUTOTUNE).cache()
val_ds = val_ds.batch(nval_cs).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(ntest_ds)


# Quantized model
modelQuant = qmodel.getFullQuantModel(model, integerBits)

print("Accuracy of model after quantization")
modelQuant.evaluate(test_ds)

# Training

def scheduler(epoch, lr):
  if epoch < (3):
    return 5e-4
  elif epoch < (6):
    return 1e-4
  else:
    return 2e-5

modelQuant.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    metrics=["accuracy"],
)

modelQuant.fit(
    train_ds,
    validation_data=val_ds,
    epochs=9,
    callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)]
)

for w in modelQuant.weights:
    w.assign(q(w, integerBits))

print("Accuracy of model after post quantization training")
modelQuant.evaluate(test_ds)

# Copy quantized weights to the original model for saving
model.set_weights(modelQuant.get_weights)

model.save(modelPath+"_PQT")