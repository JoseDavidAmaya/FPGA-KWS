"""This script trains a KWS model with 1 GRU layer
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import tensorflow as tf
import keras

from modules.kws import dataset
from modules.kws import model
from modules.quant import quantSigm, quantTanh

datasetConfig = model.GRU_L_datasetConfig
modelSettings = model.GRU_L_settings
modelName = "GRU_L"
WBstd = 0.0



# Load dataset
splits, splits_length, in_shape, out_shape = dataset.get_dataset_speech_commands(datasetConfig)

train_ds, val_ds, test_ds = splits
ntrain_ds, nval_cs, ntest_ds = splits_length

batch_size = 100
#train_ds = train_ds.shuffle(ntrain_ds).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#val_ds = val_ds.batch(nval_cs).prefetch(tf.data.AUTOTUNE).cache()
val_ds = val_ds.batch(nval_cs).prefetch(tf.data.AUTOTUNE)

# Prepare model
model = model.createGRUModel(modelSettings, in_shape, out_shape, [quantTanh, quantSigm], [WBstd, WBstd])

model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    metrics=["accuracy"],
)

# Train
EPOCHS = 30

def scheduler(epoch, lr): # Speech commands
  if epoch < (10):
    return 5e-4
  elif epoch < (20):
    return 1e-4
  else:
    return 2e-5

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,

    callbacks=[
      tf.keras.callbacks.LearningRateScheduler(scheduler),
    ]
)

model.save("models/" + modelName)

model.evaluate(test_ds.batch(ntest_ds))