"""GRU model for KWS (Based on https://github.com/ARM-software/ML-KWS-for-MCU)
"""

import keras
from aconnect.layers import GRUCell_AConnect

GRU_L_datasetConfig = {"features_type": "logmel", "feature_standardization": [True, True], "num_features": 40}
GRU_SC_datasetConfig = {"features_type": "logmel", "feature_standardization": [True, True], "num_features": 24, "stride_s": 40e-3}

GRU_L_settings = [400]
GRU_SC_settings = [160] # Multiple of 8

def createGRUModel(modelSettings, input_shape, output_shape, activationFunctions=["tanh", "sigmoid"], AConnectSettings=[0.0, 0.0]):
    GRU_units, = modelSettings
    activation, recurrent_activation = activationFunctions
    Wstd, Bstd = AConnectSettings
    model = keras.models.Sequential(
          [
            keras.layers.Input(shape=input_shape), # define input shape so we can unroll it
            keras.layers.RNN(GRUCell_AConnect(GRU_units, reset_after=False, activation=activation, recurrent_activation=recurrent_activation, implementation=2, Wstd=Wstd, Bstd=Bstd, pool=2, isBin=False), unroll=True),
            keras.layers.Dense(output_shape, use_bias=True),
          ]
        )
    
    return model