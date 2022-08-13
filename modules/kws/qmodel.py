"""Quantized version of the GRU model
"""
import tensorflow as tf

from modules.quant import q, quantSigm, quantTanh
from modules.qlayers import GRUCell_FakeQuant

def getFullQuantModel(model, integerBits): # Assumes quantTanh and quantSigm as activation functions
  gruconfig = model.layers[0].cell.get_config()
  gruconfig.pop("activation")
  gruconfig.pop("recurrent_activation")

  try:
    gruconfig.pop("Wstd")
    gruconfig.pop("Bstd")
    gruconfig.pop("errDistr")
    gruconfig.pop("pool")
    gruconfig.pop("isBin")
    gruconfig.pop("d_type")
  except:
    pass

  model_fakeQuant = tf.keras.Sequential([
                                        tf.keras.layers.Input(shape=model.input.shape[1:]),
                                        tf.keras.layers.Lambda(lambda x: q(x, integerBits)),
                                        tf.keras.layers.RNN(GRUCell_FakeQuant(integerBits=integerBits, **gruconfig, activation=quantTanh, recurrent_activation=quantSigm)), # , activation=quantTanh, recurrent_activation=quantSigm
                                        tf.keras.layers.Dense(**model.layers[1].get_config()),
                                        tf.keras.layers.Lambda(lambda x: q(x, integerBits)), # Quantizes the output of the dense layer so we don't have to edit the layer to quantize its activations
                                                                                        # Note that this means we apply the quantization as such: q(Wx + b) and not q(q(Wx) + b), this results in some differences
  ])

  model_fakeQuant.compile( 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        metrics=["accuracy"],
    )
  
  model_fakeQuant.set_weights(model.get_weights())

  for weight in model_fakeQuant.weights: # Quantize parameters, this is necessary for layers that don't have a quantized version
    weight.assign(q(weight.numpy(), integerBits)) # With forced ammount of int_bits
  
  return model_fakeQuant