import tensorflow as tf
import numpy as np
from keras import backend

def q(x, integerBits):
  numbits = 8
  m = 2**integerBits
  return tf.quantization.fake_quant_with_min_max_vars(x, -m, m-(m/(2**(numbits-1))), num_bits=numbits)

def qInteger(var, integerBits):
  frac_bits = 7-integerBits
  quantVar = q(var, integerBits)
  quantVarInt = (quantVar*(2**frac_bits)).numpy().astype(np.int8)
  return quantVarInt

# Activation functions
def quantTanh(A):
  return backend.maximum(backend.minimum(A, 1.0), -1.0)

def quantSigm(A):
  A = (A + 1.0) / 2.0
  return backend.maximum(backend.minimum(A, 1.0), 0.0)