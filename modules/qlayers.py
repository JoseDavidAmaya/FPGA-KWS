import tensorflow as tf

import collections
import functools
import warnings

import numpy as np
import keras
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.saving.saved_model import layer_serialization
from keras.utils import control_flow_util
from keras.utils import generic_utils
from keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

# GRU cell with fake_quant after every W*x + b operation (or any vector-vector operation)

from keras.layers.recurrent import _caching_device
from keras.layers.recurrent import _config_for_enable_caching_device

from modules.quant import q

class GRUCell_FakeQuant(keras.layers.recurrent.GRUCell):

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=False,

               # FakeQuant
               integerBits=32, # Ideally there would be a different maxVal for different operations, however tests using a single maxVal showed great accuracy
               # --

               **kwargs):

    super(GRUCell_FakeQuant, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        reset_after=reset_after,
        **kwargs)
    
    self.integerBits = integerBits

  def call(self, inputs, states, training=None): # Using implementation = 2
    h_tm1 = states[0] if tf.nest.is_nested(states) else states  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = q(self.bias, self.integerBits), None
      else:
        input_bias, recurrent_bias = tf.unstack(q(self.bias, self.integerBits))

    kernel = q(self.kernel, self.integerBits)
    recurrent_kernel = q(self.recurrent_kernel, self.integerBits)

    if 0. < self.dropout < 1.:
      inputs = inputs * dp_mask[0]

    # inputs projected by all gate matrices at once
    matrix_x = q(backend.dot(inputs, kernel), self.integerBits)
    if self.use_bias:
      # biases: bias_z_i, bias_r_i, bias_h_i
      matrix_x = q(backend.bias_add(matrix_x, input_bias), self.integerBits)

    x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

    x_z = x_z
    x_r = x_r
    x_h = x_h

    if self.reset_after:
      # hidden state projected by all gate matrices at once
      matrix_inner = q(backend.dot(h_tm1, recurrent_kernel), self.integerBits)
      if self.use_bias:
        matrix_inner = q(backend.bias_add(matrix_inner, recurrent_bias), self.integerBits)
    else:
      # hidden state projected separately for update/reset and new
      matrix_inner = q(backend.dot(
          h_tm1, recurrent_kernel[:, :2 * self.units]), self.integerBits)

    recurrent_z, recurrent_r, recurrent_h = tf.split(
        matrix_inner, [self.units, self.units, -1], axis=-1)

    recurrent_z = recurrent_z
    recurrent_r = recurrent_r
    recurrent_h = recurrent_h

    z = self.recurrent_activation(q(x_z + recurrent_z, self.integerBits))
    r = self.recurrent_activation(q(x_r + recurrent_r, self.integerBits))

    z = q(z, self.integerBits)
    r = q(r, self.integerBits)

    if self.reset_after:
      recurrent_h = q(r * recurrent_h, self.integerBits)
    else:
      recurrent_h = q(backend.dot(
          q(r * h_tm1, self.integerBits), recurrent_kernel[:, 2 * self.units:]), self.integerBits)

    hh = self.activation(q(x_h + recurrent_h, self.integerBits))

    hh = q(hh, self.integerBits)
    # previous and candidate state mixed by update gate
    h = q(z * h_tm1, self.integerBits) + q(q(1 - z, self.integerBits) * hh, self.integerBits)

    h = q(h, self.integerBits)

    new_state = [h] if tf.nest.is_nested(states) else h
    return h, new_state

  def get_config(self):
    config = {
        # FakeQuant
        'integerBits':
            self.integerBits,
        # --
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(GRUCell_FakeQuant, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))