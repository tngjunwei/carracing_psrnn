import tensorflow as tf
import numpy as np

class PSRNNCell(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units, params, reuse=None):
    super(PSRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._params = params

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    
    weights_initializer = tf.constant(self._params.W_FE_F.T.astype(np.float32))
    weights = tf.get_variable(
        "weights",
        initializer=weights_initializer)
    
    bias_initializer = tf.constant(self._params.b_FE_F.T.astype(np.float32))
    biases = tf.get_variable(
        "bias",
        initializer=bias_initializer)

    # print("State: ", state.shape) #(20,20)
    # print("weights: ", weights.shape) #(20,400)
    # print("biases: ", biases.shape) #(1,400)
    # print("inputs: ", inputs.shape) #(20,20)

    W = tf.add(tf.matmul(state, weights), biases)
    batchedW = tf.split(W, W.shape[0], 0) # List of 20 (1,1225) vectors
    batchedInputs = tf.split(inputs, inputs.shape[0], 0) # List of 20 (1,) vectors
    batched_output = []
    for W, input in zip(batchedW, batchedInputs):
      W_square = tf.transpose(tf.reshape(W, [self._num_units, self._num_units])) # W_square: (35,35)
      new_s = tf.matmul(input,W_square)  # (1,1) x (35,35)
      new_s_normalized = new_s/tf.norm(new_s)
      batched_output.append(new_s_normalized)
    output = tf.concat(batched_output,0)
    return output, output






