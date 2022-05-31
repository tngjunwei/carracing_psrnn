"""Example / benchmark for building a PTB PSRNN model for character prediction.

To run:

$ python ptb_word_lm.py --data_path=data/

python c:/Users/Carlton/Dropbox/psrnn_tensorflow/psrnn_code/ptb_psrnn_random/ptb_word_lm.py --data_path=c:/Users/Carlton/Dropbox/psrnn_tensorflow/git/psrnn/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

# tf.compat.v1.enable_eager_execution()

import reader
import psrnn_cell_impl
import two_stage_regression
import os
# from vae import ConvVAE

# vae = ConvVAE()
# vae.load_json()

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")

FLAGS = flags.FLAGS
losses = []

g = tf.Graph()

class Config(object):
  # Two Stage Regression Parameters
  nRFF_Obs = 1000 #2000
  nRFF_P = 1000
  nRFF_F = 1000
  dim_Obs = 80 #20
  dim_P = 80 #20
  dim_F = 80 #20
  reg_rate = 1*10**-3
  obs_window = 40
  kernel_width_Obs = 2 #2
  kernel_width_P = 0.2 #0.2
  kernel_width_F = 0.2 #0.2
    
  # BPTT parameters
  init_scale = 0.0
  learning_rate = 0.1
  max_grad_norm = 0.25
  num_layers = 2
  num_steps = 20  #20
  max_epoch = 10
  keep_prob = 1.0
  lr_decay = 1.0
  batch_size = 50 #20
  vocab_size = 8  #49
  seed = 0
  hidden_size = dim_Obs


# input shape = (batch_size, num_steps, 64, 64, 3)
def sequence_loss(logits,
                targets,
                weights,
                average_across_timesteps=True,
                average_across_batch=True,
                softmax_loss_function=None,
                name=None):
  """Weighted cross-entropy loss for a sequence of logits.

  Depending on the values of `average_across_timesteps` and
  `average_across_batch`, the return Tensor will have rank 0, 1, or 2 as these
  arguments reduce the cross-entropy at each target, which has shape
  `[batch_size, sequence_length]`, over their respective dimensions. For
  example, if `average_across_timesteps` is `True` and `average_across_batch`
  is `False`, then the return Tensor will have shape `[batch_size]`.

  Args:
    logits: A Tensor of shape
      `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
      int. The target represents the true class at each timestep.
    weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
      float. `weights` constitutes the weighting of each prediction in the
      sequence. When using `weights` as masking, set all valid timesteps to 1
      and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide the cost by the total label weight across timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A float Tensor of rank 0, 1, or 2 depending on the
    `average_across_timesteps` and `average_across_batch` arguments. By default,
    it has rank 0 (scalar) and is the weighted average cross-entropy
    (log-perplexity) per symbol.

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """

# if len(logits.get_shape()) != 3:
#   raise ValueError("Logits must be a "
#                    "[batch_size x sequence_length x logits] tensor")
# if len(targets.get_shape()) != 2:
#   raise ValueError("Targets must be a [batch_size x sequence_length] "
#                    "tensor")
# if len(weights.get_shape()) != 2:
#   raise ValueError("Weights must be a [batch_size x sequence_length] "
#                    "tensor")

  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    # num_classes = array_ops.shape(logits)[2]
    # logits_flat = array_ops.reshape(logits, [-1, num_classes])
    # targets_flat = array_ops.reshape(targets, [-1, num_classes])

    # crossent = tf.reduce_sum(tf.pow((targets_flat - logits_flat),2), axis=1)


    crossent = tf.reduce_sum(tf.square(logits - targets), reduction_indices = [2,3,4])

    #print(array_ops.shape(crossent))

    # targets = array_ops.reshape(targets, [-1])
    # if softmax_loss_function is None:
    #   crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
    #       labels=targets, logits=logits_flat)
    # else:
    #   crossent = softmax_loss_function(labels=targets, logits=logits_flat)
    # crossent *= array_ops.reshape(weights, [-1])

    if average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent)
      total_size = math_ops.reduce_sum(weights)
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      crossent /= total_size
    else:
      batch_size = array_ops.shape(logits)[0]
      sequence_length = array_ops.shape(logits)[1]
      crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
    if average_across_timesteps and not average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[1])
      total_size = math_ops.reduce_sum(weights, axis=[1])
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      crossent /= total_size
    if not average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[0])
      total_size = math_ops.reduce_sum(weights, axis=[0])
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      crossent /= total_size
    return crossent


# def onehot(data, dim):
#   data_onehot = np.zeros((dim, len(data)))
#   data_onehot[np.array(data),np.arange(len(data))] = 1
#   return data_onehot

def data_type():
  return tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    print("EPOCH SIZE: ", self.epoch_size)
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, config.vocab_size, name=name)

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, params):
    self._input = input_
    self.config = config
    self.is_training = is_training

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    def lstm_cell():
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return psrnn_cell_impl.PSRNNCell(
            size,
            params,
            reuse=tf.get_variable_scope().reuse)
      else:
        return psrnn_cell_impl.PSRNNCell(
            size,
            params)  
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    psrnns = [attn_cell() for _ in range(config.num_layers)];
    cell = tf.contrib.rnn.MultiRNNCell(
        psrnns, state_is_tuple=True)
  
    self._initial_state = []
    for i in range(config.num_layers):
        self._initial_state.append(tf.constant(np.ones((batch_size,1)).dot(params.q_1.T), dtype=data_type()))
    self._initial_state = tuple(self._initial_state) # (20, 35)
      
    # convert to one-hot encoding
    #inputs = tf.one_hot(input_.input_data, config.vocab_size)

    # JW - just pass in latent vector z into psrnn, instead of one-hot encoding it
    inputs = input_.input_data
    print("Inputs: ", inputs.shape) #(20, 20, 32)

    # random fourier features
    W_rff = tf.get_variable('W_rff', initializer=tf.constant(params.W_rff.astype(np.float32)), dtype=data_type())
    b_rff = tf.get_variable('b_rff', initializer=tf.constant(params.b_rff.astype(np.float32)), dtype=data_type())
    
    self._W_rff = W_rff
    self._b_rff = b_rff

    print("W_rff: ", W_rff.shape)    # (35, 2000)
    print("b_rff: ", b_rff.shape)    # (2000,)

    z = tf.tensordot(tf.cast(inputs, dtype=tf.float32), W_rff,axes=[[2],[0]]) + b_rff
    inputs_rff = tf.cos(z)*np.sqrt(2.)/np.sqrt(config.nRFF_Obs)

    print("inputs_rff: ", inputs_rff.shape) #(20,20,2000)
    print("z: ", z.shape) #(20,20,2000)

    # dimensionality reduction
    U = tf.get_variable('U', initializer=tf.constant(params.U.astype(np.float32)),dtype=data_type())
    U_bias = tf.get_variable('U_bias',[config.hidden_size],initializer=tf.constant_initializer(0.0))

    print("U: ", U.shape) #(2000,20)

    inputs_embed = tf.tensordot(inputs_rff, U, axes=[[2],[0]]) + U_bias

    
    print("inputs_embed: ", inputs_embed.shape) #(20,20,20)

    # update rnn state
    inputs_unstacked = tf.unstack(inputs_embed, num=num_steps, axis=1) 

    outputs, state = tf.contrib.rnn.static_rnn(
        cell, inputs_unstacked, initial_state=self._initial_state)

    # reshape output
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    
    # # softmax  (JW - why softmax??)
    initializer = tf.constant(params.W_pred.T.astype(np.float32))
    softmax_w = tf.get_variable("softmax_w", initializer=initializer, dtype=data_type())
    initializer = tf.constant(params.b_pred.T.astype(np.float32))
    softmax_b = tf.get_variable("softmax_b", initializer=initializer, dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    print(output.shape) #(400, 20)
    print(logits.shape) #(400, 20)

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size * num_steps, vocab_size])
    print("Logit shape: ", logits.shape) #(50 * 20, 32) # send in individual vectors to be decoded
    # Worked until here
    # use the contrib sequence loss and average over the batches

    # Plan: Train vae but save weights of decoder. Initialise the layers here as trainable=False. Load weights from json. Output of decoder layer is input of loss function

    h = tf.layers.dense(logits, 4*256, name="dec1_fc")
    h = tf.reshape(h, [-1, 1, 1, 4*256])
    h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec1_deconv1")
    h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec1_deconv2")
    h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec1_deconv3")
    self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec1_deconv4")

    # duplicate layers 
    original = tf.layers.dense(input_.targets, 4*256, name="dec2_fc_o")
    original = tf.reshape(original, [-1, 1, 1, 4*256])
    original = tf.layers.conv2d_transpose(original, 128, 5, strides=2, activation=tf.nn.relu, name="dec2_deconv1_o")
    original = tf.layers.conv2d_transpose(original, 64, 5, strides=2, activation=tf.nn.relu, name="dec2_deconv2_o")
    original = tf.layers.conv2d_transpose(original, 32, 6, strides=2, activation=tf.nn.relu, name="dec2_deconv3_o")
    self.original = tf.layers.conv2d_transpose(original, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec2_deconv4_o")


    print("Decoder output shape: ", self.y.shape) #(1000, 64, 64, 3)

    self.y = tf.reshape(self.y, [batch_size, num_steps, 64, 64, 3])
    self.original = tf.reshape(self.original, [batch_size, num_steps, 64, 64, 3])

    print("Loss input shape: ", self.y.shape, self.original.shape)

    loss = sequence_loss(
        self.y,
        self.original,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )
          # Aim: loss.shape = (num_steps, )
    print("Loss: ", loss.shape) #(20,)
    
    self._cost = cost = tf.reduce_mean(loss)
    self._final_state = state
    self.pred = self.y

    # initialize vars
    self.init = tf.global_variables_initializer()

    tvars = tf.trainable_variables()
    self.assign_ops = {}
    for var in tvars:
      if 'dec' in var.name:
        pshape = var.get_shape()
        pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
        assign_op = var.assign(pl)
        self.assign_ops[var] = (assign_op, pl)

    if not self.is_training:
      return


    # Think about how to store the reference of these weights and load them afterwards
    # To-do: rewrite code for training in main()
    train_vars = [var for var in tvars if 'dec' not in var.name]


    self._lr = tf.Variable(0.0, trainable=False)
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, train_vars),
                                      self.config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, train_vars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})    
  
  def init_weights(self, session):
    with open("./vae.json", 'r') as f:
      params = json.load(f)

    with g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        if 'dec1' in var.name:
          pshape = tuple(var.get_shape().as_list())
          p = np.array(params[idx])
          #print(f"pshape: {pshape}, p.shape: {p.shape}")
          assert pshape == p.shape, "inconsistent shape"
          assign_op, pl = self.assign_ops[var]
          session.run(assign_op, feed_dict={pl.name: p/10000.})
          idx += 1

    with g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        if 'dec2' in var.name:
          pshape = tuple(var.get_shape().as_list())
          p = np.array(params[idx])
          #print(f"pshape: {pshape}, p.shape: {p.shape}")
          assert pshape == p.shape, "inconsistent shape"
          assign_op, pl = self.assign_ops[var]
          session.run(assign_op, feed_dict={pl.name: p/10000.})
          idx += 1

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  # @property
  # def num_correct_pred(self):
  #     return self._num_correct_pred

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
  
def run_epoch(session, model, eval_op=None, verbose=False, save_params=False, epoch=0):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  correct_pred = 0.0
  iters = 0
  state = session.run(model.initial_state)
  pred = original = None

  fetches = {
      "cost": model.cost,
      # "num_correct_pred": model.num_correct_pred,
      "final_state": model.final_state,
      "pred": model.pred,
      "original": model.original
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
      
    feed_dict = {}
    for i, s in enumerate(model.initial_state):
      feed_dict[s] = state[i]

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    pred = vals['pred']
    original = vals["original"]

    costs += cost
    iters += model.input.num_steps
    # correct_pred += vals["num_correct_pred"]
    
    perplexity = np.exp(costs / iters)
    bpc = np.log2(perplexity)
    # accuracy = correct_pred/iters
    accuracy = 0

    if verbose and model.input.epoch_size < 10:
      # print("%.3f perplexity: %.3f bpc: %.3f cost: %.3f, accuracy: %.3f speed: %.0f wps" %
      #       (step * 1.0 / model.input.epoch_size, perplexity, bpc, cost, accuracy,
      #        iters * model.input.batch_size / (time.time() - start_time)))
      pass
    elif verbose and step % (model.input.epoch_size // 10) == 10:
      # print("%.3f perplexity: %.3f bpc: %.3f cost: %.3f, accuracy: %.3f speed: %.0f wps" %
      #       (step * 1.0 / model.input.epoch_size, perplexity, bpc, cost, accuracy,
      #        iters * model.input.batch_size / (time.time() - start_time)))
      pass
  
  x = np.array(pred).reshape((-1, 64, 64, 3))
  interval = x.shape[0] // 5
  pred = x[0::interval].reshape((-1, 64, 3))
  original = np.array(original).reshape((-1, 64, 64, 3))[0::interval].reshape((-1, 64, 3))
  img = np.concatenate([pred, original], axis=1)
  cv2.imwrite(f"./img/sequence_{epoch}.png", 255.0 * img)

  return perplexity, bpc, accuracy, np.mean(cost)


def main(_):
  
  # if not FLAGS.data_path:#43493
  #   raise ValueError("Must set --data_path to PTB data directory")

  # raw_data = reader.ptb_raw_data(FLAGS.data_path)
  # train_data, valid_data, test_data, _, word_to_id = raw_data


  raw_data = np.load("./series.npz")
  data_mu = raw_data["mu"] #(1000, 200, 32)
  data_logvar = raw_data["logvar"]
  data_action =  raw_data["action"]

  # max_seq_len = 999
  # batch_size = 100
  # N_data = len(data_mu) # should be 10k (nope, 1199 now)

  # def random_batch():
  #   indices = np.random.permutation(N_data)[0:batch_size]
  #   mu = data_mu[indices]
  #   logvar = data_logvar[indices]
  #   action = data_action[indices]
  #   s = logvar.shape
  #   z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  #   return z, action

  # raw_z, raw_a = random_batch()

  mu = data_mu
  logvar = data_logvar
  raw_a = data_action

  #mu = np.tile(mu, (10, 1))
  #logvar = np.tile(logvar, (10, 1)) #200
  s = logvar.shape
  raw_z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  #raw_z = np.tile(raw_z, (200, 1))
  
  print("Raw z shape: ", raw_z.shape)
  #inputs = raw_z
  inputs = raw_z[:, :, 0:Config.vocab_size]

  #inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
  # inputs = np.swapaxes(inputs, 1, 2)

  print("Before reshape: ", inputs.shape) 
  inputs = np.reshape(inputs, (-1, Config.vocab_size))
  print("Shape of training inputs: ", inputs.shape) 

  #inputs = inputs[:5000] # reduce training size
  
  # TRAJ_UNIT = 200 # how long each trajectory is
  # TRAIN_NUM = 500 # how many complete trajectories to use #500
  # VAL_NUM = 50  #250

  # train_size = TRAJ_UNIT * TRAIN_NUM
  # val_size = TRAJ_UNIT * VAL_NUM

  train_size = int(0.8 * inputs.shape[0])
  val_size = inputs.shape[0] - train_size
  #val_size = train_size
  two_stage_size = int(train_size)

  # train_size = 50000
  # val_size = 20000

  train_data = inputs[:train_size]
  valid_data = inputs[train_size:train_size+val_size]
  #two_stage_data = train_data[:TRAJ_UNIT*TRAIN_NUM] #take first trajectory for two-stage regression
  two_stage_data = train_data[:two_stage_size]

  print("Train shape: ", train_data.shape)
  print("Val shape: ", valid_data.shape)

  config = Config()
  eval_config = Config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  
  # print the config
  for i in inspect.getmembers(config):
    # Ignores anything starting with underscore 
    # (that is, private and protected attributes)
    if not i[0].startswith('_'):
        # Ignores methods
        if not inspect.ismethod(i[1]):
            print(i)
            
  # reshape array of raw training labels
  # raw_train_data = np.array(train_data).reshape([-1,len(train_data)])
  # raw_train_data = np.array(train_data).reshape([1,-1])
  # convert characters to one-hot encoding
  # train_data_onehot = onehot(train_data, config.vocab_size)

  # JW
  #train_data_onehot = train_data

  # perform two stage regression to obtain initialization for PSRNN
  params = two_stage_regression.two_stage_regression(
          two_stage_data, # train_data
          two_stage_data, # train_data_onehot
          config.kernel_width_Obs, config.kernel_width_P, config.kernel_width_F, 
          config.seed, 
          config.nRFF_Obs, config.nRFF_P, config.nRFF_F,
          config.dim_Obs, config.dim_P, config.dim_F, 
          config.reg_rate,
          config.obs_window)

  with g.as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
                                                
    
    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input, params=params)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input, params=params)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    # with tf.name_scope("Test"):
    #   test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
    #   with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #     mtest = PTBModel(is_training=False, config=eval_config,
    #                      input_=test_input, params=params)

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:

      valid_perplexity_all = []        
      valid_bpc_all = []
      valid_acc_all = []

      m.init_weights(session)
      #mvalid.init_weights(session)
     
      valid_perplexity, valid_bpc, valid_acc, cost = run_epoch(session, mvalid)
      print("Epoch: %d Valid Perplexity: %.3f Valid BPC: %.3f Valid Accuracy: %.3f Cost: %.3f"
            % (0, valid_perplexity, valid_bpc, valid_acc, cost))
      valid_perplexity_all.append(valid_perplexity)
      valid_bpc_all.append(valid_bpc)
      valid_acc_all.append(valid_acc)
      
      for i in range(config.max_epoch):
        m.assign_lr(session, config.learning_rate * config.lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        
        train_perplexity, train_bpc, train_acc, cost = run_epoch(session, m, eval_op=m.train_op, verbose=True, epoch=i)
        losses.append(cost)
        print("Epoch: %d Train Perplexity: %.3f Train BPC: %.3f Train Accuracy: %.3f Cost: %.3f" 
              % (i + 1, train_perplexity, train_bpc, train_acc, cost))
        
        valid_perplexity, valid_bpc, valid_acc, cost = run_epoch(session, mvalid, epoch=-i)
        print("Epoch: %d Valid Perplexity: %.3f Valid BPC: %.3f Valid Accuracy: %.3f Cost: %.3f" 
              % (i + 1, valid_perplexity, valid_bpc, valid_acc, cost))
        valid_perplexity_all.append(valid_perplexity)
        valid_bpc_all.append(valid_bpc)
        valid_acc_all.append(valid_acc)

      # test_perplexity, test_bpc, test_acc = run_epoch(session, mtest, save_params=True)
      # print("Test Perplexity: %.3f Test BPC: %.3f Test Accuracy: %.3f" % (test_perplexity, test_bpc, test_acc))
      
      # print("validation perplexity\n",valid_perplexity_all)
      # print("validation bpc\n",valid_bpc_all)
      # print("validation acc\n",valid_acc_all)

if __name__ == "__main__":
  tf.app.run()
  
