
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from curses import raw


import os


import tensorflow as tf
import numpy as np


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return list(f.read())

def _build_vocab(filename):
  data = _read_words(filename)

  word_to_id = {}
  
  for word in data:
      if word not in word_to_id:
          word_to_id[word] = len(word_to_id)

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts characters to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_producer(raw_data, batch_size, num_steps, vocab_size, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """

  # raw_data -> (1197801, 35) # initially was (121540,) a bag of words
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps, vocab_size]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)
    
    data_len = tf.size(raw_data) // vocab_size  # no. of z vector
    batch_len = data_len // batch_size 
    data = tf.reshape(raw_data[0 : batch_size * batch_len, :],
                      [batch_size, batch_len, vocab_size]) 

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps, 0],
                         [batch_size, (i + 1) * num_steps, vocab_size])
    x.set_shape([batch_size, num_steps, vocab_size])
    y = tf.strided_slice(data, [0, i * num_steps + 1, 0],
                         [batch_size, (i + 1) * num_steps + 1, vocab_size])
    y.set_shape([batch_size, num_steps, vocab_size])


    print("X: ", x.shape) #(20, 20, 35)
    print("Y: ", y.shape) #(20, 20, 35)

    return x, y
