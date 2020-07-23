import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python.ops import rnn_cell
import pandas as pd
import numpy as np


#self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)

with tf.Session() as sess:
  # Restore variables from disk.
  saver = tf.train.import_meta_graph('./checkpoint/gru-model-2.meta')

  #saver = tf.train.Saver()
  saver.restore(sess, "./checkpoint/gru-model-2")
  print("Model restored.")



