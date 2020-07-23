from keras.models import load_model
from DeepMF_keras.evaluate import evaluate_model
from DeepMF_keras.dataset import DataSet
from DeepMF_keras.dmf import DMF
from keras.utils import get_custom_objects
import numpy as np
import time
import logging
import math
import os
import pandas as pd
from argparse import ArgumentParser
from time import time
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.models import model_from_json
import json
import tensorflow as tf


#load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("pharmacy-data_u[64,32]_i[128,32]_32_1587378518.h5")
# print("Loaded model from disk")

# class CustomInitializer:
#     def __call__(self, shape, dtype=None):
#         return DMF.init_normal(shape, dtype=dtype)
#
# ci = CustomInitializer()



# latest = tf.train.latest_checkpoint('./training/checkpoints')
# print(latest)

#get_custom_objects().update({'InitNormal': DMF.InitNormal(0.01)})
model = load_model('./model/pharmacy-data_u[64, 32]_i[128, 32]_128_1589905568_randomnormal.h5',
                   custom_objects={"predict": DMF})


#(hits, ndcgs, topN_df) = evaluate_model(model, DataSet.test_ratings, DataSet.test_negatives,
#                                        DataSet.data_matrix, 10)

# print('hits:{0}\nndcgs:{1}\n'.format(hits,ndcgs))

# hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
#
# print('HR = %.4f, NDCG = %.4f' % (hr, ndcg))
# print(topN_df)

