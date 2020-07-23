import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import argparse

import model
import evaluation

from sklearn.model_selection import train_test_split

"""Uncomment the dataset the dataset path for experimenting """

#PATH_TO_TRAIN = '/home/nick/Desktop/thesis/datasets/pharmacy-data/rnn_train_indexed.csv'
PATH_TO_TRAIN = '/home/nick/Desktop/thesis/datasets/pharmacy-data/api-data/rnn-data/trainset.csv'
PATH_TO_TEST = '/home/nick/Desktop/thesis/datasets/pharmacy-data/api-data/rnn-data/testset.csv'
#PATH_TO_TEST = '/home/nick/Desktop/thesis/datasets/pharmacy-data/rnn_test_indexed.csv'


class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 50
    dropout_p_hidden=1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'cookie_id'
    item_key = 'product_id'
    time_key = 'timestamp'
    #time_key = 'event_time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=200, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='top1', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    #data = pd.read_csv(PATH_TO_TRAIN, dtype={'product_id': np.int64})
    data = pd.read_csv(PATH_TO_TRAIN)
    #valid = pd.read_csv(PATH_TO_TEST, dtype={'product_id': np.int64})
    valid = pd.read_csv(PATH_TO_TEST)
    # data.drop(columns='Unnamed: 0',axis=1,inplace=True)
    # valid.drop(columns='Unnamed: 0',axis=1,inplace=True)
    print(data,valid)
    #data, valid = train_test_split(data, random_state=42)
    args = Args()
    args.n_items = len(data['product_id'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args.dropout_p_hidden)
    print(args.loss)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data,args.loss)
        else:
            print("Testing")
            rec , mrr , topN = evaluation.evaluate_sessions_batch(gru, data, valid,session_key='cookie_id', item_key='product_id', time_key='timestamp')
            print('Recall@20: {}\tMRR@20: {}'.format(rec, mrr))
            print(topN)
            topN = pd.DataFrame(topN)
            """store Top20 products for every user_session in a csv file"""
            topN.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/api-data/rnn-data/top1-top20_2.csv')
