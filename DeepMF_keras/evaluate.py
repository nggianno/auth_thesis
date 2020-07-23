import numpy as np
import math
import heapq
import pandas as pd

_model = None
_test_ratings = None
_test_negatives = None
_data_matrix = None


def evaluate_model(model, test_ratings, test_negatives, data_matrix, k):
    global _model
    global _test_ratings
    global _test_negatives
    global _data_matrix

    _model = model
    _test_ratings = test_ratings #<class 'list'>
    _test_negatives = test_negatives #<class 'list'>
    _data_matrix = data_matrix

    # print(_test_ratings)
    # print(len(_test_ratings))
    # print(_test_negatives)

    hits, ndcgs = [], []
    topN_df = pd.DataFrame(columns=np.arange(len(_test_ratings)))
    for i in range(len(_test_ratings)):
        (hr, ndcg, topN_df) = _evaluate_one_rating(topN_df, i, k=k)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs, topN_df


def _evaluate_one_rating(topN_df, idx, k):
    rating = _test_ratings[idx]
    items = _test_negatives[idx]
    user = rating[0]
    #user += 1
    gt_item = rating[1]
    #gt_item += 1
    items.append(gt_item)
    #print(user,gt_item)
    #print(_data_matrix)

    items_input = []
    users_input = []
    for item in items:
        items_input.append(_data_matrix[:, item])
        users_input.append(_data_matrix[user])

    predictions = _model.predict([np.array(users_input), np.array(items_input)],
                                 batch_size=100 + 1,
                                 verbose=0)
    #print('Predictions:{}'.format(predictions))
    map_item_score = {}
    i = idx
    #print(list(enumerate(items)))
    for idx, item in enumerate(items):
        map_item_score[item] = predictions[idx]

    map_item_df = pd.DataFrame.from_dict(map_item_score)
    #print('Map Item Score{}'.format(map_item_df))
    map_item_df.index = [user]
    map_item_df = map_item_df.T
    map_item_df.sort_values(by=int(map_item_df.columns.values),ascending=False,inplace=True)
    map_item_df = map_item_df.head(20) #get Top-20 item predictions
    vector = np.array(map_item_df.index.values)

    #print('Index:{}'.format(i))

    topN_df[i] = vector

    items.pop()
    rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
    hr = get_hit_ratio(rank_list, gt_item)
    ndcg = get_ndcg(rank_list, gt_item)
    return hr, ndcg, topN_df


def get_hit_ratio(rank_list, gt_item):
    if gt_item in rank_list:
        return 1
    return 0


def get_ndcg(rank_list, gt_item):
    for idx, item in enumerate(rank_list):
        if item == gt_item:
            return math.log(2) / math.log(idx + 2)
    return 0
