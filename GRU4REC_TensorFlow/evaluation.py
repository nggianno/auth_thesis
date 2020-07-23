
import numpy as np
import pandas as pd


def evaluate_sessions_batch(model, train_data, test_data, cut_off=20, batch_size=50, session_key='user_id', item_key='product_id', time_key='timestamp'):
    
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : It contains the transactions of the train set. In evaluation phase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : float Recall@N
          float MRR@N
          pandas.DataFrame
          Top-N items based on prediction scores for every user session of the test set.
          Columns: user session ids of test data; rows: Top-N itemids
   
    '''
    model.predict = False
    """Build itemidmap from train data."""
    #print(train_data)
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
    print(itemidmap)
    itemidmap.to_csv('./item_lookup.csv')

    test_data.sort_values([session_key, time_key], inplace=True)
    """Build sessionidmap from test data."""
    sessionids = test_data[session_key].unique()
    sessionidmap = pd.Series(data=np.arange(len(sessionids)), index=sessionids)
    print(sessionidmap)
    topN_df = pd.DataFrame(columns=sessionidmap.index.values)

    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    offset_sessionids = np.array(sessionidmap)
    #print(offset_sessionids)

    """compute the cumulative sum for the size of each session(per clicks)"""
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    # print(offset_sessions.size)
    # print(offset_sessions)
    evaluation_point_count = 0
    mrr, recall = 0.0, 0.0
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    #iterids = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    in_idx = np.zeros(batch_size, dtype=np.int32)
    np.random.seed(42)
    while True:
        #print(iters)
        valid_mask = iters >= 0
        """valid_mask contains an array of boolean values of the iters. 
        If sum is zero, all sessions have be accessed so the evaluation stops"""
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        #print('Start valid:{}'.format(start_valid))
        #print('End valid:{}'.format(end[valid_mask]))
        """minlen --->  minimum length of a session"""
        minlen = (end[valid_mask]-start_valid).min()
        #print('Minlen:{}'.format(minlen))
        in_idx[valid_mask] = test_data[item_key].values[start_valid]

        for i in range(minlen-1):
            out_idx = test_data[item_key].values[start_valid+i+1]
            # print('Out_idx:{}'.format(out_idx))
            preds = model.predict_next_batch(iters, in_idx, itemidmap, batch_size)
            preds.fillna(0, inplace=True)
            #print(preds)
            #print(preds)
            #print('Predictions{}\n:'.format(preds))
            in_idx[valid_mask] = out_idx
            #ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            ranks = (preds.values.T[valid_mask].T > np.diag(preds.loc[in_idx].values)[valid_mask]).sum(axis=0) + 1
            # print('Ranks{}'.format(ranks))
            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evaluation_point_count += len(ranks)

            """Get a dataframe with the top-20 itemids for every event of the batch"""
            preds = pd.DataFrame(preds)
            top_preds = pd.DataFrame(columns=preds.columns)

            for i in range(50):

                #sort by prediction value
                vector = preds[i].sort_values(ascending=False)
                #get top-20
                vector = vector.head(20)
                print(preds)
                #pass item ids
                #for i in range(20): vector.index.values[i] = itemidmap.index[vector.index.values[i]]
                top_preds[i] = vector.index.values

            #print(top_preds)


        start = start+minlen-1
        mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
        #print('Mask{}'.format(mask))

        """Dataset of Top-N products for every session"""

        for idx in mask:
            #print(iters[idx])
            idx2 = sessionidmap.index[iters[idx]]
            #print(idx2)
            #print(top_preds[idx])
            topN_df[idx2] = top_preds[idx]
            #print(topN_df)
            maxiter += 1
            #print(maxiter)
            if maxiter >= len(offset_sessions)-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter+1]

    return recall/evaluation_point_count, mrr/evaluation_point_count, topN_df
