import pandas as pd
import numpy as np

def change_ui_index(df,userkey,itemkey):

    #df[userkey] = df.groupby(userkey).ngroup()
    df[itemkey] = df.groupby(itemkey).ngroup()

    return df

def itemmap(df,itemkey):

    df['idx'] = df[itemkey].astype("category").cat.codes
    item_lookup = df[[itemkey, 'idx']].drop_duplicates()
    item_lookup['idx'] = item_lookup.idx.astype(str)
    item_lookup.to_csv('./item_lookup3.csv')


def dwelltime_edit_dataset(df,userkey,timekey):



    df = df.append([df[df['dwelltime'] >= '0 days 00:03:00.000000000']], ignore_index=True)
    #df = df.append([df[df.event_type.eq('add to cart')]] * CART_WEIGHT, ignore_index=True)

    df.sort_values(by=[userkey, timekey], inplace=True)

    df = pd.DataFrame(df)

    return df

def train_test_split(data,userkey,timekey,date_split):

    train_data = data[data[timekey] <= date_split]
    test_data = data[data[timekey] > date_split]
    train_data.sort_values(by=userkey, inplace=True)
    test_data.sort_values(by=userkey, inplace=True)

    return train_data,test_data

def drop_single_sessions(data,userkey):

    session_freq = data[userkey].value_counts()
    session_freq = pd.DataFrame({userkey: session_freq.index, 'number of events': session_freq.values})
    session_freq = session_freq[session_freq['number of events'] == 1]
    list = session_freq[userkey]
    data = data[~data[userkey].isin(list)]

    return data

def match_train_test_items(train_data,test_data,itemkey):

    print(test_data.shape)
    unique_products = train_data[itemkey].unique()
    test_data = test_data[test_data[itemkey].isin(unique_products)]
    print(test_data.shape)

    return test_data


if __name__ == '__main__':

    # set keys
    #userkey = 'user_id'
    userkey = 'cookie_id'
    itemkey = 'product_id'
    timekey = 'timestamp'

    date_split = '2020-06-01 00:00:00'


    df = pd.read_csv('/home/ubuntu/django-api/datasets/rnn_df_dwelltime.csv')
    print(df.shape)
    itemmap(df,itemkey)
    df = change_ui_index(df,userkey=userkey,itemkey=itemkey)
    df = dwelltime_edit_dataset(df,userkey,itemkey)
    print(df.shape)

    df.sort_values(by=timekey, inplace=True)
    print(df[timekey])

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df)

    df = df[df[timekey] > date_split]
    df.to_csv('/home/ubuntu/django-api/gru4rec/June0119dataset.csv')

    """1) split rnn dataset into training set and testing set based on chronological sequence"""
    #train_df, test_df = train_test_split(df,userkey,timekey,date_split)
    #train_df = drop_single_sessions(train_df, userkey)
    #print(train_df.shape,test_df.shape)

    """2) match trainset with testset items"""
    #test_df = match_train_test_items(train_df,test_df,itemkey)
    #print(test_df.shape)

    """3) clear single-action sessions from test set"""

    #test_df = drop_single_sessions(test_df,userkey)
    test_df = drop_single_sessions(df,userkey)
    test_df = match_train_test_items(df,test_df,itemkey)
    print(test_df.shape)

    #train_df.to_csv('/home/ubuntu/django-api/gru4rec/trainset2.csv')
    #test_df.to_csv('/home/ubuntu/django-api/gru4rec/testset2.csv')
    test_df.to_csv('/home/ubuntu/django-api/gru4rec/June0119testset.csv')

