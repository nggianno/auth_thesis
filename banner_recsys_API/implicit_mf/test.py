import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse


def create_sparse_matrix(data,user_key = 'cookie_id',item_key='product_id'):

    data[user_key] = data[user_key].astype("category")
    data[item_key] = data[item_key].astype("category")
    data['user'] = data[user_key].cat.codes
    data['item'] = data[item_key].cat.codes

    # Create a lookup frame so we can get the brand names back in
    # readable form later.
    user_lookup = data[['user', user_key]].drop_duplicates()
    item_lookup = data[['item', item_key]].drop_duplicates()

    user_lookup['user'] = user_lookup.user.astype(str)
    user_lookup = pd.DataFrame(user_lookup)
    user_lookup.set_index(user_key,inplace=True)
    item_lookup['item'] = item_lookup.item.astype(str)
    print(user_lookup,item_lookup)

    data = data.drop([user_key,item_key], axis=1)

    # Create lists of all users, items and their event_strength values
    users = list(np.sort(data.user.unique()))
    items = list(np.sort(data.item.unique()))
    actions = list(data.rating)

    #print(users,brands,actions)
    # Get the rows and columns for our new matrix
    rows = data.user.astype(int)
    cols = data.item.astype(int)
    #print(rows,cols)
    # Create a sparse matrix for our users and brands containing eventStrength values
    data_sparse_new = csr_matrix((actions, (cols, rows)), shape=(len(items), len(users)))

    return data_sparse_new, user_lookup, item_lookup


TRAIN_PATH = '/home/ubuntu/django-api/datasets/ratings_df.csv'
ratings_df = pd.read_csv(TRAIN_PATH)
item_user_csr, user_lookup, item_lookup = create_sparse_matrix(ratings_df)
user_lookup.to_csv('user_lookup.csv')
item_lookup.to_csv('item_lookup.csv')
print(user_lookup, item_lookup)
sparse.save_npz("item_user_csr.npz", item_user_csr)
#item_user_csr = sparse.load_npz("item_user_csr.npz")
print(item_user_csr)

