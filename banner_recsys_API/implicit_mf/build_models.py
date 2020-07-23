import implicit
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import pickle
from scipy.sparse import csr_matrix
from implicit.evaluation import precision_at_k, ndcg_at_k, train_test_split
from scipy import sparse


class Implicit:
    def __init__(self,userkey,itemkey):
        print("Implicit Constructed!")
        self.userkey = userkey
        self.itemkey = itemkey

    def create_sparse_matrix(self,data):

        data[self.userkey] = data[self.userkey].astype("category")
        data[self.itemkey] = data[self.itemkey].astype("category")
        # data['brand'] = data['brand'].astype("category")
        data['user'] = data[self.userkey].cat.codes
        data['item'] = data[self.itemkey].cat.codes

        # Create a lookup frame so we can get the brand names back in
        # readable form later.
        user_lookup = data[['user', self.userkey]].drop_duplicates()
        item_lookup = data[['item', self.itemkey]].drop_duplicates()
        # brand_lookup['brand_id'] = item_lookup.brand_id.astype(str)
        user_lookup['user'] = user_lookup.user.astype(str)
        user_lookup = pd.DataFrame(user_lookup)
        user_lookup.set_index(self.userkey, inplace=True)
        item_lookup['item'] = item_lookup.item.astype(str)
        #print(user_lookup, item_lookup)

        data = data.drop([self.userkey, self.itemkey], axis=1)
        #print(data)

        # Create lists of all users, items and their event_strength values
        users = list(np.sort(data.user.unique()))
        items = list(np.sort(data.item.unique()))
        actions = list(data.rating)

        # Get the rows and columns for our new matrix
        rows = data.user.astype(int)
        cols = data.item.astype(int)
        # Create a sparse matrix for our users and brands containing eventStrength values
        data_sparse_new = csr_matrix((actions, (cols, rows)), shape=(len(items), len(users)))

        return data_sparse_new, user_lookup, item_lookup

    def get_recommendations(self,model_name,user_id,training):

        #TRAIN_PATH = '/home/ubuntu/django-api/datasets/ratings_df.csv'
        #ratings_df = pd.read_csv(TRAIN_PATH)
        #item_user_csr, user_lookup, item_lookup = self.create_sparse_matrix(ratings_df)
        item_user_csr = sparse.load_npz('/home/ubuntu/django-api/djangoAPI/implicit_mf/item_user_csr.npz')
        user_lookup = pd.read_csv('/home/ubuntu/django-api/djangoAPI/implicit_mf/user_lookup.csv',index_col='cookie_id')
        item_lookup = pd.read_csv('/home/ubuntu/django-api/djangoAPI/implicit_mf/item_lookup.csv')
        user_idx = user_lookup.loc[int(user_id)].at['user']
        #print(user_lookup, item_lookup)
        item_lookup['item'] = item_lookup.item.astype(str)
        alpha_val = 15
        item_user_csr = (item_user_csr * alpha_val).astype('double')

        """initialize a model --- choose a model
        BPR , ALS, LMF, """
        if training == True:

            if model_name == 'als':
                model = implicit.als.AlternatingLeastSquares(factors=100)
            elif model_name == 'bpr':
                model = implicit.bpr.BayesianPersonalizedRanking(factors=80)
            elif model_name == 'lmf':
                model = implicit.lmf.LogisticMatrixFactorization(factors=100)
            elif model_name == 'itemknn':
                model = implicit.nearest_neighbours.ItemItemRecommender()
            else:
                ('Wrong algorithm\n')

            model.fit(item_user_csr)

            with open('{}_model.sav'.format(model_name), 'wb') as pickle_out:
                pickle.dump(model, pickle_out)

        else:

            with open('{}_model.sav'.format(model_name), 'rb') as pickle_in:
                model = pickle.load(pickle_in)

        user_items_csr = item_user_csr.T.tocsr()
        recommendations = model.recommend(int(user_idx), user_items_csr,filter_already_liked_items = True,N=20)
        recommendations = pd.DataFrame(recommendations, columns=['item', 'score'])
        recommendations['item'] = recommendations.item.astype(str)
        recommendations = recommendations.merge(item_lookup, on='item')
        top_products = recommendations['product_id']

        return top_products


    def train_test_split_recommend(self,model,user_item_csr,user_lookup,user_id):

        train, test = train_test_split(user_item_csr)
        # train the model on a sparse matrix of item/user/confidence weights
        model.fit(train.T.tocsr())

        """Calculate Precision@N & NDCG@N"""
        precision = precision_at_k(model, train, test, K=20)
        ndcg = ndcg_at_k(model, train, test, K=20)

        print('Precision@20: {0}\n NDCG@20: {1}\n'.format(precision, ndcg))

        """Recommend items to every user"""
        top_rec_4all = model.recommend_all(test,filter_already_liked_items=True)
        top_rec_4all = top_rec_4all.T
        # top_rec_4all = pd.DataFrame(top_rec_4all)
        top_rec_4all = pd.DataFrame(data=top_rec_4all, columns=user_lookup.index.categories)
        print('Recommendations Dataframe:\n{}'.format(top_rec_4all))

        top_products = top_rec_4all[user_id]

        return top_products

