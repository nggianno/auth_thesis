import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from lightfm import cross_validation
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
from lightfm.evaluation import auc_score



class LightFactorizationMachines:

    def __init__(self,userkey,itemkey):
        print("LightFM class constructed\n")
        self.userkey = userkey
        self.itemkey = itemkey

    def create_sparse_matrix(self,data):

        data[self.userkey] = data[self.userkey].astype("category")
        data[self.itemkey] = data[self.itemkey].astype("category")
        # data['brand'] = data['brand'].astype("category")
        data['user'] = data[self.userkey].cat.codes
        data['item'] = data[self.itemkey].cat.codes
        print(data)

        # Create a lookup frame so we can get the brand names back in
        # readable form later.
        user_lookup = data[['user', self.userkey]].drop_duplicates()
        item_lookup = data[['item', self.itemkey]].drop_duplicates()
        user_lookup['user'] = user_lookup.user.astype(str)
        user_lookup = pd.DataFrame(user_lookup)
        user_lookup.set_index(self.userkey, inplace=True)
        item_lookup['item'] = item_lookup.item.astype(str)
        item_lookup = pd.DataFrame(item_lookup)
        item_lookup.set_index(self.itemkey,inplace=True)
        print(user_lookup, item_lookup)

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

    def run_lightfm(self,method,user_id,lr,epochs,training):

        #TRAIN_PATH = '/home/nick/Desktop/thesis/datasets/pharmacy-data/ratings-data/ratings.csv'
        TRAIN_PATH = '/home/ubuntu/django-api/datasets/ratings_df.csv'

        traindata = pd.read_csv(TRAIN_PATH)
        # print(traindata)
        # print('\n')
        csr_data, user_lookup, item_lookup = self.create_sparse_matrix(traindata)

        user_id = user_lookup.loc[user_id]
        print(user_id)

        user_items_train = csr_data.T.tocsr()

        print('\n')
        # print(user_items_train.shape)
        # print(user_items_train)

        # print("Splitting the data into train/test set...\n")
        # train, test = cross_validation.random_train_test_split(user_items_train)
        # print(train, test)
        if training is True:

            if method == 'bpr':

                model = LightFM(learning_rate=lr, loss='bpr')

            elif method == 'warp':

                model = LightFM(learning_rate=lr, loss='warp')

            print("Fitting models...\n")

            model.fit(user_items_train, epochs=epochs)

            with open('lightFM_model.sav', 'wb') as pickle_out:
                pickle.dump(model, pickle_out)

        else:

            with open('lightFM_model.sav', 'rb') as pickle_in:
                model = pickle.load(pickle_in)

        print("Generating recommendations for user...\n")
        # test_userids = test.row
        # test_itemids = test.col
        user_items_train = user_items_train.tocoo()
        itemids = user_items_train.col


        #test_userids = np.unique(test_userids)
        itemids = np.unique(itemids)

        #print(test_userids, test_itemids)

        scores = model.predict(user_id, itemids)
        top_items = itemids[np.argsort(-scores)]

        print("user_id:{}".format(user_id))

        top_items = top_items[:20]

        recommendations = []

        for i in range(20):
            recommendations.append(item_lookup.index[top_items[i]])


        return model,recommendations,user_items_train


    def evaluate(self,model,user_items_train):

        print("Splitting the data into train/test set...\n")
        train, test = cross_validation.random_train_test_split(user_items_train)
        print(train, test)


        print("Evaluating methods...\n")


        train_recall_10 = recall_at_k(model, train, k=10).mean()
        test_recall_10 = recall_at_k(model, test, k=10).mean()

        train_recall_20 = recall_at_k(model,train,k=20).mean()
        test_recall_20 = recall_at_k(model,test,k=20).mean()

        train_precision_10 = precision_at_k(model, train, k=10).mean()
        test_precision_10 = precision_at_k(model, test, k=10).mean()

        train_precision_20 = precision_at_k(model,train,k=20).mean()
        test_precision_20 = precision_at_k(model,test,k=20).mean()


        print("Train : Recall@10:{0:.3f}, Recall@20:{1:.3f}".format(train_recall_10,train_recall_20))
        print("Test : Recall@10:{0:.3f}, Recall@20:{1:.3f}".format(test_recall_10,test_recall_20))

        print("Train: Precision@10:{0:.3f}, Precision@20:{1:.3f}".format(train_precision_10, train_precision_20))
        print("Test: Precision@10:{0:.3f}, Precision@20:{1:.3f}".format(test_precision_10, test_precision_20))
