import numpy as np
import pandas as pd
from hpfrec import HPF
from poismf import PoisMF
import dill
import pickle
from scipy.sparse import coo_matrix



class PoissonFactorization():

    def __init__(self,userkey,itemkey):
        print("Poisson Predictor Constructed!")
        self.userkey = userkey
        self.itemkey = itemkey

    def build_dataset(self,path):
        print("HPF class constructed")
        df = pd.read_csv(path)
        df.rename(columns={self.userkey: "UserId", self.itemkey: "ItemId", "rating": "Count"}, inplace=True)

        return df

    def HPF(self,df,userid,train):

        if train is True:

            """Train Model"""

            recommender = HPF(
                k=20, a=0.3, a_prime=0.3, b_prime=1.0,
                c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
                stop_crit='train-llk', check_every=10, stop_thr=5e-4,
                users_per_batch=None, items_per_batch=None, step_size=lambda x: 1 / np.sqrt(x + 2),
                maxiter=200, reindex=True, verbose=True,
                random_seed=None, allow_inconsistent_math=False, full_llk=False,
                alloc_full_phi=False, keep_data=True, save_folder=None,
                produce_dicts=True, keep_all_objs=True, sum_exp_trick=False)

            recommender.fit(df, val_set=df.sample(10 ** 4))
            dill.dump(recommender, open("HPF_obj.dill", "wb"))

        else:
            """Load Model"""

            recommender = dill.load(open("HPF_obj.dill", "rb"))

        predictions = recommender.topN(user=userid, n=20, exclude_seen=True)

        return predictions

    def PMF(self,df,userid,train):

        #print("Poisson Matrix Factorization\n")

        if train is True:

            model = PoisMF()
            model.fit(df)
            with open('poisson_model.sav', 'wb') as pickle_out:
                pickle.dump(model, pickle_out)

        else:

            with open('poisson_model.sav', 'rb') as pickle_in:
                model = pickle.load(pickle_in)

        print("TopN Recommendation for userid:{}".format(userid))
        predictions = model.topN(userid, n = 20)

        return predictions
