import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import math


class PreprocessData:
    def __init__(self,userkey,itemkey,timekey):
        """Class Constructor"""
        print("Start preprocessing...\n")
        self.userkey = userkey
        self.itemkey = itemkey
        self.timekey = timekey

    def clean_data(self,df):
        """CLEAN DATASET """

        '''1) Drop unnecessary columns of pandas Dataframe / Non Available Values / Duplicates'''
        # data.dropna(inplace=True)
        # print(data.head(10))
        df.drop_duplicates(inplace=True)

        # sort dataset by user session and event time
        df.sort_values(by=[self.userkey, self.timekey], inplace=True)

        '''2) Drop session ids or user ids with a single action'''
        session_freq = df[self.userkey].value_counts()
        # print(session_freq)
        session_freq = pd.DataFrame({self.userkey: session_freq.index, 'number of events': session_freq.values})
        session_freq = session_freq[session_freq['number of events'] == 1]
        list = session_freq[self.userkey]
        df = df[~df[self.userkey].isin(list)]
        # print(data[userkey].value_counts())

        """ 3) Delete rare product_ids """
        product_freq = df[self.itemkey].value_counts()
        # print(product_freq)
        product_freq = pd.DataFrame({self.itemkey: product_freq.index, 'product_frequency': product_freq.values})
        product_freq = product_freq[product_freq['product_frequency'] == 1]
        list2 = product_freq[self.itemkey]
        df = df[~df[self.itemkey].isin(list2)]
        # print(data[itemkey].value_counts())

        return df

    def build_dataset_for_rnn(self,df):
        '''duplicate * (weight of views) addtocart and purchase rows '''

        CART_WEIGHT = 2
        PURCHASE_WEIGHT = 4

        df = df.append([df[df.event_type.eq('cart')]] * CART_WEIGHT, ignore_index=True)

        df = df.append([df[df.event_type.eq('buy')]] * PURCHASE_WEIGHT, ignore_index=True)
        df.sort_values(by=[self.userkey, self.timekey], inplace=True)

        df = pd.DataFrame(df)

        return df

    def build_dataset_with_ratings(self,df):

        df = df[[self.userkey, self.itemkey, 'event_type']]
        """Rating function --- Rating range 0-5
            rating(i) = (view_num(i) * 0.10 + addtocart_num(i) * 0.30 + transaction_num(i) * 0.60)*5"""

        event_type_rating = {
            'view': 1,
            'add to cart': 1.5,
            'buy': 3,
        }

        df['rating'] = df['event_type'].apply(lambda x: event_type_rating[x])

        ratings_df = df.groupby([self.userkey,self.itemkey]).sum().reset_index()
        ratings_df = pd.DataFrame(ratings_df)

        return ratings_df

    def threshold_rating(self,df, upper_thres=5):
        df['rating'] = df['rating'].apply(lambda x: upper_thres if (x > upper_thres) else x)

        return df

    def change_ui_index(self,df):
        df[self.userkey] = df.groupby(self.userkey).ngroup()
        df[self.itemkey] = df.groupby(self.itemkey).ngroup()

        return df

    def normalize_ratings(self,df):
        """Normalize ratings in 0 to 5 scale"""

        x = np.array(df['rating'].values).reshape(-1, 1)  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
        x_scaled = min_max_scaler.fit_transform(x)
        x_scaled = np.array(x_scaled)
        df['rating'] = x_scaled.round(2)
        print(df)

        return df

    def compute_dwell_time(self,df):
        times_t = np.roll(df[self.timekey], -1)  # Take time row
        times_dt = df[self.timekey]  # Copy, then displace by one

        diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference

        length = len(df[self.itemkey])

        # cummulative offset start for each session
        offset_sessions = np.zeros(df[self.userkey].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = df.groupby(self.userkey).size().cumsum()

        offset_sessions = offset_sessions - 1
        offset_sessions = np.roll(offset_sessions, -1)

        # session transition implies zero-dwell-time
        # note: paper statistics do not consider null entries,
        # though they are still checked when augmenting
        np.put(diffs, offset_sessions, np.zeros((offset_sessions.shape)), mode='raise')

        return diffs

    def join_dwell_reps(self,df, dt, threshold=2000):
        # Calculate d_ti/threshold + 1
        # then add column to dataFrame

        dt //= threshold
        dt += 1
        df['DwellReps'] = pd.Series(dt.astype(np.int64), index=dt.index)

        return df

    def get_session_duration_arr(self,df):
        """Compute session duration for each session"""
        df[self.timekey] = pd.to_datetime(df[self.timekey])
        df = df.groupby(self.userkey)[self.timekey].agg(
            lambda x: max(x) - min(x)).to_frame().rename(columns={self.timekey: 'Duration'})
        df = pd.DataFrame(df)
        return df

    def user_item_mapping(self,df):

        df.sort_values(self.userkey, inplace=True)

        userids = df[self.userkey].unique()
        useridmap = pd.Series(data=np.arange(len(userids)), index=userids)
        useridmap.index.names = [self.userkey]


        itemids = df[self.itemkey].unique()
        itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
        itemidmap.index.names = [self.itemkey]

        return useridmap,itemidmap

    def get_statistics(self,df):
        """Calculate percentage of event_types"""
        event_freq = df['event_type'].value_counts()
        print(event_freq)

        event_percentage = event_freq / len(df) * 100
        print(event_percentage)
        event_percentage = pd.DataFrame({'event_type': event_freq.index.values, 'percent': event_percentage})
        event_percentage.reset_index(inplace=True)
        event_percentage.drop(columns='index', inplace=True)
        print(event_percentage)
        # print(event_freq.index.values)
        sns.set(style="whitegrid")
        # tips = sns.load_dataset("event_percentage")
        ax = sns.barplot(x="event_type", y="percent", data=event_percentage)
        plt.show()

        """Userids vs Cookieids"""
        unique_users = df['user_id'].nunique()
        print("Unique logged users:{}".format(unique_users))
        unique_cookies = df[self.userkey].nunique()
        print("Unique unlogged users:{}".format(unique_cookies))

        users_pie_df = pd.DataFrame([unique_users, unique_cookies], index=['Registed Users', 'Unregistered Users'],
                                    columns=['number'])

        print(users_pie_df)
        # ax1 = plt.subplots()
        users_pie_df.plot(kind='pie', subplots=True, figsize=(8, 8))
        plt.show()

    def edit_column_context(self, df, down_datetime_limit, up_datetime_limit):
        print(df.shape)
        df.drop_duplicates(inplace=True)
        df = df.sort_values(by='timestamp')
        df[self.userkey] = df[self.userkey].astype(str)
        df[self.timekey] = df[self.timekey].map(lambda x: x[:19])

        """Choose datetime limits for dataset"""
        df = df[df.timestamp >= down_datetime_limit]
        df = df[df.timestamp <= up_datetime_limit]
        # df_sorted['timestamp'] = df_sorted['timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%d-%m %H:%M:%S')
        #                                 if x > '2020-05-31 00:00:00' else datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df.sort_values(by=self.timekey, inplace=True)

        """clean users with awkward names """
        df['user_id'] = df['user_id'].map(lambda x:
                            math.nan if x == 'guest' or x == 'undefined' or x == 'userID6089190622420959043' else x)

        return df

    def update_userids(self,df):
        df['user_id'] = df['user_id'].fillna(df.groupby('cookie_id')['user_id'].transform('first'))
        # another way to do it
        # df['user_id'] = df.groupby('cookie_id').transform(lambda x: x.fillna(x.iloc[0]))
        return df

    def delete_outliers(self,df):

        df['user_id'] = df['user_id'].fillna(-1)
        df['user_id'] = df['user_id'].map(lambda x: int(x) if x != -1 else x)
        # print(data.user_id)
        user_cookie_freq = df.groupby(by=['cookie_id'])['user_id'].nunique()
        user_cookie_freq = pd.DataFrame(user_cookie_freq)
        user_cookie_freq = user_cookie_freq[user_cookie_freq['user_id'] > 1]
        list = user_cookie_freq.index.values
        df = df[~df['cookie_id'].isin(list)]
        user_cookie_df = df.groupby(by=['cookie_id'])['user_id'].unique()
        user_cookie_df = pd.DataFrame(user_cookie_df)
        # user_cookie_df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/api-data/up_to_27_May/user_cookie.csv')

        return df
