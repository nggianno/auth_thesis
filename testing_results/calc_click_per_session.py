import pandas as pd
import datetime

"""Click per Session Prediction"""
def get_session_duration_arr(data,userkey,timekey):
    """Compute session duration for each session"""
    data[timekey] = pd.to_datetime(data[timekey])
    df = data.groupby(userkey)[timekey].agg(
        lambda x: max(x) - min(x)).to_frame().rename(columns={timekey: 'Duration'})
    df = pd.DataFrame(df)
    return df

def create_sessions(data):

    for i in range(len(data)):
        data['timestamp'].iloc[i] = datetime.datetime.strptime(data['timestamp'].iloc[i], '%Y-%m-%d %H:%M:%S')

    cond1 = data.timestamp - data.timestamp.shift(1) > pd.Timedelta(30, 'm')
    cond2 = data.cookie_id != data.cookie_id.shift(1)
    data['session_id'] = (cond1 | cond2).cumsum()

    return data

if __name__ == '__main__':

    ratings_df = pd.read_csv('/home/nick/Desktop/AB Testing Results/ratings_df_1.csv')
    unique_cookies = ratings_df['cookie_id'].unique()
    results = pd.read_csv('/home/nick/Desktop/AB Testing Results/recsys_bannerinteractions_2020_06_28_07_07.csv')
    results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
    results = results[results['timestamp'] > '2020-06-28 19:00:00']
    #results = results[(results['timestamp'] < '2020-07-02 19:00:00') | (results['timestamp'] > '2020-07-03 19:00:00')]
    #results = results[(results['timestamp'] < '2020-07-05 19:00:00') | (results['timestamp'] > '2020-07-06 19:00:00')]
    results = results[results['timestamp'] < '2020-06-29 19:00:00']

    results = results[results['cookie_id'].isin(unique_cookies)]

    print(results)

    ml_alg_df = results[results['source'] == 1]
    default_alg_df = results[results['source'] == 0]
    print(ml_alg_df.shape, default_alg_df.shape)

    ml_alg_df.sort_values(by=['cookie_id','timestamp'],inplace=True)
    default_alg_df.sort_values(by=['cookie_id','timestamp'],inplace=True)

    ml_alg_df = ml_alg_df[['cookie_id','banner_id','banner_pos','timestamp','event_type']]
    default_alg_df = default_alg_df[['cookie_id','banner_id','banner_pos','timestamp','event_type']]

    print(ml_alg_df,default_alg_df)

    ml_alg_df = create_sessions(ml_alg_df)
    default_alg_df = create_sessions(default_alg_df)

    ml_alg_df = pd.DataFrame(ml_alg_df)
    ml_alg_df.drop_duplicates(['banner_pos','event_type','session_id'],inplace=True)
    print(ml_alg_df)
    ml_alg_df.to_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_06_29_source1.csv')

    default_alg_df = pd.DataFrame(default_alg_df)
    default_alg_df.drop_duplicates(['banner_pos', 'event_type', 'session_id'], inplace=True)
    print(default_alg_df)
    default_alg_df.to_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_06_29_source0.csv')

    print(ml_alg_df['session_id'].nunique())
    print(default_alg_df['session_id'].nunique())
