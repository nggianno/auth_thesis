import numpy as np
import pandas as pd
import datetime



def compute_dwell_time(df,userkey,itemkey,timekey):
    times_t = np.roll(df[timekey], -1)  # Take time row
    times_dt = df[timekey]  # Copy, then displace by one

    diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference
    diffs = np.array(diffs)

    length = len(df[itemkey])

    # cummulative offset start for each session
    offset_sessions = np.zeros(df[userkey].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = df.groupby(userkey).size().cumsum()

    offset_sessions = offset_sessions - 1
    offset_sessions = np.roll(offset_sessions, -1)

    # session transition implies zero-dwell-time
    # note: paper statistics do not consider null entries,
    # though they are still checked when augmenting
    np.put(diffs, offset_sessions, np.zeros((offset_sessions.shape)), mode='raise')

    return diffs


def join_dwell_reps(df, dt, threshold=2000):
    # Calculate d_ti/threshold + 1
    # then add column to dataFrame

    dt //= threshold
    dt += 1
    df['DwellReps'] = pd.Series(dt.astype(np.int64), index=dt.index)

    return df

def get_session_duration_arr(data,userkey,timekey):
    """Compute session duration for each session"""
    data.event_time = pd.to_datetime(data.event_time)
    df = data.groupby(userkey)[timekey].agg(
        lambda x: max(x) - min(x)).to_frame().rename(columns={timekey: 'Duration'})
    df = pd.DataFrame(df)
    return df

if __name__ == "__main__":

    #PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/brand_dataset.csv'
    PATH = '/home/ubuntu/django-api/datasets/rnn_df.csv'
    userkey = 'cookie_id'
    itemkey = 'product_id'
    timekey = 'timestamp'

    df = pd.read_csv(PATH)
    #df.drop(columns=['Unnamed: 0'],inplace=True)
    print(df)
    df.sort_values(by=[userkey,timekey],inplace=True)
    df[timekey] = pd.to_datetime(df[timekey])
    df[timekey]=df[timekey].dt.tz_localize(None)
    print(df)
    # df[timekey] = datetime.datetime.strptime()
    # print(df)
    """Compute dwelltime"""
    dw_t = compute_dwell_time(df,userkey,itemkey,timekey)
    dw_t = pd.DataFrame(dw_t)

    final_df = df

    print(final_df,dw_t)


    dw_t.rename(columns={0:timekey},inplace=True)
    print(dw_t)

    """The last clickstream info of every session has invalid dwelltime due to the substraction of 2 different time periods
    So we will clean these outliers and will take the average dwelltime of the session instead"""

    """Calculate the average dwelltime of the records, ignoring outliers"""
    cleaned_dwt = dw_t[(dw_t[timekey] != '0 days 00:00:00') & (dw_t[timekey] < '0 days 01:00:00')]
    # #cleaned_dwt = dw_t[dw_t[timekey] != '0 days 00:00:00']
    cleaned_dwt.rename(columns = {"timestamp": "dwelltime"},inplace=True)
    print(cleaned_dwt)

    av_dwt = cleaned_dwt['dwelltime'].mean()
    print(av_dwt)

    """Replace last click duration of each session with the average dwelltime"""
    dw_t.loc[(dw_t[timekey] == '0 days 00:00:00') ^ (dw_t[timekey] >= '0 days 01:00:00')] = av_dwt
    #dw_t['timestamp'].apply(lambda x: av_dwt if x == '00:00:00' else x)
    dw_t.rename(columns = {"timestamp": "dwelltime"},inplace=True)
    # print(type(dw_t['dwelltime'][2]))
    print(dw_t)
    #print(final_df)
    final_df['dwelltime'] = dw_t['dwelltime']
    final_df = pd.DataFrame(final_df)
    dwelltime_freqs = pd.DataFrame(final_df['dwelltime'].value_counts())

    print(final_df,dwelltime_freqs)

    #Extract to csv
    final_df.to_csv('/home/ubuntu/django-api/datasets/rnn_df_dwelltime.csv')

