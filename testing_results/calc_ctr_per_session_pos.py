import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime


def get_ctr_df(df0,df1):
    """Click Per View Prediction"""
    # SOURCE = 1
    click_events = df1[df1['event_type'] == 'click']
    # print(click_events)
    print('Till now we have {} click events on Banners!'.format(len(click_events)))

    index_num0 = df0['banner_pos'].nunique()
    index_num1 = df1['banner_pos'].nunique()
    print(index_num0,index_num1)

    banner_pos_stats = df1.groupby(['banner_pos', 'event_type']).event_type.count().unstack(fill_value=0).stack()
    print(banner_pos_stats)

    banner_id_stats = df1.groupby(['banner_id', 'event_type']).event_type.count().unstack(fill_value=0).stack()
    print(banner_id_stats)
    print(banner_id_stats.index.values)
    print(len(banner_pos_stats), len(banner_id_stats))


    ctr_per_position = []
    ctr_per_id = []
    for i in range(int(len(banner_pos_stats) / 2)):
        ctr_per_position.append(banner_pos_stats.loc[i + 1, 'click'] / (
                    banner_pos_stats.loc[i + 1, 'click'] + banner_pos_stats.loc[i + 1, 'view']))
    for i in range(0, len(banner_id_stats), 2):
        ctr_per_id.append(banner_id_stats.iloc[i] / (banner_id_stats.iloc[i] + banner_id_stats.iloc[i + 1]))

    banner_id_stats = pd.DataFrame(banner_id_stats)
    banner_id_stats.reset_index(inplace=True)
    banner_ids = banner_id_stats['banner_id'].unique()
    print(banner_ids)

    ctr_per_position_1 = pd.DataFrame(ctr_per_position)
    ctr_per_id = pd.DataFrame(ctr_per_id)
    ctr_per_position_1.columns = ['CTR/Session(%)']
    ctr_per_id.columns = ['CTR/Session(%)']
    ctr_per_id.index = banner_ids
    ctr_per_position_1.set_index(pd.Index(np.arange(index_num1)+1), inplace=True)
    ctr_per_position_1.reset_index(inplace=True)
    ctr_per_position_1.rename(columns={'index': 'Banner Position'}, inplace=True)
    ctr_per_position_1['source'] = 1
    print("Click-Through-Rate per banner position:\n {}".format(ctr_per_position_1))

    # SOURCE = 0
    click_events = df0[df0['event_type'] == 'click']
    print(click_events)
    print('Till now we have {} click events on Banners!'.format(len(click_events)))

    # print(click_events['banner_pos'].value_counts())

    banner_pos_stats = df0.groupby(['banner_pos', 'event_type']).event_type.count().unstack(fill_value=0).stack()
    print(banner_pos_stats)


    banner_id_stats = df0.groupby(['banner_id', 'event_type']).event_type.count().unstack(fill_value=0).stack()
    print(banner_id_stats)
    print(banner_id_stats.index.values)
    print(len(banner_pos_stats), len(banner_id_stats))
    # banner_pos_stats.to_csv('/home/ubuntu/django-api/testing_results/banner_position_stats.csv')
    # banner_id_stats.to_csv('/home/ubuntu/django-api/testing_results/banner_id_stats.csv')

    ctr_per_position = []
    ctr_per_id = []
    for i in range(int(len(banner_pos_stats) / 2)):
        ctr_per_position.append(banner_pos_stats.loc[i + 1, 'click'] / (
                    banner_pos_stats.loc[i + 1, 'click'] + banner_pos_stats.loc[i + 1, 'view']))
    for i in range(0, len(banner_id_stats), 2):
        ctr_per_id.append(banner_id_stats.iloc[i] / (banner_id_stats.iloc[i] + banner_id_stats.iloc[i + 1]))


    banner_id_stats = pd.DataFrame(banner_id_stats)
    banner_id_stats.reset_index(inplace=True)
    banner_ids = banner_id_stats['banner_id'].unique()
    print(banner_ids)

    ctr_per_position_0 = pd.DataFrame(ctr_per_position)
    ctr_per_id = pd.DataFrame(ctr_per_id)
    ctr_per_position_0.columns = ['CTR/Session(%)']
    ctr_per_id.columns = ['CTR/Session(%)']
    ctr_per_id.index = banner_ids
    ctr_per_position_0.set_index(pd.Index(np.arange(index_num0)+1), inplace=True)
    #ctr_per_position_0.set_index(pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9]), inplace=True)
    ctr_per_position_0.reset_index(inplace=True)
    ctr_per_position_0.rename(columns={'index': 'Banner Position'}, inplace=True)
    ctr_per_position_0['source'] = 0
    print("Click-Through-Rate per banner position:\n {}".format(ctr_per_position_0))
    print("Click-Through-Rate per banner ID:\n {}".format(ctr_per_id))

    # teliko = ctr_per_position_0.merge(ctr_per_position_1,on='Banner Position')
    teliko = pd.concat([ctr_per_position_0, ctr_per_position_1], ignore_index=False)
    teliko = teliko.groupby(['Banner Position', 'source']).sum().reset_index()
    # teliko = teliko.groupby('Banner Position').plot
    teliko['CTR/Session(%)'] = teliko['CTR/Session(%)'] * 100
    print(teliko)
    teliko = teliko[:20]

    return teliko

if __name__ == '__main__':

    # df0_1 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_06_29_source0.csv')
    # df1_1 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_06_29_source1.csv')
    # df0_2 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_06_30_source0.csv')
    # df1_2 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_06_30_source1.csv')
    # df0_3 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_01_source0.csv')
    # df1_3 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_01_source1.csv')
    # df0_4 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_02_source0.csv')
    # df1_4 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_02_source1.csv')
    # df0_5 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_04_source0.csv')
    # df1_5 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_04_source1.csv')
    # df0_6 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_05_source0.csv')
    # df1_6 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_05_source1.csv')
    # df0_7 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_07_source0.csv')
    # df1_7 = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_07_07_source1.csv')
    # df0_1 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_06_29_source0.csv')
    # df1_1 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_06_29_source1.csv')
    # df0_2 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_06_30_source0.csv')
    # df1_2 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_06_30_source1.csv')
    # df0_3 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_01_source0.csv')
    # df1_3 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_01_source1.csv')
    # df0_4 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_02_source0.csv')
    # df1_4 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_02_source1.csv')
    # df0_5 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_04_source0.csv')
    # df1_5 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_04_source1.csv')
    # df0_6 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_05_source0.csv')
    # df1_6 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_05_source1.csv')
    # df0_7 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_07_source0.csv')
    # df1_7 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_07_source1.csv')

    df0_7days = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_7days_source0.csv')
    df1_7days = pd.read_csv('/home/nick/Desktop/AB Testing Results/session_dataframes/session_df_7days_source1.csv')

    df0_7days = df0_7days.loc[:, ~df0_7days.columns.str.contains('^Unnamed')]
    df1_7days = df1_7days.loc[:, ~df1_7days.columns.str.contains('^Unnamed')]



    teliko = get_ctr_df(df0_7days,df1_7days)
    print(teliko)
    #
    # teliko1= get_ctr_df(df0_1,df1_1)
    # teliko2 = get_ctr_df(df0_2, df1_2)
    # teliko3 = get_ctr_df(df0_3, df1_3)
    # teliko4 = get_ctr_df(df0_4, df1_4)
    # teliko5 = get_ctr_df(df0_5, df1_5)
    # teliko6 = get_ctr_df(df0_6, df1_6)
    # teliko7 = get_ctr_df(df0_7,df1_7)


    # fig, axs = plt.subplots(nrows=3,ncols=3)
    sns.set(style="darkgrid")
    sns.barplot(x='Banner Position', y='CTR/Session(%)', hue='source', data=teliko, palette="rocket")

    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko1,palette="rocket",ax=axs[0,0]).set_title('Day 1')
    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko2,palette="rocket",ax=axs[0,1]).set_title('Day 2')
    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko3,palette="rocket",ax=axs[0,2]).set_title('Day 3')
    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko4,palette="rocket",ax=axs[1,0]).set_title('Day 4')
    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko5,palette="rocket",ax=axs[1,1]).set_title('Day 5')
    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko6,palette="rocket",ax=axs[1,2]).set_title('Day 6')
    # sns.barplot(x='Banner Position',y='CTR/Session(%)',hue='source',data=teliko7,palette="rocket",ax=axs[2,1]).set_title('Day 7')


    plt.show()
    # frames = [teliko1,teliko2,teliko3,teliko4,teliko5,teliko6]
    # result = pd.concat(frames)
    # print(result)