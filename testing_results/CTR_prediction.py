import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


ratings_df = pd.read_csv('/home/nick/Desktop/AB Testing Results/ratings_df_05_07.csv')
unique_cookies = ratings_df['cookie_id'].unique()
#print(unique_cookies)

results = pd.read_csv('/home/nick/Desktop/AB Testing Results/banner_interactions_2020_06_19.csv')
results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
print(results.shape)
results.drop_duplicates(inplace=True)
print(results.shape)

#results = results[results['timestamp'] > '2020-06-28 19:00:00']
#results = results[(results['timestamp'] < '2020-07-02 19:00:00') | (results['timestamp'] > '2020-07-03 19:00:00')]
#results = results[(results['timestamp'] < '2020-07-05 19:00:00') | (results['timestamp'] > '2020-07-06 19:00:00')]
#results = results[results['timestamp'] < '2020-07-07 19:00:00']

#(results['timestamp'] < '2020-07-02 19:00:00')]

#print(results['banner_id'].value_counts())
print(results.shape)
print(results)

"""filter banner interactions of non-existing cookies"""
results = results[results['cookie_id'].isin(unique_cookies)]
print(results.shape)
#print(results)

ml_alg_df = results[results['source']==1]
default_alg_df = results[results['source']==0]
print(ml_alg_df.shape,default_alg_df.shape)

index_num0 = default_alg_df['banner_pos'].nunique()
index_num1 = ml_alg_df['banner_pos'].nunique()
print(index_num0, index_num1)
ml_alg_df.to_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_07_source1.csv')
default_alg_df.to_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_07_source0.csv')

"""Click Per View Prediction"""
#SOURCE = 1
click_events = ml_alg_df[ml_alg_df['event_type']=='click']
#print(click_events)
print('Till now we have {} click events on Banners!'.format(len(click_events)))

#print(click_events['banner_pos'].value_counts())

banner_pos_stats = ml_alg_df.groupby(['banner_pos','event_type']).event_type.count().unstack(fill_value=0).stack()
print(banner_pos_stats)

banner_id_stats = ml_alg_df.groupby(['banner_id','event_type']).event_type.count().unstack(fill_value=0).stack()
print(banner_id_stats)
print(banner_id_stats.index.values)
print(len(banner_pos_stats),len(banner_id_stats))
#banner_pos_stats['source'] = 1
#banner_pos_stats.to_csv('/home/nick/Desktop/AB Testing Results/banner_pos_stats_source1.csv')
#banner_id_stats.to_csv('/home/ubuntu/django-api/testing_results/banner_id_stats.csv')


ctr_per_position = []
ctr_per_id = []
for i in range(int(len(banner_pos_stats)/2)):
    ctr_per_position.append(banner_pos_stats.loc[i+1,'click'] / (banner_pos_stats.loc[i+1,'click']+banner_pos_stats.loc[i+1,'view']))
for i in range(0,len(banner_id_stats),2):
    ctr_per_id.append(banner_id_stats.iloc[i] / (banner_id_stats.iloc[i] + banner_id_stats.iloc[i+1]))

banner_id_stats = pd.DataFrame(banner_id_stats)
banner_id_stats.reset_index(inplace=True)
banner_ids = banner_id_stats['banner_id'].unique()
print(banner_ids)

ctr_per_position_1 = pd.DataFrame(ctr_per_position)
ctr_per_id_1 = pd.DataFrame(ctr_per_id)
ctr_per_position_1.columns = ['CTR (%)']
ctr_per_id_1.columns = ['CTR (%)']
ctr_per_id_1.index = banner_ids
ctr_per_id_1.reset_index(inplace=True)
print(ctr_per_id_1)
ctr_per_id_1.rename(columns={'index':'Banner ID'},inplace=True)
ctr_per_id_1['source'] = 1

ctr_per_position_1.set_index(pd.Index(np.arange(index_num1) + 1), inplace=True)
ctr_per_position_1.reset_index(inplace=True)
ctr_per_position_1.rename(columns={'index':'Banner Position'},inplace=True)
ctr_per_position_1['source'] = 1

print("Click-Through-Rate per banner position:\n {}".format(ctr_per_position_1))
print("Click-Through-Rate per banner ID:\n {}".format(ctr_per_id_1))

"""SOURCE = 0"""
click_events = default_alg_df[default_alg_df['event_type']=='click']
print(click_events)
print('Till now we have {} click events on Banners!'.format(len(click_events)))

#print(click_events['banner_pos'].value_counts())

banner_pos_stats = default_alg_df.groupby(['banner_pos','event_type']).event_type.count().unstack(fill_value=0).stack()
print(banner_pos_stats)

banner_id_stats = default_alg_df.groupby(['banner_id','event_type']).event_type.count().unstack(fill_value=0).stack()
print(banner_id_stats)
print(banner_id_stats.index.values)
print(len(banner_pos_stats),len(banner_id_stats))


ctr_per_position = []
ctr_per_id = []
for i in range(int(len(banner_pos_stats)/2)):
    ctr_per_position.append(banner_pos_stats.loc[i+1,'click'] / (banner_pos_stats.loc[i+1,'click']+banner_pos_stats.loc[i+1,'view']))
for i in range(0,len(banner_id_stats),2):
    ctr_per_id.append(banner_id_stats.iloc[i] / (banner_id_stats.iloc[i] + banner_id_stats.iloc[i+1]))

banner_id_stats = pd.DataFrame(banner_id_stats)
banner_id_stats.reset_index(inplace=True)
banner_ids = banner_id_stats['banner_id'].unique()
print(banner_ids)


ctr_per_position_0 = pd.DataFrame(ctr_per_position)
ctr_per_id_0 = pd.DataFrame(ctr_per_id)
ctr_per_position_0.columns = ['CTR (%)']
ctr_per_id_0.columns = ['CTR (%)']
ctr_per_id_0.index = banner_ids
ctr_per_id_0.reset_index(inplace=True)
ctr_per_id_0.rename(columns={'index':'Banner ID'},inplace=True)
ctr_per_id_0['source'] = 0

ctr_per_position_0.set_index(pd.Index(np.arange(index_num0) + 1), inplace=True)
ctr_per_position_0.reset_index(inplace=True)
ctr_per_position_0.rename(columns={'index':'Banner Position'},inplace=True)
ctr_per_position_0['source'] = 0
print("Click-Through-Rate per banner position:\n {}".format(ctr_per_position_0))
print("Click-Through-Rate per banner ID:\n {}".format(ctr_per_id_0))



banner_position_ctr = pd.concat([ctr_per_position_0,ctr_per_position_1],ignore_index=False)
banner_position_ctr = banner_position_ctr.groupby(['Banner Position','source']).sum().reset_index()
# teliko = teliko.groupby('Banner Position').plot
banner_position_ctr['CTR (%)'] = banner_position_ctr['CTR (%)'] * 100
banner_position_ctr = banner_position_ctr[:20]
print(banner_position_ctr)

banner_id_ctr = pd.concat([ctr_per_id_0,ctr_per_id_1],ignore_index=False)
banner_id_ctr = banner_id_ctr.groupby(['Banner ID','source']).sum().reset_index()
banner_id_ctr['CTR (%)'] = banner_id_ctr['CTR (%)'] * 100
# banner_id_ctr = banner_id_ctr[:20]
print(banner_id_ctr)
banner_id_ctr = banner_id_ctr[banner_id_ctr['CTR (%)'] != 0]
#banner_id_ctr.to_csv('/home/nick/Desktop/AB Testing Results/banner_id_ctr_day5.csv')

sns.set(style="darkgrid")
#sns.barplot(x='Banner ID',y='CTR (%)',hue='source',data=banner_id_ctr,palette="Set1")
sns.barplot(x='Banner Position',y='CTR (%)',hue='source',data=banner_position_ctr,palette="rocket")
plt.show()


