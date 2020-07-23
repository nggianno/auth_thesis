import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""Algorithm Comparison Plot"""
# d = {'Algorithm': ['ALS CF','ALS CF','BPR CF','BPR CF','NCF','NCF','GRU4Rec','GRU4Rec'], 'source': [0,1,0,1,0,1,0,1],
#      'Overall CTR (%)': [0.44,0.48,0.65,0.61,0.39,0.82,0.37,0.43]}
"""CTR/Session"""
d = {'Day': ['1','1','2','2','3','3','4','4','5','5','6','6','7','7'], 'source': [0,1,0,1,0,1,0,1,0,1,0,1,0,1],
     'CTR/Session (%)': [9.52,8.65,17.89,11.60,15.54,17.30,10.43,11.21,11.15,11.04,11.53,11.79,11.15,12.31]}
"""CTR/View"""
# d = {'Day': ['1','1','2','2','3','3','4','4','5','5','6','6','7','7'], 'source': [0,1,0,1,0,1,0,1,0,1,0,1,0,1],
#      'CTR/View (%)': [0.44,0.48,0.65,0.61,0.39,0.82,0.37,0.43,0.64,0.37,0.51,0.59,0.17,0.22]}
# #'Overall_CTR': [0.004051,0.004414,0.006373,0.006183,0.004941,0.008618]
ovrl_ctr = pd.DataFrame(data=d)
print(ovrl_ctr)
# day2 = pd.read_csv('/home/nick/Desktop/AB Testing Results/banner_id_ctr_day2.csv')
# day5 = pd.read_csv('/home/nick/Desktop/AB Testing Results/banner_id_ctr_day5.csv')

# df0 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_05_source0.csv')
# df1 = pd.read_csv('/home/nick/Desktop/AB Testing Results/views_dataframes/df_07_05_source1.csv')
#
# day2 = day2.loc[:, ~day2.columns.str.contains('^Unnamed')]
# day5 = day5.loc[:, ~day5.columns.str.contains('^Unnamed')]
# print(day2,day5)
#
#
# print(np.arange(9)+1)

sns.set(style="darkgrid")
sns.barplot(x='Day',y='CTR/Session (%)',hue='source',data=ovrl_ctr,palette="rocket")
plt.show()
# fig, axs = plt.subplots(ncols=2)
# sns.set(style="darkgrid")
# sns.barplot(x='Banner ID', y='CTR (%)', hue='source', data=day2, palette="Set1", ax=axs[0]).set_title(
#      'Day 2')
#
# sns.barplot(x='Banner ID', y='CTR (%)', hue='source', data=day5, palette="Set1", ax=axs[1]).set_title(
#      'Day 5')
# plt.show()