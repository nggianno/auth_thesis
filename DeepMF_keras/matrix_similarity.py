import pandas as pd
import numpy as np

df1 = pd.read_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/api-data/rnn-data/top1-top20.csv',index_col=[0])
df2 = pd.read_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/api-data/rnn-data/top1-top20_2.csv',index_col=[0])

print(df1,df2)
corr_vector = df1.corrwith(df2, axis = 0)
print(corr_vector)

print(len(df1.columns))
print(df1.columns[1])

arr_comm = []
#print(df1['0'])
for i in range(len(df1.columns)):

    # arr1 = np.array(df1[str(i)])
    # arr2 = np.array(df2[str(i)])
    arr1 = np.array(df1[df1.columns[i]])
    arr2 = np.array(df2[df2.columns[i]])
    print(arr1,arr2)
    common_products = list(set(arr1).intersection(set(arr2)))
    arr_comm.append(len(common_products))


print(arr_comm)

arr_comm = pd.DataFrame(arr_comm)
#print(arr_comm)
print(arr_comm[0].value_counts())