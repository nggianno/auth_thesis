import pandas as pd
import numpy as np

"""
topn_df = pd.read_csv('/home/ubuntu/django-api/neural_cf/top20products.csv')
topn_df = topn_df.loc[:, ~topn_df.columns.str.contains('^Unnamed')]
useridmap = pd.read_csv('/home/ubuntu/django-api/neural_cf/user_lookup.csv')
itemidmap = pd.read_csv('/home/ubuntu/django-api/neural_cf/item_lookup.csv')
print(topn_df)

#useridmap.set_index('user',inplace=True)
useridmap = useridmap.loc[:, ~useridmap.columns.str.contains('^Unnamed')]
cookies = np.array(useridmap['user'])
topn_df.columns = cookies

for i in range(len(cookies)):

    topn_vec = pd.DataFrame(topn_df.iloc[:,i])
    topn_vec.columns = ['item_id']
    topn_vec = topn_vec.merge(itemidmap, on='item_id')
    top_products = topn_vec['item']
    topn_df.iloc[:, i] = top_products

topn_df = pd.DataFrame(topn_df)
topn_df.to_csv('/home/ubuntu/django-api/neural_cf/top_products.csv')
"""
topn_df = pd.read_csv('/home/ubuntu/django-api/neural_cf/top_products.csv')
topn_df = topn_df.loc[:, ~topn_df.columns.str.contains('^Unnamed')]
topn_df = topn_df.T
print(topn_df)
topn_df.to_csv('/home/ubuntu/django-api/neural_cf/top_products_t.csv')



