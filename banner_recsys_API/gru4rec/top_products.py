import MySQLdb as dbapi
import pandas as pd
import numpy as np

topn_df = pd.read_csv('/home/ubuntu/django-api/gru4rec/top_products_gru2.csv')
topn_df = topn_df.loc[:, ~topn_df.columns.str.contains('^Unnamed')]
print(topn_df)
itemidmap = pd.read_csv('/home/ubuntu/django-api/gru4rec/item_lookup3.csv')
itemidmap = itemidmap.loc[:, ~itemidmap.columns.str.contains('^Unnamed')]
print(itemidmap)

cookies = np.array(topn_df.columns)
print(cookies)

for i in range(len(cookies)):

    topn_vec = pd.DataFrame(topn_df.iloc[:,i])
    topn_vec.columns = ['idx']
    topn_vec = topn_vec.merge(itemidmap, on='idx')
    top_products = topn_vec['product_id']
    topn_df.iloc[:, i] = top_products

topn_df = pd.DataFrame(topn_df)
topn_df.to_csv('/home/ubuntu/django-api/gru4rec/top_products3.csv')

#get top_products transpose
#topn_df_t = pd.read_csv('/home/ubuntu/django-api/gru4rec/top_products2.csv')
#topn_df_t = topn_df_t.loc[:, ~topn_df_t.columns.str.contains('^Unnamed')]
#topn_df_t = topn_df_t.T
#print(topn_df_t)
#topn_df_t.to_csv('/home/ubuntu/django-api/gru4rec/top_products_t.csv')

#db=dbapi.connect(host='localhost',user='nick_gianno',passwd='2020Nick@#$2020')
#cur=db.cursor()
#cur.execute("select p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20 from nick_gianno.gru_top_products where cookie_id = 1002876249421387911;")
#top_products = cur.fetchall()
#top_products = pd.DataFrame(top_products)
#top_products = np.array(top_products.iloc[0])
#print(top_products)


