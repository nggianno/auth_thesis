from django.shortcuts import render
from datetime import date
import csv
import pandas as pd
import numpy as np
import MySQLdb as dbapi
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.renderers import JSONRenderer
from rest_framework import status
from scipy.sparse import csr_matrix
from recsys.models import ProductInteractions, BannerInteractions, BannerProduct, BannerStatus
from recsys.serializers import ProductInteractionsSerializer, BannerInteractionsSerializer, BannerProductSerializer, BannerStatusSerializer
from implicit_mf import build_models
from bannerrec import topn_banners
from poisson_factorization import PF
from lightfm_models.lightfm import LightFactorizationMachines



class BannerRec(APIView):

    """View to list all banners related to user which will be requested"""

    def get(self,request,user_id='default',format=None):


        banner_product_df = pd.read_csv('/home/ubuntu/django-api/datasets/recsys_bannerproduct.csv')
        banner_product_df.drop_duplicates(inplace=True)
        banner_product_df['product_id'] = banner_product_df['product_id'].astype(str)
        #banner_product_df['banner_id'] = banner_product_df['banner_id'].astype(int)

        db=dbapi.connect(host='localhost',user='nick_gianno',passwd='2020Nick@#$2020',)

        #QUERY = 'SELECT banner_id,product_id FROM nick_gianno.recsys_bannerproduct;'
        cur = db.cursor()
        #cur.execute(QUERY)
        #banner_product_df = cur.fetchall()
        #banner_product_df = pd.DataFrame(banner_product_df)
        #banner_product_df.columns = ['banner_id','product_id']
        #banner_product_df.drop_duplicates(inplace=True)
        #banner_product_df['product_id'] = banner_product_df['product_id'].astype(str)
        #banner_product_df['banner_id'] = banner_product_df['banner_id'].astype(int)


        today = date.today()
        cur_date = today.strftime("%Y-%m-%d")
        #cur.execute("SELECT distinct banner_id,status FROM nick_gianno.recsys_bannerstatus where last_update > '{}' and status=1;".format(cur_date))
        cur.execute("SELECT distinct banner_id FROM nick_gianno.recsys_bannerstatus where last_update > '{}' and status = 1;".format(cur_date))
        active_banners_df = cur.fetchall()
        active_banners_df = pd.DataFrame(active_banners_df)
        #active_banners_df.drop(columns=0,inplace=True)
        active_banners_df.columns = ['banner_id']
        #active_banners = pd.DataFrame(active_banners_df['banner_id'])
        active_banners_df['banner_id']=active_banners_df['banner_id'].astype(int)

        #active_banners_df = pd.read_csv('/home/ubuntu/django-api/datasets/recsys_bannerstatus.csv')

        banner_product_df = banner_product_df.merge(active_banners_df,on='banner_id')
        #banner_product_df = banner_product_df.loc[:, ~banner_product_df.columns.str.contains('^Unnamed')]
        cur.execute("select exists(select * from nick_gianno.ratings where cookie_id = {});".format(user_id))
        #cur.execute("select exists(select * from nick_gianno.gru_top_products where cookie_id = {});".format(user_id))
        #cur.execute("select exists(select * from nick_gianno.neural_cf_top_products where cookie_id = {});".format(user_id))
        found = cur.fetchall()
        found = pd.DataFrame(found)
        found = found[0].loc[0]

        path = '/home/ubuntu/django-api/datasets/ratings_df.csv'
        #ratings_df = pd.read_csv(path)
        #ratings_df = ratings_df.loc[:, ~ratings_df.columns.str.contains('^Unnamed')]
        #unique_cookies = ratings_df['cookie_id'].unique()

        #test_data = pd.read_csv('/home/ubuntu/django-api/gru4rec/testset.csv')
        #itemidmap = pd.read_csv('/home/ubuntu/django-api/gru4rec/item_lookup.csv')
        #itemidmap = itemidmap.loc[:,~itemidmap.columns.str.contains('^Unnamed')]
        #unique_cookies = test_data['cookie_id'].unique()
        #if int(user_id) in unique_cookies:

        if found:

            """GRU4REC
            #topn_df = pd.read_csv('/home/ubuntu/django-api/gru4rec/top_products_gru.csv')
            #topn_df = topn_df.loc[:, ~topn_df.columns.str.contains('^Unnamed')]
            #topn_vec = pd.DataFrame(topn_df[user_id])
            #topn_vec.rename(columns={user_id: 'idx'}, inplace=True)
            #topn_vec = topn_vec.merge(itemidmap, on='idx')
            #top_products = topn_vec['product_id']
            cur.execute("select p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20 from nick_gianno.gru_top_products where cookie_id={}".format(user_id))
            top_products = cur.fetchall()
            top_products = pd.DataFrame(top_products)
            top_products = np.array(top_products.iloc[0])"""


            """Neural Collaborative Filtering

            cur.execute("select p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20 from nick_gianno.neural_cf_top_products where cookie_id={}".format(user_id))
            top_products = cur.fetchall()
            top_products = pd.DataFrame(top_products)
            top_products = top_products.loc[0]"""

            #top20_df = pd.read_csv('/home/ubuntu/django-api/neural_cf/top20products.csv')
            #top20_df = top20_df.loc[:, ~top20_df.columns.str.contains('^Unnamed')]


            #useridmap = pd.read_csv('/home/ubuntu/django-api/neural_cf/user_lookup.csv')
            #itemidmap = pd.read_csv('/home/ubuntu/django-api/neural_cf/item_lookup.csv')


            #useridmap.set_index('user',inplace=True)
            #useridx = useridmap.loc[int(user_id),'user_id']
            #print(useridx)
            #topk_vec = pd.DataFrame(top20_df[str(useridx)])
            #topk_vec.rename(columns={str(useridx): 'item_id'}, inplace=True)

            #itemidmap.reset_index(inplace=True)
            #topk_vec = topk_vec.merge(itemidmap, on='item_id')
            #top_products = topk_vec['item']


            """LightFm
            lightfm_obj = LightFactorizationMachines(userkey='cookie_id',itemkey='product_id')
            model, top_products, user_item_csr = lightfm_obj.run_lightfm\
                (method='warp',user_id=int(user_id),lr=0.01,epochs=50,training=False)
            #top_products = list(map(int, top_products))"""

            """Poisson Matrix Factorization
            pf_obj = PF.PoissonFactorization(userkey='cookie_id', itemkey='product_id')
            data = pf_obj.build_dataset(path)
            #top_products = pf_obj.HPF(data,int(user_id),train=True)
            #top_products = pf_obj.PMF(data, int(user_id), train=False) --Bad results"""

            """Implicit ALS/BPR/LMF/ITEM_KNN(DONE)"""
            implicit_obj = build_models.Implicit(userkey='cookie_id', itemkey='product_id')
            top_products = implicit_obj.get_recommendations(model_name='bpr', user_id=int(user_id), training=False)
            print('Top Products:\n{}'.format(top_products))

        else:

            """show top seller products"""
            #cur.execute("select product_id from nick_gianno.ratings group by product_id order by count(rating) DESC limit 20;")
            #top_products = cur.fetchall()
            #top_products = pd.DataFrame(top_products)
            #top_products = top_products[0]
            top_products = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
            #ratings_df.sort_values(by='rating', inplace=True, ascending=False)
            #product_popularity = ratings_df.groupby('product_id').rating.count()
            #product_popularity.sort_values(ascending=False, inplace=True)
            #top_products = product_popularity.index.values
            #top_products = top_products[:20]

        """Relate products with banners"""
        banner_obj = topn_banners.BannerRec(userkey = 'cookie_id',itemkey = 'product_id',timekey = 'timestamp',bannerkey='banner_id')
        banner_recommendations = banner_obj.get_top_banners(np.array(top_products),banner_product_df)
        top10banners = banner_recommendations['banner_id']

#        if found:
#            pass
#        else:
#            top10banners = [1377,1378,1380,1381,1361,1376,1375,1373,1374]

        return Response(top10banners,status=status.HTTP_200_OK)




"""Export json data stored in tables in csv files"""

def export_product_interactions(request):

    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)
    writer.writerow(['user_id', 'product_id', 'cookie_id','timestamp','event_type'])

    for record in ProductInteractions.objects.all().values_list('user_id', 'product_id','cookie_id', 'timestamp','event_type'):
        writer.writerow(record)

    response['Content-Disposition'] = 'attachment; filename="product_interactions.csv"'

    return response


def export_banner_product(request):

    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)
    writer.writerow(['banner_id', 'product_id'])

    for record in BannerProduct.objects.all().values_list('banner_id', 'product_id'):
        writer.writerow(record)

    response['Content-Disposition'] = 'attachment; filename="banner_product.csv"'

    return response


def export_banner_interactions(request):

    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)
    writer.writerow(['user_id','cookie_id','banner_id','banner_pos','timestamp','event_type','source'])

    for record in BannerInteractions.objects.all().values_list('user_id','cookie_id','banner_id','banner_pos','timestamp','event_type','source'):

        writer.writerow(record)

    response['Content-Disposition'] = 'attachment; filename="banner_interactions.csv"'

    return response


def export_banner_status(request):

    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)
    writer.writerow(['banner_id', 'status','last_update'])

    for record in BannerStatus.objects.all().values_list('banner_id','status','last_update'):
        writer.writerow(record)

    response['Content-Disposition'] = 'attachment; filename="banner_status.csv"'

    return response
