import pandas as pd
import numpy as np


class BannerRec:
    def __init__(self,userkey,itemkey,timekey,bannerkey):
        print("BannerRec Constructed!\n")
        self.userkey = userkey
        self.itemkey = itemkey
        self.timekey = timekey
        self.bannerkey = bannerkey

    def get_top_banners(self,top_products_arr, banner_product_df, N=10):

        product_list = top_products_arr

        bannerIDs = pd.Series(banner_product_df[self.bannerkey].unique())
        products_per_banner = banner_product_df[self.bannerkey].value_counts()

        rows_list = []

        for i in range(banner_product_df[self.bannerkey].nunique()):
            banner_product_list = banner_product_df[banner_product_df[self.bannerkey] == bannerIDs[i]]
            arr1 = np.array(banner_product_list[self.itemkey])
            arr2 = np.array(product_list)
            common_products = list(set(arr1).intersection(set(arr2)))
            print(common_products)
            dict = {}
            dict.update({self.bannerkey: bannerIDs[i], 'product_included': len(common_products)/products_per_banner.loc[bannerIDs[i]]})
            # top_banners.append({'banner_id':banner_ids[i], 'product_included': len(common_products)},ignore_index=True)
            rows_list.append(dict)

        top_banners = pd.DataFrame(rows_list)
        top_banners.sort_values(by=['product_included','banner_id'], ascending=False, inplace=True)
        top_banners = top_banners.head(N)

        return top_banners
