from rest_framework import serializers
from recsys.models import ProductInteractions, BannerInteractions, BannerProduct, BannerStatus

class ProductInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ProductInteractions
        fields = ('user_id', 'product_id','cookie_id','timestamp','event_type')

class BannerInteractionsSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = BannerInteractions
        fields = ('user_id','cookie_id','banner_id','banner_pos','timestamp','event_type','source')

class BannerProductSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = BannerProduct
        fields = ('banner_id', 'product_id')


class BannerStatusSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = BannerStatus
        fields = ('banner_id','status','last_update')
