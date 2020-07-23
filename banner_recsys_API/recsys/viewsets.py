from django.shortcuts import render
import json
import pandas as pd
from django.http import HttpResponse
from django.core import exceptions
from django.http import JsonResponse,Http404
from recsys.models import ProductInteractions, BannerInteractions, BannerProduct, BannerStatus
from recsys.serializers import ProductInteractionsSerializer, BannerInteractionsSerializer, BannerProductSerializer, BannerStatusSerializer
from rest_framework import viewsets,permissions,views,status
from rest_framework.decorators import action
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from django_filters import rest_framework
from rest_framework import filters
from django_filters import FilterSet



class ProductInteractionsFilter(rest_framework.FilterSet):

    class Meta:
        model = ProductInteractions
        fields = {
            'user_id':['icontains'],
            'product_id':['icontains'],
            'timestamp':['iexact','lte','gte'],
            'event_type':['iexact']
        }

class ProductInteractionsViewSet(viewsets.ModelViewSet):

    """
    API endpoint that allows product interactions to be viewed or edited.
    """

    queryset = ProductInteractions.objects.all()
    serializer_class = ProductInteractionsSerializer
    filterset_class = ProductInteractionsFilter

    filter_backends = [rest_framework.DjangoFilterBackend]
    #filter_fields = ('user_id', 'product_id','event_type')
    #search_fields = ('user_id', 'product_id','event_type')

    #overwrite GET and return records that contain purchase as event_type
    # def get_queryset(self):
    #     return ProductInteractions.objects.filter(event_type__icontains = 'purchase')

    def list(self, request):
        queryset = ProductInteractions.objects.all()
        serializer = ProductInteractionsSerializer(queryset, many=True)
        #transform serializer data to pandas dataframe
        df = pd.DataFrame.from_dict(serializer.data)
        #print(df)
        #df.to_csv('/home/nick/Desktop/thesis/datasets/pharmacy-data/dj_user_product.csv')
        return Response(serializer.data)

    """Receive multiple POSTS"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(ProductInteractionsViewSet, self).get_serializer(*args, **kwargs)


    """DELETE all records"""

    def delete(self, request):

        queryset = ProductInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)



class BannerProductFilter(rest_framework.FilterSet):

    class Meta:
        model = BannerProduct
        fields = {
            'banner_id':['iexact'],
            'product_id':['icontains'],
        }

class BannerProductViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows banners/products to be viewed or edited.
    """
    queryset = BannerProduct.objects.all()
    serializer_class = BannerProductSerializer
    filterset_class = BannerProductFilter

    filter_backends = [rest_framework.DjangoFilterBackend]

    """Receive multiple POSTS"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerProductViewSet, self).get_serializer(*args, **kwargs)

    """DELETE all records"""
    def delete(self, request):
        queryset = BannerProduct.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)


class BannerInteractionsFilter(rest_framework.FilterSet):

    class Meta:
        model = BannerInteractions
        fields = {
            'user_id':['icontains'],
            'banner_id': ['iexact'],
            'banner_pos':['iexact'],
            'timestamp':['iexact','lte','gte'],
        }

class BannerInteractionsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows banners-interactions to be viewed or edited.
    """
    queryset = BannerInteractions.objects.all()
    serializer_class = BannerInteractionsSerializer
    filterset_class = BannerInteractionsFilter

    """Receive multiple posts"""
    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerInteractionsViewSet, self).get_serializer(*args, **kwargs)

    """DELETE all records"""
    def delete(self, request):
        queryset = BannerInteractions.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)


class BannerStatusViewSet(viewsets.ModelViewSet):

    queryset = BannerStatus.objects.all()
    serializer_class = BannerStatusSerializer

    """Receive multiple posts"""

    def get_serializer(self, *args, **kwargs):
        if self.request.method.lower() == 'post':
            data = kwargs.get('data')
            kwargs['many'] = isinstance(data, list)
        return super(BannerStatusViewSet, self).get_serializer(*args, **kwargs)

    """DELETE all records"""

    def delete(self, request):
        queryset = BannerStatus.objects.all()
        queryset.delete()
        return Response(status=status.HTTP_200_OK)
