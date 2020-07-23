"""djangoAPI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from recsys import viewsets, views


router = routers.DefaultRouter()
router.register(r'product-interactions', viewsets.ProductInteractionsViewSet)
router.register(r'banner-product-relations', viewsets.BannerProductViewSet)
router.register(r'banner-interactions',viewsets.BannerInteractionsViewSet)
router.register(r'banner-status',viewsets.BannerStatusViewSet)


urlpatterns = [
    path('export/product-interactions', views.export_product_interactions),
    path('export/banner-product',views.export_banner_product),
    path('export/banner-interactions',views.export_banner_interactions),
    path('export/banner-status',views.export_banner_status),
    path('', include(router.urls)),
    path('admin/', admin.site.urls),
    path('banner-rec/<slug:user_id>',views.BannerRec.as_view())
    #path('banner-rec/',views.BannerRec.as_view())
    #path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
