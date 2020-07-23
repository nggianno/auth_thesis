from django.db import models

# Create your models here.
from django.db import models
from django.core.validators import RegexValidator
from django.template.defaultfilters import slugify

alphanumeric = RegexValidator(r'^[0-9a-zA-Z]*$', 'Only alphanumeric characters are allowed.')

class ProductInteractions(models.Model):

    user_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])
    product_id = models.CharField(max_length=64, blank=True, null=False, validators=[alphanumeric])
    cookie_id = models.CharField(max_length=64,blank=True,null=False,validators=[alphanumeric])
    timestamp = models.DateTimeField(auto_now_add=False,null=False)
    event_type = models.CharField(max_length=20,blank=True,null=False)

    class Meta:
        ordering = ['timestamp']


class BannerInteractions(models.Model):

    user_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])
    cookie_id = models.CharField(max_length=64,blank=True,null=False,validators=[alphanumeric])
    banner_id = models.IntegerField(blank=True, null=False)
    banner_pos = models.IntegerField(blank=True,null=False)
    timestamp = models.DateTimeField(auto_now_add=False, null=False)
    event_type = models.CharField(max_length=20,blank=True,null=False)
    source = models.IntegerField(blank=True,null=False,default=0)


    class Meta:
        ordering = ['timestamp']


class BannerProduct(models.Model):

    banner_id  = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])
    product_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])

    class Meta:
        ordering = ['banner_id']
#        unique_together = (('banner_id','product_id'))



class BannerStatus(models.Model):

    banner_id = models.CharField(max_length=64, blank=False, null=False, validators=[alphanumeric])
    status = models.CharField(max_length=16,blank=False,null=False,validators=[alphanumeric])
    last_update = models.DateTimeField(auto_now_add=False, blank=True, null=False)


    class Meta:
        ordering = ['-status']
