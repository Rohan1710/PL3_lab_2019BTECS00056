from django.db import models

# Create your models here.

class Form(models.Model):
    name = models.CharField(max_length=122)
    age = models.CharField(max_length=4)
    phone = models.CharField(max_length= 22)
    address = models.CharField(max_length=144)
    payment_status = models.BooleanField()
    batch = models.CharField(max_length=50)
    date = models.DateField()

# class Batch(models.Model):
