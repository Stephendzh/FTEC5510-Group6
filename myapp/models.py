from django.db import models

# Create your models here.
from django.db import models

# Create your models here.

class userinfo(models.Model):
    Email = models.CharField(max_length=32)
    Password = models.CharField(max_length=64)