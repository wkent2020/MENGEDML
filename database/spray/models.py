from django.db import models

# Create your models here.



class Frame(models.Model):
	frame_index = models.IntegerField('Number of Frames',default=0)

class Spray(models.Model):
    label = models.CharField('Spray Label',max_length=100)
    date_added = models.DateTimeField('Date Published')
    num_frames = models.IntegerField('Number of Frames',default=0)