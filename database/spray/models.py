from django.db import models
from datetime import datetime    

# Create your models here.

class Spray(models.Model):
    label = models.CharField('Spray Label',max_length=100)
    date_added = models.DateTimeField('Date Published',default=datetime.now)
    num_frames = models.IntegerField('Number of Frames',default=0)
    cropping = models.IntegerField('Cropping Value', default=0)
    xdim = models.IntegerField('X-Dimension', default=0)
    ydim = models.IntegerField('Y-Dimension', default=0)
    light_energy = models.DecimalField('X-ray Energy', decimal_places=2, max_digits=100,default=0)
    alpha = models.DecimalField('Alpha Coefficient', decimal_places=20, max_digits=100,default=0)
    norm = models.CharField(max_length=100, choices=[('SB', 'Single background'),('DBM','Double background by mean'),('DBB','Double background by background')],default='SB')
    # doubledim = contains four number 
    num_frames_avg = models.IntegerField('Number of Background Frames for Averaging', default=5)
    backg_image_index = models.IntegerField('Last background frame', default=10)
    # backg_section = models.ListCharField(IntegerField,size=0)
    max_pixel = models.IntegerField('Maximum Pixel Value', default=0)
    min_pixel = models.IntegerField('Minimum Pixel Value', default=0)
    rescaling_range = models.CharField(max_length=100, choices=[('FL', 'Floats'),('8B','8-bit'),('16B','16-bit')], default='8B')
    # thresholding_values = ListCharField(IntegerField,size=0)
    # Average Nozzle corrdinate =  ListCharField(IntegerField,size=0)
    # Average Nozzle angle =  ListCharField(IntegerField,size=0)



class Frame(models.Model):
	frame_index = models.IntegerField('Frame index',default=0)
	date_added = models.DateTimeField('Date Published',default=datetime.now)
	frame_angle = models.DecimalField('Spray angle', decimal_places=3, max_digits=10, default=0)
	num_droplets = models.IntegerField('Number of droplets',default=0)
	mean_pixel = models.DecimalField('Mean pixel value', decimal_places=3, max_digits=10,default=0)
	std_pixel = models.DecimalField('Standard deviation pixel value', decimal_places=3, max_digits=10,default=0)
	# thresholding_values = ListCharField(IntegerField,size=0)
	# thresholding_values = models.BinaryField()
	# image_raw = field relating to picture itself
	# image_norm = field relating to normalized image
	# image_binary = field relating to binary image
	# droplet_contours = array of contours
	# Nozzle corrdinate =  ListCharField(IntegerField,size=0)
	# Nozzle angle =  ListCharField(IntegerField,size=0)
	# number_density = ListCharField(IntegerField,size=0)
	# mass_density = ListCharField(IntegerField,size=0)
	# area_density = ListCharField(IntegerField,size=0)
	# dispersity_density_cart = ListCharField(IntegerField,size=0)
	# dispersity_density_polar = ListCharField(IntegerField,size=0)


class Droplets(models.Model):
	label = models.CharField('Droplet Label',max_length=100,default='None')
	spray_label = models.CharField('Droplet Label',max_length=100,default='Unknown')
	# droplet_contour = models.ArrayField()
	# 
	frame_index = models.CharField('Droplet Label',max_length=100,default='0')
	date_added = models.DateTimeField('Date Published')
	outer_perimeter = models.DecimalField('Outer Perimeter', decimal_places=2, max_digits=15,default=0)
	inner_perimeter = models.DecimalField('Inner Perimeter', decimal_places=2, max_digits=15,default=0)
	row_center_loc = models.IntegerField('Row center location', default = 0)
	col_center_loc = models.IntegerField('Column center location', default = 0)
	row_coa = models.IntegerField('Row center of area', default = 0)
	col_coa = models.IntegerField('Column center of area', default = 0)
	mom_iner = models.DecimalField('Moment of Inertia', decimal_places=5, max_digits=15,default=0)
	rg = models.DecimalField('Radius of gyration', decimal_places=5, max_digits=15,default=0)
	density = models.DecimalField('Density', decimal_places=2, max_digits=15,default=0)
	area = models.DecimalField('Area', decimal_places=2, max_digits=15,default=0)
	mass = models.DecimalField('Mass', decimal_places=2, max_digits=15,default=0)
	std_pixelvals = models.DecimalField('Standard Deviation of Pixel Values', decimal_places=2, max_digits=15,default=0)
	major_axis = models.DecimalField('Major Axis', decimal_places=2, max_digits=15, default=-1)
	minor_axis = models.DecimalField('Minor Axis', decimal_places=2, max_digits=15, default=-1)
	ellips_aratio = models.DecimalField('Ellips Aspect Ratio', decimal_places=2, max_digits=15, default=-1)
	ellips_orientation = models.DecimalField('Ellips Angle of Orientation', decimal_places=2, max_digits=15, default=-1)
	nozzle_distance = models.DecimalField('Distance from Nozzle', decimal_places=2, max_digits=15, default=0)
	nozzle_angle = models.DecimalField('Angle from Nozzle', decimal_places=5, max_digits=15, default=0)
	# moments = models.ListCharField(DecimalField,size=0)
	# location = models.ListCharField(IntegerField,size=0)
	# r_g = models.
	# bound_rec = ListCharField(IntegerField,size=0)
	# cropped = 2D array

# class ThicknessImage(models.Model)
	# spray = 
	# frame_index = 


