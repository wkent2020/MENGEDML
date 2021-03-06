# Generated by Django 3.0 on 2020-02-05 01:19

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('spray', '0005_droplets'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='droplets',
            name='perimeter',
        ),
        migrations.AddField(
            model_name='droplets',
            name='col_center_loc',
            field=models.IntegerField(default=0, verbose_name='Column center location'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='col_coa',
            field=models.IntegerField(default=0, verbose_name='Column center of area'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='ellips_aratio',
            field=models.DecimalField(decimal_places=2, default=-1, max_digits=15, verbose_name='Ellips Aspect Ratio'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='ellips_orientation',
            field=models.DecimalField(decimal_places=2, default=-1, max_digits=15, verbose_name='Ellips Angle of Orientation'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='frame_index',
            field=models.CharField(default='0', max_length=100, verbose_name='Droplet Label'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='inner_perimeter',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Inner Perimeter'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='major_axis',
            field=models.DecimalField(decimal_places=2, default=-1, max_digits=15, verbose_name='Major Axis'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='minor_axis',
            field=models.DecimalField(decimal_places=2, default=-1, max_digits=15, verbose_name='Minor Axis'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='mom_iner',
            field=models.DecimalField(decimal_places=5, default=0, max_digits=15, verbose_name='Moment of Inertia'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='nozzle_angle',
            field=models.DecimalField(decimal_places=5, default=0, max_digits=15, verbose_name='Angle from Nozzle'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='nozzle_distance',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Distance from Nozzle'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='outer_perimeter',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Outer Perimeter'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='rg',
            field=models.DecimalField(decimal_places=5, default=0, max_digits=15, verbose_name='Radius of gyration'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='row_center_loc',
            field=models.IntegerField(default=0, verbose_name='Row center location'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='row_coa',
            field=models.IntegerField(default=0, verbose_name='Row center of area'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='spray_label',
            field=models.CharField(default='Unknown', max_length=100, verbose_name='Droplet Label'),
        ),
        migrations.AddField(
            model_name='droplets',
            name='std_pixelvals',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Standard Deviation of Pixel Values'),
        ),
        migrations.AddField(
            model_name='frame',
            name='date_added',
            field=models.DateTimeField(default=datetime.datetime.now, verbose_name='Date Published'),
        ),
        migrations.AddField(
            model_name='frame',
            name='frame_angle',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=10, verbose_name='Spray angle'),
        ),
        migrations.AddField(
            model_name='frame',
            name='mean_pixel',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=10, verbose_name='Mean pixel value'),
        ),
        migrations.AddField(
            model_name='frame',
            name='num_droplets',
            field=models.IntegerField(default=0, verbose_name='Number of droplets'),
        ),
        migrations.AddField(
            model_name='frame',
            name='std_pixel',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=10, verbose_name='Standard deviation pixel value'),
        ),
        migrations.AddField(
            model_name='spray',
            name='alpha',
            field=models.DecimalField(decimal_places=20, default=0, max_digits=100, verbose_name='Alpha Coefficient'),
        ),
        migrations.AddField(
            model_name='spray',
            name='backg_image_index',
            field=models.IntegerField(default=10, verbose_name='Last background frame'),
        ),
        migrations.AddField(
            model_name='spray',
            name='cropping',
            field=models.IntegerField(default=0, verbose_name='Cropping Value'),
        ),
        migrations.AddField(
            model_name='spray',
            name='light_energy',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100, verbose_name='X-ray Energy'),
        ),
        migrations.AddField(
            model_name='spray',
            name='max_pixel',
            field=models.IntegerField(default=0, verbose_name='Maximum Pixel Value'),
        ),
        migrations.AddField(
            model_name='spray',
            name='min_pixel',
            field=models.IntegerField(default=0, verbose_name='Minimum Pixel Value'),
        ),
        migrations.AddField(
            model_name='spray',
            name='norm',
            field=models.CharField(choices=[('SB', 'Single background'), ('DBM', 'Double background by mean'), ('DBB', 'Double background by background')], default='SB', max_length=100),
        ),
        migrations.AddField(
            model_name='spray',
            name='num_frames_avg',
            field=models.IntegerField(default=5, verbose_name='Number of Background Frames for Averaging'),
        ),
        migrations.AddField(
            model_name='spray',
            name='rescaling_range',
            field=models.CharField(choices=[('FL', 'Floats'), ('8B', '8-bit'), ('16B', '16-bit')], default='8B', max_length=100),
        ),
        migrations.AddField(
            model_name='spray',
            name='xdim',
            field=models.IntegerField(default=0, verbose_name='X-Dimension'),
        ),
        migrations.AddField(
            model_name='spray',
            name='ydim',
            field=models.IntegerField(default=0, verbose_name='Y-Dimension'),
        ),
        migrations.AlterField(
            model_name='droplets',
            name='area',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Area'),
        ),
        migrations.AlterField(
            model_name='droplets',
            name='density',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Density'),
        ),
        migrations.AlterField(
            model_name='droplets',
            name='label',
            field=models.CharField(default='None', max_length=100, verbose_name='Droplet Label'),
        ),
        migrations.AlterField(
            model_name='droplets',
            name='mass',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=15, verbose_name='Mass'),
        ),
        migrations.AlterField(
            model_name='frame',
            name='frame_index',
            field=models.IntegerField(default=0, verbose_name='Frame index'),
        ),
        migrations.AlterField(
            model_name='spray',
            name='date_added',
            field=models.DateTimeField(default=datetime.datetime.now, verbose_name='Date Published'),
        ),
    ]
