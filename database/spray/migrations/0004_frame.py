# Generated by Django 3.0 on 2020-01-17 20:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('spray', '0003_auto_20191210_2147'),
    ]

    operations = [
        migrations.CreateModel(
            name='Frame',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame_index', models.IntegerField(default=0, verbose_name='Number of Frames')),
            ],
        ),
    ]
