from django.http import HttpResponse
from spray.models import Frame
from django.shortcuts import render
# Create your views here.

def index(request):

	# array = run()

	# x  = Frame()
	# x.frame_index = 1
	# x.frame_angle = 75
	# x.num_droplets = 1000
	# x.mean_pixel = 200
	# x.save()


	return render(request, 'spray/index.html')


