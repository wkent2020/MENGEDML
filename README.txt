-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
README for MENG_Design, fuel spray image processing project
By Gabriela Basel, Mike Gu, and Kevin Slater
-----------------------------------------------------------------------------
Included in this README:
  - How to run image processing and analysis with the code as-is
  - Brief descriptions of files contained in this directory
  - List of parameters whose values can be changed
  - How to specify which methods to use from those available in these scripts
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------


-----------------------------------------------------------------------------
RUNNING IMAGE PROCESSING AND ANALYSIS
-----------------------------------------------------------------------------
*** to use the code as is, run.py is the only file which need be edited
*** change all parameter values and choose methodology inside run.py
*** execute command $ python run.py
      to perform image processing
-----------------------------------------------------------------------------


-----------------------------------------------------------------------------
FILES CONTAINED IN THIS DIRECTORY
-----------------------------------------------------------------------------
analysis_script.py:
    This file is run by run.py to perform analysis.  Refreshes directories,
    normalizes images, performs image thresholding, shape isolation,
    and kmeans clustering of shapes, and prints binary processed images
    and clustered shapes
input.json:
    Contains all parameter values and method choices, and acts as the input
    file for the analysis script.  Edit the parameter values and choose
    methods in run.py, not directly in this file.
lib_clustering.py:
    Contains kmeans function
lib_images.py:
    Contains Image and Shape classe definitions
lib_normalize.py:
    Contains normalization functions
misc_functions.py:
    Contains function to overlay all shapes in a given cluster on top of
    each other to produce a representative shape of the cluster.  Add
    new analysis functions here.
model_building.ipynb:
    Jupyter notebook containing machine learning analysis.
run.py:
    Run file: assigns parameters, chooses methods, writes input json, and
    runs analsys.  When not editing source function or editing/adding new
    methodology, edit this file.
-----------------------------------------------------------------------------


-----------------------------------------------------------------------------
ALL INPUT DATA SHOULD BE CHANGED IN RUN.PY
See below the various parameters that can be changed
# ----------------------------------------------------------------------------

dir_name: name of the directory containing all images to be analyzed
refresh:  choose whether to erase existing analysis or perform new analysis
            on top of existing analysis. assign value of 'linux' if you are
            running a linux os and want to delete existing analyses, assign
            value of 'windows' if you are running a windows os and want to
            delete existing analyses, and assign any other value to keep
            existing analysis

-----------------------------------------
normalization paramters
-----------------------------------------
normal: choose whether to normalize the images or not.  value of 1 to
          normalize images, and 0 to skip normalizing the images
n_imgs: if normalizing images, specify which images to normalize in the
          following format:
          [integer of first image to normalize, integer of last image to normalize]
ndir: if normalizing images, name of directory where normalized images will be saved
n_frames: number of frames at the beginning of the spray which contain no
          fuel and can be used to calculate the constant background
common: if normalizing images, this is the portion of the filename shared by
          all files in the directory to be normalized.  it is not necessary
          to specify a value here, but can serve to shorten the filenames
          of new images
bkgd_section: if normalizing images, specify the portion of the background
                which can be used to caluclate the average background value of
                an image.  this should be a section of the image which never
                contains fuel throughout the entirety of the spray, such as an
                area near the edge of the image.  this should be specified in 
                the following format:
                [top, bottom, left, right]
                where each entry in the list is the pixel value on that edge
                of the portion of the background chosen.  for example,
                [0, 100, 0, 100] would specify a 100 x 100 pixel square in the
                upper left corner of the image.
-----------------------------------------

-----------------------------------------
parameters to alter all images
  some sprays show x-ray artifacts on all images.  Change these cut_ values
  to crop a portion of all images in a spray prior to analysis (in number of pixels)
-----------------------------------------
cut_top:  crop this number of pixels off the top of every image
cut_bottom: crop this number of pixels off the bottom of every image
cut_left: crop this number of pixels off the left of every image
cut_right:  crop this number of pixels off the right of every image
angle:  rotate the image clockwise by this angle so that the spray is oriented
          downward, simplifying analysis
-----------------------------------------

-----------------------------------------
image processing parameters
-----------------------------------------
common_len: for shape image label purposes, the number of letters to chop off
              the beginning of image labels to avoid huge shape label names 
              (ignore if normalizing images)
sdn:  name of directory where shape images will be saved
ldn:  name of directory where shape images will be saved, sorted by cluster 
bdn:  name of directory where binary images will be saved
thresh: pixel threshold value, above which all pixels will be ignored (ignore
          if employing multithresholding)
sapmin: the mimimum area a shape can have (in number of pixels) to be included
          in analysis
sapmax: the maximum area a shape can have (in number of pixels) to be included
          in analysis
ssrm: the maximum ratio of shape length:shape width or shape width: shape height
          a shape can have to be included in analysis
pa_ratio: the maximum ratio of shape perimeter: shape area a shape can have to
            be included in analysis
mthresh:  if using multithresholding to find a thresholding value, specify the
            number of clusters to use.  specify 0 to use a different thresholding
            method
Otsu: choose whether to use Otsu thresholding to threshold the images or not.
        value of 1 to use Otsu thresholding, and 0 to use a different method.
-----------------------------------------

-----------------------------------------
image classification/clustering parameters
-----------------------------------------
kmeans: choose whether to cluster the shapes or not.  value of 1 to
          cluster, and 0 to skip clustering
nkc:  if running kmeans clustering, specify the number of clusters to sort
        the shapes into
save: choose whether to save clustered shapes as images as well as images
        with the shapes highlighted or not.  value of 1 to save these images,
        and 0 to not save these images. 
cdir: name of directory where images with shapes highlighted will be saved
-----------------------------------------
----------------------------------------------------------------------------
