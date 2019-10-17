import os

# ----------------------------------------------------------------------------
## PARAMETERS
##    assign image processing and shape classification parameters and methods
##    see the README for additional descriptions of parameters
# ----------------------------------------------------------------------------
dir_name = "./H_Spray/80bar/T2"  #a directory containing all images to be analyzed
refresh = "linux"     #if this value is 'linux or 'windows,' for the os in use,
                        #all directories listen in the json will be deleted
                        #before analysis.  assign refresh to any other value
                        #to keep existing data and add new analysis into
                        #the directories without replacing old data
                        

# -----------------------------------------
# normalization paramters
# -----------------------------------------
normal = 1            #do we need to normalize the images? 1 for yes, 0 for no
removeBackground = 0  #do we need to remove background? 1 for yes, 0 for no
eightBit = 0          #convert images to 8-bit? 1 for 8 bit, 0 for 16 bit
n_imgs = [1,399]      #which images in the directory, dir_name, 
                        #need to be normalized
ndir  = "./norm_imgs/"#name of directory where normalized images will be saved
n_frames = 6          #number of frames at the beginning of a spary
                        #with no fuel which can be used to calculate background
common = "/T2-X=0.00_Y=1.00__"  #portion of image file name which is shared
                                  #by all files in the directory of images to
                                  #be normalized
bkgd_section = [0,100,412,512]    #if using one-reference-point normalization,
                                  #choose the section of the image to assign
                                  #as background (should be a section of the
                                  #image which never has fuel throughout the
                                  #entire spray).  the formatting of this list
                                  #is [top, bottom, left, right]
# -----------------------------------------

# -----------------------------------------
# alter all images
# -----------------------------------------
  # some spray show x-xray artifacts on all images.  Change these cut_ values
  # to crop a portion of all images in a spray prior to analysis (in number of pixels)
cut_top = 0
cut_bottom = 0
cut_left = 0
cut_right = 0
angle = 0             #rotate image so that the spray is oriented downward
# -----------------------------------------

# -----------------------------------------
# image processing parameters
# -----------------------------------------
common_len = 7        #for shape image label purposes:
                        #number of letters to chop off of beginning of
                        #image labels to avoid huge shape labels (ignore if
                        #normalizing images)
sdn = "./bigsplits/"  #name of directory where shape images will be saved
ldn = "./labels/"     #name of directory where shape images will be saved, 
                        #sorted by cluster
bdn = "./binary/"     #name of directory where binary images will be saved, 
thresh = 80           #pixel threshold (ignore if employing multithresholding)
sapmin = 300           #shape area pixel minimum
sapmax = 350          #shape area pixel maximum
ssrm = 3.45           #maximum ratio of shape length:shape width (or vise versa)
pa_ratio = 0.7        #maximum ratio of shape perimeter to area
mthresh = 4           #number of clusters for initial image thresholding 
                        #using multithresholding
# -----------------------------------------

# -----------------------------------------
# image classification/clustering parameters
# -----------------------------------------
kmeans = 0         #do we want to run kmeans clustering? 1 for yes, 0 for no
nkc = 4               #number of clusters for k-means clustering of 
                        #shape classification
Otsu = 0              #1 to use Otsu thresholding, 0 to use a different method

save = 1              #do we want to save clustered shapes and images with 
                        #shapes highlighted?  1 for yes, 0 for no
cdir  = "./clustered_imgs/" #name of directory where images with shapes
                            #highlighted will be saved
# -----------------------------------------
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
## Write JSON input file
# ----------------------------------------------------------------------------
num_files = len(os.listdir("./" + dir_name))

with open("input.json",'w') as l:
  l.write('{\n\t"img_dir" : "' + dir_name + '",\n')
  l.write('\t"refresh" : "' + refresh + '",\n')
  l.write('\t"normalize" : ' + str(normal) + ',\n')
  l.write('\t"removeBackground" : ' + str(removeBackground) + ',\n')
  l.write('\t"eightBit" : ' + str(eightBit) + ',\n')
  l.write('\t"norm_imgs" : [' + str(n_imgs[0]) + ',' + str(n_imgs[1]) +'],\n')
  l.write('\t"normalized_img_dir" : "' + ndir + '",\n')
  l.write('\t"nframes" : ' + str(n_frames) + ',\n')
  l.write('\t"common_name" : "' + common + '",\n')
  l.write('\t"bkgd_section" : [' + str(bkgd_section[0]) + ',' + \
                                  str(bkgd_section[1]) + ',' + \
                                  str(bkgd_section[2]) + ',' + \
                                  str(bkgd_section[3])  + '],\n')
  l.write('\t"label_common_length" : ' + str(common_len) + ',\n')
  l.write('\t"img_names": [\n')
  n = 1
  for f in os.listdir("./" + dir_name):
    if n < num_files:
      l.write('\t\t"' + f + '",\n')
      n += 1
    else:
      l.write('\t\t"' + f + '"\n')
  l.write('\t\t],\n')
  l.write('\t"shape_dir_name" : "' + sdn + '",\n')
  l.write('\t"label_dir_name" : "' + ldn + '",\n')
  l.write('\t"binary_dir_name" : "' + bdn + '",\n')
  l.write('\t"pixel_threshold": ' + str(thresh) + ',\n')
  l.write('\t"shape_area_pixel_minimum": ' + str(sapmin) + ',\n')
  l.write('\t"shape_area_pixel_maximum": ' + str(sapmax) + ',\n')
  l.write('\t"shape_side_ratio_maximum": ' + str(ssrm) + ',\n')
  l.write('\t"max_ratio": ' + str(pa_ratio) + ',\n')
  l.write('\t"cut_top": ' + str(cut_top) + ',\n')
  l.write('\t"cut_bottom": ' + str(cut_bottom) + ',\n')
  l.write('\t"cut_left": ' + str(cut_left) + ',\n')
  l.write('\t"cut_right": ' + str(cut_right) + ',\n')
  l.write('\t"angle": ' + str(angle) + ',\n')
  l.write('\t"num_kmeans_clusters": ' + str(nkc) + ',\n')
  l.write('\t"kmeans": ' + str(kmeans) + ',\n')
  l.write('\t"kthresh": ' + str(mthresh) + ',\n')
  l.write('\t"Otsu": ' + str(Otsu) + ',\n')
  l.write('\t"save": ' + str(save) + ',\n')
  l.write('\t"clustered_img_dir": "' + str(cdir) + '"\n')
  l.write('}')
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
## RUN ANALYSIS
# ----------------------------------------------------------------------------
#os.system("python -i analysis_script.py")
#print("Will: Done")
os.system("python -i run.py")
# ----------------------------------------------------------------------------
