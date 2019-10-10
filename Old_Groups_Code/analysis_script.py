from lib_images import *
from lib_clustering import *
from lib_normalize import *
import os
import sys
import json

print("Made it")

with open('input.json') as f:
  data = json.load(f)

# ----------------------------------------------------------------------------
## REFRESH DIRECTORIES
#     clear out directories
# ----------------------------------------------------------------------------
if data["refresh"] == 'linux':
  os.system("rm -r %s"%(data["shape_dir_name"]))
  os.system("rm -r %s"%(data["label_dir_name"]))
  os.system("rm -r %s"%(data["normalized_img_dir"]))
  os.system("rm -r %s"%(data["clustered_img_dir"]))
  os.system("rm -r %s"%(data["binary_dir_name"]))

  os.system("mkdir %s"%(data["shape_dir_name"][2:-1]))
  os.system("mkdir %s"%(data["label_dir_name"]))
  os.system("mkdir %s"%(data["normalized_img_dir"]))
  os.system("mkdir %s"%(data["clustered_img_dir"]))
  os.system("mkdir %s"%(data["binary_dir_name"]))
  for i in range(data["num_kmeans_clusters"]):
    os.chdir(data["label_dir_name"])
    os.system("mkdir %s"%(i))
    os.chdir("..")

elif data["refresh"] == 'windows':
  if (os.path.isdir(data["shape_dir_name"]) == True) and (os.path.isdir(data["label_dir_name"]) == True):
      os.system("rmdir {} /s".format(data["shape_dir_name"][2:-1]))
      os.system("rmdir {} /s".format(data["label_dir_name"][2:-1]))
      os.system("rmdir {} /s".format(data["normalized_img_dir"][2:-1]))
      os.system("rmdir {} /s".format(data["clustered_img_dir"][2:-1]))
      os.system("rmdir {} /s".format("binary"))

  os.system("mkdir {}".format(data["normalized_img_dir"][2:-1]))
  os.system("mkdir {}".format(data["shape_dir_name"][2:-1]))
  os.system("mkdir {}".format(data["label_dir_name"][2:-1]))
  os.system("mkdir {}".format(data["clustered_img_dir"][2:-1]))
  os.system("mkdir {}".format("binary"))

  os.chdir(data["label_dir_name"])
  for i in range(data["num_kmeans_clusters"]):
    os.system("mkdir %s"%(i))
  os.chdir("..")
# ----------------------------------------------------------------------------

print("Refresh done")
# ----------------------------------------------------------------------------
## LOAD IN DATA FROM JSON FILE
# ----------------------------------------------------------------------------
img_dir = data["img_dir"]
#img_dir = "./norm_imgs/"
img_names = data["img_names"]
img_locs = [img_dir+name for name in img_names]
shape_dir = data["shape_dir_name"]
label_dir = data["label_dir_name"]
common_len = data["label_common_length"]
all_shapes = []
all_imgs = []
# ----------------------------------------------------------------------------
print("Did this")

# ----------------------------------------------------------------------------
## NORMALIZE IMAGES
if data["normalize"]:
  print("\n Normalizing...")
  normalize()
  img_locs = [data["normalized_img_dir"]+str(n)+".tif" for n in \
            range(data["norm_imgs"][0]-1,data["norm_imgs"][1])]
  common_len = len(data["normalized_img_dir"])
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## MULTITHRESHOLDING
#   run kmeans on the first few images and average the
#   clusters to get the threshold value
# ----------------------------------------------------------------------------
print("\n Performing multithresholding...")
if data["kthresh"] != 0:
    center1 = [] #darkest cluster
    center2 = [] #next darkest cluster
    total = 10
    for loc in img_locs[:total]:
        img = Image(loc)
        min_center = min(img.center)[0]
        min2_center = min(img.center[img.center != min(img.center)])
        #print(img.center)
        #print(min_center)
        #print(min2_center)
        center1.append(min_center)
        center2.append(min2_center)
    center1 = np.median(center1)
    center2 = np.median(center2)
    #thresh_val = (center1 + center2)/2
    thresh_val = center2
    #print(thresh_val)
    data["pixel_threshold"] = thresh_val
    data["kthresh"] = 0         
    with open('input.json','w') as f:
        json.dump(data,f,indent=4)
    print("final multithresholding value = %s \n" %(thresh_val))
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## PERFORM IMAGE PROCESSING
# ----------------------------------------------------------------------------
with open('input.json') as f:
  data = json.load(f)
print("\n Processing images...")
count = 0
for loc in img_locs:
    if count % 10 == 0:
      if data["normalize"]:
        print("processing image %s of %s" %(loc[len(data["normalized_img_dir"]):-4],len(img_locs)))
      else:
        print("processing image %s of %s" %(loc[len(img_dir):-1],len(img_locs))) 
    count += 1
    img = Image(loc)
    all_imgs.append(img)
    all_shapes = np.concatenate((all_shapes,img.big_shapes))
    img.splitShapes(shape_dir)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## PERFORM KMEANS CLUSTERING
# ----------------------------------------------------------------------------
if data["kmeans"]:
  print("\n Performing kmeans clustering...")
  ret,label,center = kmeans(all_shapes,data["num_kmeans_clusters"])
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## SAVE CLUSTERED SHAPES AND IMAGES WITH HIGHLIGHTED SHAPES TO DIRECTORIES
# ----------------------------------------------------------------------------
if data["save"]:
  print("\n Printing shapes to directories...")
  for img in all_imgs:
      img.writeLabels(label_dir)
      img.drawLabels()
      if img.clusters_image is not None:
        write(img.clusters_image,data["clustered_img_dir"]+img.location[common_len:-4]+".jpg")
# ----------------------------------------------------------------------------
