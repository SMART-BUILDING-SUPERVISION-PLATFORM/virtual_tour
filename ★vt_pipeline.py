from Py360Convert.convert360 import convert_e_2_c, convert_separated_6c_2_c, convert_c_2_e
import os 
import argparse
import numpy as np
from PIL import Image
from HLoc_CamPose import recon
from vt_camera_pose import extract_cam_pose
from vt_json import make_json
import re

'''
## Requirement ## 
<Hierarchical-Localization>
-> https://github.com/cvg/Hierarchical-Localization

<py360converter>
-> zip file 

<virtual tour>
-> https://github.com/digicademy/virtualTour
'''


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
1. Convert 1 equirectangular image to 6 perspective images (overlapped)
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
print("\n1. Convert 1 equirectangular image to 6 perspective images (overlapped)")
#########################################
# Setting
PROJECT_NAME = "test1"   # @@@@@ Project name @@@@@  
root_name = "./virtualTour/"
resource = "resources/"
input_path = root_name + resource + PROJECT_NAME + "/"
in_path = input_path + "panos/"
input_imgs = input_path + "mapping/"  # output folder
w = 2880
h = 1440
#########################################
if not os.path.isdir(input_imgs):
    os.makedirs(input_imgs)
    
# load image list
image_path = in_path
img_list = os.listdir(image_path) # load all images in folders
# img_list_jpg = [img for img in img_list if img.endswith(".jpg")] 
print(os.getcwd())
print ("img_list: {}".format(img_list))

face_k=['F', 'R', 'B', 'L', 'U', 'D']
img_list_np = []
image_num = 0
for i in img_list:
    img = np.array(Image.open(image_path + i))
    if len(img.shape) == 2:
        img = img[..., None]
    out_filename = i.split(".")[0]
    separated_cubemap, dict_separated_cubemap = convert_e_2_c(img, w, input_imgs, out_filename, overlap=1.5, mode= "bilinear", cube_format = "dice_overlap")
    print()


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
2. Extract camera pose by SfM 
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
print("\n2. Extract camera pose by SfM")

#########################################
# Setting
output_path =  input_path
#########################################
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Remove file including "_U" or "_D" 
for filename in os.listdir(input_imgs):
    file_path = os.path.join(input_imgs, filename)
    if '_U' in filename or '_D' in filename:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
                
model = recon(input_imgs, output_path, retrieval_type='netvlad', feature_type='superpoint_inloc', match_type='superglue')


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
3. Make setting json file for virtual tour
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
print("\n3. Make setting json file for virtual tour")

#########################################
# Setting 
# in_path = "input/test/panos/"
json_path = root_name + resource + "json"
map_path = resource + "maps/sampleMap.jpg"
out_path = json_path + "/sample.json"
#########################################
if not os.path.isdir(json_path):
    os.makedirs(json_path)

for root, dirs, files in os.walk(output_path):
    for dir_name in dirs:
        if "sfm" in dir_name:
            sfm_input_path = output_path + "/"+ dir_name
            print(sfm_input_path)

cam_positions, cam_info, images_path, img_pairs = extract_cam_pose(sfm_input_path, input_path)

imgs_path =  [input_imgs + img_path for img_path in images_path]
make_json(map_path, cam_positions, cam_info, img_pairs, pano_path= resource + PROJECT_NAME + "/panos/", imgs_path=imgs_path, output_path=out_path)