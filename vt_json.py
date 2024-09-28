import json
import glob
import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def orientation(A, B):
    """
    Function to determine whether vector B is on the left or right side
    of the line segment formed by vector A's start and end points.

    A, B: NumPy arrays representing 2D vectors in the form (x, y).
    Returns: One of 'left', 'right', or 'collinear'.
    """
    # Calculate the cross product of vectors A and B.
    cross_product = np.cross(A, B)

    # Determine the orientation based on the sign of the cross product.
    if cross_product > 0:
        return  -1 # 'left'
    elif cross_product < 0:
        return 1 # 'right'
    else:
        return 2 # 'collinear'
    
def make_json(map_path, cam_locations, cam_info, img_pairs, pano_path, imgs_path, output_path):
    #########################################
    if not os.path.isdir("virtualTour/resources/json"):
        os.makedirs("virtualTour/resources/json")
        output_path = "virtualTour/resources/json/sample.json"
    # if not os.path.isdir("./resources/panos"):
    #     hq = "./resources/panos/hq"
    #     mobile = "./resources/panos/mobile"
    #     os.makedirs(hq)
    #     os.makedirs(mobile)
        # shutil.copytree(pano_path, hq)
        # shutil.copytree(pano_path, mobile)
    if not os.path.isdir("virtualTour/resources/maps"):
        os.makedirs("virtualTour/resources/maps")
    #########################################
    
    data = {
        "startLocation": 1,
        "maps": [
            {
                "uid": 1,
                "image": map_path,
                "mapSpots": []
            }
        ],
        "locations": []
    }

    # #### Draw mapspots in map ##### 
    # cam_location_path = "./resources/cam_locations/cam_location.csv"
    # map_spots = np.loadtxt(cam_location_path, delimiter=",") # required!!!!!
    map_spots = cam_locations
    # cam_lnfo_path = "./resources/cam_locations/cam_info.json"
    # cam_info = json.load(open(cam_lnfo_path, 'r'))

    front = np.array(cam_info["front"])
    loc = np.array(cam_info["loc"])
    images = np.array(cam_info["images"]) 

    roi_point = []
    roi_point_ = []
    for f, l, img in zip(front, loc, images):
        if "F" in img:
            roi_point.append(f)
            roi_point_.append(l)
    roi_point = np.array(roi_point)
    roi_point_ = np.array(roi_point_)

    # # test-draw camera position
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xs=roi_point[:,0], ys=roi_point[:,1], zs=roi_point[:,2], c="red")
    # ax.scatter(xs=roi_point_[:,0], ys=roi_point_[:,1], zs=roi_point_[:,2], c="blue")
    # ax.set_xlim(-5,5)
    # ax.set_ylim(-5,5)
    # ax.set_zlim(-5,5)
    # plt.show()

    map_spots = np.array(cam_info["loc"])
    # Align to 0,0 coordinates on the screen
    xmin = np.min(map_spots[:, 0])
    ymin = np.min(map_spots[:, 2])
    if xmin < 0:
        map_spots[:, 0] = map_spots[:, 0] - xmin + 3
    elif  xmin > 0: 
        map_spots[:, 0] = map_spots[:, 0] + xmin
        
    if ymin < 0:
        map_spots[:, 2] = map_spots[:, 2] - ymin + 3
    elif  ymin > 0: 
        map_spots[:, 2] = map_spots[:, 2] + ymin

    # Fit map to set size
    xmax = np.max(map_spots[:, 0])
    ymax = np.max(map_spots[:, 2])
    max_width = 400
    max_height = 400
    if xmax > max_width and ymax > max_height:
        if xmax >= ymax:
            map_spots *= max_width / xmax
        else:
            map_spots *= max_height / ymax

    elif xmax > max_width:
        map_spots *= max_width / xmax 
    else:
        map_spots *= max_height / ymax 
        
    # Record mapSpots
    data["maps"][0]["mapSpots"] = []
    for i in range(len(map_spots)):
        data["maps"][0]["mapSpots"].append(
            {
                "uid": i+1,
                "tooltip": "Map Spot {}" .format(i+1),
                "mapPosX": int(map_spots[i][0]),
                "mapPosY": int(map_spots[i][2])
            }
        )

    # Load panorama images for connecting panorama images
    # pano_path = "./resources/panos" 
    # images_folder_path = glob.glob(pano_path + '*')
    # images_path = []
    # for img_foler_path in images_folder_path:
    #     tmp_images_path = glob.glob(img_foler_path + '/*')
    #     images_path.append(sorted(tmp_images_path))
        
    dict_img_pairs = {}
    for present_idx, img_pair in enumerate(img_pairs):
        # make pano path
        present_img_name_ = imgs_path[present_idx].split("/")[-1]
        present_img_name = re.sub(r'_.', '', present_img_name_)
        for target_idx in img_pair[0]:
            # make pano path
            target_img_name_ = imgs_path[target_idx].split("/")[-1]
            target_img_name = re.sub(r'_.', '', target_img_name_)
            if target_img_name != present_img_name:
                if present_img_name not in dict_img_pairs.keys():
                    dict_img_pairs[present_img_name] = []
                if target_img_name not in dict_img_pairs[present_img_name]:
                    dict_img_pairs[present_img_name].append(target_img_name)
    print(dict_img_pairs)

    # Connect transit points in panorama images
    # @@@@@@@@@@@@@@@@@@@@@@@@ setting required @@@@@@@@@@@@@@@@@@@@@@@@
    data["locations"] = []
    img_names = [] 
    for i in cam_info["images"]:
        result = re.sub(r'_.', '', i)
        img_names.append(result)
    
    for present_pano_name, target_pano_names in dict_img_pairs.items():
        present_pano_path = pano_path + present_pano_name
        present_idx = img_names.index(present_pano_name)
        # transiiton (waypoint) for each panorama
        cameraTargets = {"-1": {"lat": 0, "lon": 180}} 
        transitions = []
        for target_pano_name in target_pano_names:              
            target_idx = img_names.index(target_pano_name)
            # waypoint location in panorama
            v1 = np.array(cam_info["front"][present_idx])-np.array(cam_info["loc"][present_idx])  # vector of angle for present camera's front view 
            v2 = map_spots[target_idx]-map_spots[present_idx]    # vector of angle from present camera to next camera 
            angle = angle_between((v1[0],v1[2]), (v2[0],v2[2]))
            orient = orientation((v1[0],v1[2]), (v2[0],v2[2]))  # left -1 , right 1 
            # print(np.rad2deg(angle), orient)
            lon = np.rad2deg(angle)*orient+180
            lat = np.rad2deg(0)  
            lat = np.max((-35, np.min((45, lat))))
            phi = np.deg2rad(90 - lat)
            theta = np.deg2rad(lon)
            x = 195 * np.sin(phi) * np.cos(theta)
            y = 195 * np.cos(phi)
            z = 195 * np.sin(phi) * np.sin(theta)
            
            # the location of waypoint in an image. we have to revise waypoint location as distacne far or near 
            # @@@@@@@@@@@@@@@@@@@@@@@@ setting required @@@@@@@@@@@@@@@@@@@@@@@@
            transitions.append(
                {
                    "target_location": target_idx+1,
                    "tooltip": "Location {}".format(target_idx+1),
                    "posX": x,
                    "posY": y,
                    "posZ": z
                }
            )
            
            # camera view in panorama when other panorama move to this panorama
            # @@@@@@@@@@@@@@@@@@@@@@@@ setting required @@@@@@@@@@@@@@@@@@@@@@@@
            target_lat = 0  # revise required!!!!!!!!!!!!!!!!!!!
            target_lon = 180 + (180-np.rad2deg(angle)*-orient)	# revise required!!!!!!!!!!!!!!!!!!!
            cameraTargets[target_idx+1] = {"lat": target_lat, "lon": target_lon}
        
        # connect panorama, and connect between waypoints in panorama 
        data["locations"].append(
            {
                "uid": present_idx+1,
                "image": {
                    "default": present_pano_path,
                    "hq":  present_pano_path,
                    "mobile":  present_pano_path
                },
                "mapUid": 1,
                "transitions": transitions,
                "cameraTargets": cameraTargets,
                "hotspots": []
            }
        )

    # save json file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent="\t")
    print(f"[Complete] Save to {output_path}")



if __name__ == "__main__":
        
    map_path = "resources/maps/sampleMap.png"
    data = {
        "startLocation": 1,
        "maps": [
            {
                "uid": 1,
                "image": map_path,
                "mapSpots": []
            }
        ],
        "locations": []
    }

    # #### Draw mapspots in map ##### 
    cam_location_path = "./cam_location.csv"
    map_spots = np.loadtxt(cam_location_path, delimiter=",") # required!!!!!

    # cam_lnfo_path = "./resources/cam_locations/cam_info.json"
    cam_lnfo_path = "./cam_info.json"

    cam_info = json.load(open(cam_lnfo_path, 'r'))

    front = np.array(cam_info["front"])
    loc = np.array(cam_info["loc"])
    images = np.array(cam_info["images"]) 

    roi_point = []
    roi_point_ = []
    for f, l, img in zip(front, loc, images):
        if img in ["1_F.jpg", "2_F.jpg", "3_F.jpg", "4_F.jpg"]:
            roi_point.append(f)
            roi_point_.append(l)

    roi_point = np.array(roi_point)
    roi_point_ = np.array(roi_point_)

    # test-draw camera position
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=roi_point[:,0], ys=roi_point[:,1], zs=roi_point[:,2], c="red")
    ax.scatter(xs=roi_point_[:,0], ys=roi_point_[:,1], zs=roi_point_[:,2], c="blue")
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    plt.show()


    map_spots = np.array(cam_info["loc"])
    # Align to 0,0 coordinates on the screen
    xmin = np.min(map_spots[:, 0])
    ymin = np.min(map_spots[:, 2])
    if xmin < 0:
        map_spots[:, 0] = map_spots[:, 0] - xmin
    elif  xmin > 0: 
        map_spots[:, 0] = map_spots[:, 0] + xmin
        
    if ymin < 0:
        map_spots[:, 2] = map_spots[:, 2] - ymin
    elif  ymin > 0: 
        map_spots[:, 2] = map_spots[:, 2] + ymin

    # Fit map to set size
    xmax = np.max(map_spots[:, 0])
    ymax = np.max(map_spots[:, 2])
    max_width = 400
    max_height = 400
    if xmax > max_width and ymax > max_height:
        if xmax >= ymax:
            map_spots *= max_width / xmax
        else:
            map_spots *= max_height / ymax

    elif xmax > max_width:
        map_spots *= max_width / xmax 
    else:
        map_spots *= max_height / ymax 
        
    # Record mapSpots
    data["maps"][0]["mapSpots"] = []
    for i in range(len(map_spots)):
        data["maps"][0]["mapSpots"].append(
            {
                "uid": i+1,
                "tooltip": "Map Spot {}" .format(i+1),
                "mapPosX": int(map_spots[i][0]),
                "mapPosY": int(map_spots[i][2])
            }
        )

    # Load panorama images for connecting panorama images
    pano_path = "./resource/panos" 
    images_folder_path = glob.glob(pano_path + '/*')
    images_path = []
    for img_foler_path in images_folder_path:
        tmp_images_path = glob.glob(img_foler_path + '/*')
        images_path.append(sorted(tmp_images_path))
        

    # Connect transit points in panorama images
    # @@@@@@@@@@@@@@@@@@@@@@@@ setting reqiored @@@@@@@@@@@@@@@@@@@@@@@@
    # setting reqiored !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    trasits = {"image1": ["image2"], "image2": ["image1", "image3"], "image3":["image2", "image4"], "image4":["image3"]}  

    data["locations"] = []
    for image_name_ in images_path[0]:  # 0=hp, 1=mobile
        image_name = image_name_.split("/")[-1].split(".")[0]
        img = Image.open(image_name_)
        w, h = img.size
        # transiiton (waypoint) for each panorama
        transitions = []
        
        # basic setting for camara view (when there are no connected waypoints related to panorama id)
        cameraTargets = {"-1": {"lat": 0, "lon": 180}} 



        for transit in trasits[image_name]:
            target_idx = int(re.sub(r'[^0-9]', "", transit))
            
            
            
            # waypoint location in panorama
            # @@@@@@@@@@@@@@@@@@@@@@@@ setting reqiored @@@@@@@@@@@@@@@@@@@@@@@@
            present_idx = int(image_name[-1])-1
            
            v1 = np.array(cam_info["front"][present_idx])-np.array(cam_info["loc"][present_idx])  # vector of angle for present camera's front view 
            v2 = map_spots[int(transit[-1])-1]  - map_spots[present_idx]    # vector of angle from present camera to next camera 
            
            angle = angle_between((v1[0],v1[2]), (v2[0],v2[2]))
            orient = orientation((v1[0],v1[2]), (v2[0],v2[2]))  # left -1 , right 1 
            print(np.rad2deg(angle), orient)
            lon = np.rad2deg(angle)*orient+180
            lat = np.rad2deg(0)  
            lat = np.max((-35, np.min((45, lat))))
            phi = np.deg2rad(90 - lat)
            theta = np.deg2rad(lon)
            x = 195 * np.sin(phi) * np.cos(theta)
            y = 195 * np.cos(phi)
            z = 195 * np.sin(phi) * np.sin(theta)
            # revise requried!!!!!!!!!!!!!!!!!!!     
            transitions.append(
                {
                    "target_location": target_idx,
                    "tooltip": "Location {}".format(target_idx),
                    "posX": x,
                    "posY": y,
                    "posZ": z
                }
            )
            # camera view in panorama when other panorama move to this panorama
            # @@@@@@@@@@@@@@@@@@@@@@@@ setting reqiored @@@@@@@@@@@@@@@@@@@@@@@@
            target_lat = 0  # revise requried!!!!!!!!!!!!!!!!!!!
            target_lon = 180 + (180-np.rad2deg(angle)*-orient)	# revise requried!!!!!!!!!!!!!!!!!!!
            cameraTargets[target_idx] = {"lat": target_lat, "lon": target_lon}
        
    
        # connect panorama, and connect between waypoints in panorama 
        location_idx = int(re.sub(r'[^0-9]', "", image_name))
        data["locations"].append(
            {
                "uid": location_idx,
                "image": {
                    "default": images_path[1][location_idx-1][2:],
                    "hq":  images_path[0][location_idx-1][2:],
                    "mobile":  images_path[1][location_idx-1][2:]
                },
                "mapUid": 1,
                "transitions": transitions,
                "cameraTargets": cameraTargets,
                "hotspots": []
            }
        )


    # save json file
    file_path = "./virtualTour/resources/json/sample.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent="\t")
    print(f"[Complete] Save to {file_path}")
    # "hotspots": [
    # 	{
    # 		"uid": 1,
    # 		"title": "Title",
    # 		"tooltip": "Information",
    # 		"text": "Example with audio.",
    # 		"audio": "resources/audio/audiosample",
    # 		"posX": -61,
    # 		"posY": 114,
    # 		"posZ": 145
    # 	}
    # 	]
    # "hotspots": [
    # 	{
    # 		"uid": 1,
    # 		"title": "Title",
    # 		"tooltip": "Information",
    # 		"text": "Example with image. Some text information here. <a href=\"http://www.digitale-akademie.de/\"  target=\"_blank\">Sample Link, opens in a new window.</a>",
    # 		"images": [
    # 			{
    # 				"figure": "resources/images/imagesample.png",
    # 				"caption": "Image credits or description. ? Brunhilde Escherich"
    # 			}
    # 		],
    # 		"posX": -38,
    # 		"posY": 107,
    # 		"posZ": -157
    #	]
