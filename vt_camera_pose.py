from hloc.utils.read_write_model import read_model
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import json

def quaternion_to_euler_angle_vectorized2(w, x, y, z):  # return value as degree
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1)) # roll

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))  # pitch

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))  # yaw

    return X, Y, Z
    

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
 

def Get3Dfrom2D_DepthMaps(List2D, K, R, t, depth_map, scale=1, debug=False, dataFolder=r"data/img"):
    
    
	# https://github.com/colmap/colmap/issues/1476
    # List2D : n x 2 array of pixel locations in an image
	# K : Intrinsic matrix for camera
	# R : Rotation matrix describing rotation of camera frame
	# 	  w.r.t world frame.
	# t : translation vector describing the translation of camera frame
	# 	  w.r.t world frame
	# [R t] combined is known as the Camera Pose.
	# depth_map : depth map obtained from bin file corresponding to that image
	List2D = np.array(List2D)
	List3D = []
	# t.shape = (3,1)
	print('depth_map: ', depth_map.shape)
	depth_map_copy = depth_map
	min_depth, max_depth = np.percentile(depth_map, [5, 95])

	for p in List2D:
		# skip if index is out of bound
		if p[1] >= depth_map.shape[0] or p[0] >= depth_map.shape[1]:
			continue

		# Homogeneous pixel coordinate
		p = np.array([p[0], p[1], 1]).T; p.shape = (3,1)
		# print("pixel: \n", p)

		# Transform pixel in Camera coordinate frame
		pc = np.linalg.inv(K) @ p
		# print("pc : \n", pc, pc.shape)

		# Transform pixel in World coordinate frame
		pw = t + (R@pc)
		# print("pw : \n", pw, t.shape, R.shape, pc.shape)

		# Transform camera origin in World coordinate frame
		cam = np.array([0,0,0]).T; cam.shape = (3,1)
		cam_world =  R.T @ cam + R.T @ -t 
		# print("cam_world : \n", cam_world)

		# Find a ray from camera to 3d point
		vector = pw - cam_world
		unit_vector = vector / np.linalg.norm(vector)
		# print("unit_vector : \n", unit_vector)
		
		# Point scaled along this ray
		p3D = cam_world + scale*depth_map[p[1], p[0]] * unit_vector
		if debug:
			depth_map_copy[p[1],p[0]] = max_depth + (max_depth+min_depth)/2

		# print("p3D : \n", p3D)
		List3D.append(p3D)

	if debug:
		plt.imshow(depth_map_copy)
		plt.savefig(dataFolder + '_depth_map_points.png')
		plt.show()

	return List3D


def read_data(input_path):
    camera_path = input_path / "cameras.bin"
    database_path = input_path / "database.db"
    image_path = input_path / "images.bin"
    point3d_path =  input_path / "point3D.bin"
    cameras, images, points3D = read_model(input_path, ext='.bin')
    return cameras, images, points3D


def image_pairs_n_xyz(points3D):
    image_pairs = [p3D.image_ids for _, p3D in points3D.items()]
    tr = TransactionEncoder()
    tr_arr = tr.fit(image_pairs).transform(image_pairs)
    df = pd.DataFrame(tr_arr, columns=tr.columns_)
    
    xyzs = [p3D.xyz for _, p3D in points3D.items()]
    xyzs = np.array(xyzs)
    x, y, z = xyzs.T
    return df, x, y, z

def extract_cam_pose(input_path, output_path):
    # input_path = Path("outputs/custom1/sfm")
    input_path = Path(input_path)
    cameras, images, points3D = read_data(input_path)
    ids = list(images.keys())  # IDs of the reconstructed images
    # id_ = ids[0]  # let's look at the first image
    img_path = []
    for i in ids:
        img_path.append(images[i].name)   
    
    # image list related to each image
    df, x, y, z = image_pairs_n_xyz(points3D)
    # plt.scatter(x,z)
    # max_lim = np.max((x,z))
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.show()
    img_pair = []
    num_points, num_images =  df.shape
    for img_idx in range(num_images):
        mask = df.iloc[:, img_idx]
        masked_df = df.loc[mask, :]
        related_img = masked_df.sum()
        related_img = related_img.to_numpy()
        related_img_idx = np.where(related_img>0)
        print(f"{img_idx} image: {related_img_idx}")
        img_pair.append(related_img_idx)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ray_point = np.array([[0, 0, 0], [0, 0, 1]])
    save_path = output_path + "cam_location.csv" 
    cam_info = {"images":[],"loc":[], "front":[]}
    cam_positions = []
    for idx in ids :
        # test
        # if images[idx].name.split("/")[-1] in ["1_F.jpg", "2_F.jpg", "3_F.jpg", "4_F.jpg"]:
        # print(idx)
        # print('Quaternion:', images[idx].qvec)
        quaternion = images[idx].qvec
        translation = images[idx].tvec   
        rat_mat = quaternion_rotation_matrix(quaternion)
        # rot_mat_transposed = rat_mat.transpose()
        # abs_location = -np.matmul(rot_mat_transposed, translation)
        cam_position = -rat_mat.T @ translation
        cam_position_ = rat_mat.T @ (ray_point-translation).T
        # print(cam_position)
        # print(cam_position_.T)
        # cam_info["front"].append((cam_position_.T[1]-cam_position_.T[0]).tolist())
        # print('Euler:', quaternion_to_euler_angle_vectorized2(*quaternion))
        # print('Euler:', quaternion_rotation_matrix(images[idx].qvec))  
        # print("absolute location (x10):", cam_position*10)
        # print(cam_position[0])
        # print('Position:', images[idx].tvec)

        # ax.scatter(*cam_position_.T[0], c="red")
        # ax.scatter(*cam_position_.T[1], c="blue")
        # ax.set_xlim(-5,5)
        # ax.set_ylim(-5,5)
        # ax.set_zlim(-5,5)
        if "F" in images[idx].name:
            cam_info["images"].append(images[idx].name.split("/")[-1])  # remove path 
            cam_info["loc"].append(cam_position_.T[0].tolist())
            cam_info["front"].append(cam_position_.T[1].tolist())
            cam_positions.append([cam_position[0], cam_position[1],cam_position[2]])
            pd_cam_position = pd.DataFrame({"x":[cam_position[0]],"y":[cam_position[1]],"z":[cam_position[2]]})
            # save to csv file
            pd_cam_position.to_csv(save_path, index=False, mode="a", encoding="utf-8", header=False)
            
            ax.scatter(*cam_position_.T[0], c="red")
            ax.scatter(*cam_position_.T[1], c="blue")
            ax.set_xlim(-5,5)
            ax.set_ylim(-5,5)
            ax.set_zlim(-5,5)
            print(*cam_position_.T[0], *cam_position_.T[1])
    plt.show()

    with open('./cam_info.json','w') as f:
        json.dump(cam_info, f, ensure_ascii=False, indent=4)

    # ax.scatter(x, y, z)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)

    plt.show()
    print()
    cam_positions = np.array(cam_positions)
    return cam_positions, cam_info, img_path, img_pair
    
    
if __name__ == "__main__":
    
    input_path = Path("outputs/custom1/sfm")
    cameras, images, points3D = read_data(input_path)
    ids = list(images.keys())  # IDs of the reconstructed images

    # image list related to each image
    df, x, y, z = image_pairs_n_xyz(points3D)
    # id_ = ids[0]  # let's look at the first image
    # plt.scatter(x,z)
    # max_lim = np.max((x,z))
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.show()
    
    num_points, num_images =  df.shape
    for img_idx in range(num_images):
        mask = df.iloc[:, img_idx]
        masked_df = df.loc[mask, :]
        related_img = masked_df.sum()
        related_img = related_img.to_numpy()
        related_img_idx = np.where(related_img>0)
        print(f"{img_idx} image: {related_img_idx}")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ray_point = np.array([[0, 0, 0], [0, 0, 1]])
    save_path = "./cam_location.csv" 
    cam_info = {"images":[],"loc":[], "front":[]}
    for idx in ids :
        
        # test
        if images[idx].name.split("/")[-1] in ["1_F.jpg", "2_F.jpg", "3_F.jpg", "4_F.jpg"]:
            
            print(idx)
            # print('Quaternion:', images[idx].qvec)
            quaternion = images[idx].qvec
            translation = images[idx].tvec   
            rat_mat = quaternion_rotation_matrix(quaternion)
            # rot_mat_transposed = rat_mat.transpose()
            # abs_location = -np.matmul(rot_mat_transposed, translation)
            cam_position = -rat_mat.T @ translation
            cam_position_ = rat_mat.T @ (ray_point-translation).T

            print(cam_position)
            print(cam_position_.T)
            cam_info["images"].append(images[idx].name.split("/")[-1])  # remove path 
            cam_info["loc"].append(cam_position_.T[0].tolist())
            cam_info["front"].append(cam_position_.T[1].tolist())
            # cam_info["front"].append((cam_position_.T[1]-cam_position_.T[0]).tolist())

            # print('Euler:', quaternion_to_euler_angle_vectorized2(*quaternion))
            # print('Euler:', quaternion_rotation_matrix(images[idx].qvec))  
            print("absolute location (x10):", cam_position*10)
            print(cam_position[0])

            # print('Position:', images[idx].tvec)
            print()
            ax.scatter(*cam_position_.T[0], c="red")
            ax.scatter(*cam_position_.T[1], c="blue")
            ax.set_xlim(-5,5)
            ax.set_ylim(-5,5)
            ax.set_zlim(-5,5)

        
            pd_cam_position = pd.DataFrame({"x":[cam_position[0]],"y":[cam_position[1]],"z":[cam_position[2]]})
        
            # save to csv file
            pd_cam_position.to_csv(save_path, index=False, mode="a", encoding="utf-8", header=False)
    plt.show()



    with open('./cam_info.json','w') as f:
        json.dump(cam_info, f, ensure_ascii=False, indent=4)

    # ax.scatter(x, y, z)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)

    plt.show()
    print()