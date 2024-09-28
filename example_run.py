import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

## Setup
'''Here we define some output paths. 
We will use SuperPoint local features with the SuperGlue matcher, 
but it's easy to switch to other features like SIFT or R2D2.'''
# ################################## Here!!!  ##################################
images = Path('datasets/custom1')
outputs = Path('outputs/custom1/')
# ################################## Here!!!  ##################################

sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_inloc']
matcher_conf = match_features.confs['superglue']


## 3D mapping
references = [str(p.relative_to(images)) for p in (images / 'mapping/').iterdir()]
print(len(references), "mapping images")
plot_images([read_image(images / r) for r in references], dpi=25)

'''Then we extract features and match them across image pairs. 
Since we deal with few images, we simply match all pairs exhaustively. 
For larger scenes, we would use image retrieval, as demonstrated in the other notebooks.'''
extract_features.main(feature_conf, images, image_list=references, feature_path=features)
# pairs_from_retrieval.main(sfm_pairs, image_list=references)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)

match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)


'''The we run incremental Structure-from-Motion and display the reconstructed 3D model.'''
model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
model.export_PLY(sfm_dir / "model.ply")
fig.show()


'''We also visualize which keypoints were triangulated into the 3D model.'''
# visualization.visualize_sfm_2d(model, images, color_by='visibility', n=2)
visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)

## Localization
'''Now that we have a 3D map of the scene, we can localize any image. To demonstrate this, we download a night-time image from Wikimedia.'''
# url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Paris_-_Basilique_du_Sacr%C3%A9_Coeur%2C_Montmartre_-_panoramio.jpg"

# try other queries by uncommenting their url
# url = "https://upload.wikimedia.org/wikipedia/commons/5/59/Basilique_du_Sacr%C3%A9-C%C5%93ur_%285430392880%29.jpg"
# url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/Sacr%C3%A9_C%C5%93ur_at_night%21_%285865355326%29.jpg"


# ################################## Here!!!  ##################################
query = 'query/5.jpg'
# !mkdir -p $images/query && wget $url -O $images/$query -q
plot_images([read_image(images / query)], dpi=75)
# ################################## Here!!!  ##################################


'''Again, we extract features for the query and match them exhaustively with all mapping images that were successfully reconstructed.'''
references_registered = [model.images[i].name for i in model.reg_image_ids()]
extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references_registered)
match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True);


'''We read the EXIF data of the query to infer a rough initial estimate of camera parameters like the focal length. 
Then we estimate the absolute camera pose using PnP+RANSAC and refine the camera parameters.'''
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

camera = pycolmap.infer_camera_from_image(images / query)
ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}
localizer = QueryLocalizer(model, conf)
ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
visualization.visualize_loc_from_log(images, query, log, model)

'''We visualize the correspondences between the query images a few mapping images. We can also visualize the estimated camera pose in the 3D map.'''
pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query)
fig.show()