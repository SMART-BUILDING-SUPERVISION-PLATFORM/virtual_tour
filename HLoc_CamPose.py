from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils import viz_3d

def recon(img_path, output_path, retrieval_type='netvlad', feature_type='superpoint_aachen', match_type='superglue'):
    # Setup
    '''In this notebook, we will run SfM reconstruction from scratch on a set of images. 
    We choose the South-Building dataset - we will download it later. 
    First, we define some paths.'''

    images = Path(img_path)
    outputs = Path(output_path)
    
    # images = Path('datasets/custom1')
    # outputs = Path('outputs/custom1/')
    
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    retrieval_conf = extract_features.confs[retrieval_type]
    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[match_type]

    # Find image pairs via image retrieval
    '''We extract global descriptors with NetVLAD and find for each image the most similar ones. For smaller dataset we can instead use exhaustive matching via hloc/pairs_from_exhaustive.py, which would find n(n-1)/2
    images pairs. '''
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

    # Extract and match local features
    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    # 3D reconstruction
    '''Run COLMAP on the features and matches.'''
    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
    
    return model

if __name__ == "__main__":
    print("main??")
    # Setup
    '''In this notebook, we will run SfM reconstruction from scratch on a set of images. 
    We choose the South-Building dataset - we will download it later. 
    First, we define some paths.'''
    images = Path('datasets/custom1')

    outputs = Path('outputs/custom1/')
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # Find image pairs via image retrieval
    '''We extract global descriptors with NetVLAD and find for each image the most similar ones. For smaller dataset we can instead use exhaustive matching via hloc/pairs_from_exhaustive.py, which would find n(n-1)/2
    images pairs. '''
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

    # Extract and match local features
    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    # 3D reconstruction
    '''Run COLMAP on the features and matches.'''
    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
    # model.export_PLY(sfm_dir / "model.ply")
    fig.show()



