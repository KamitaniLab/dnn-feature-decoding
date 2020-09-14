'''DNN Feature decoding - decoders training script'''


from __future__ import print_function

from itertools import product
import os
from time import time
import warnings

import bdpy
from bdpy.dataform import Features, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.util import dump_info, makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Settings ###################################################################

# Brain data
brain_dir = './data/fmri'
subjects_list = {
    'sub-01':  'sub-01_perceptionNaturalImageTraining_VC_v2.h5',
    # 'sub-02':  'sub-02_perceptionNaturalImageTraining_VC_v2.h5',
    # 'sub-03':  'sub-03_perceptionNaturalImageTraining_VC_v2.h5',
}

label = 'image_index'

rois_list = {
    'VC':  'ROI_VC = 1',
    # 'LVC': 'ROI_LVC = 1',
    # 'HVC': 'ROI_HVC = 1',
    # 'V1':  'ROI_V1 = 1',
    # 'V2':  'ROI_V2 = 1',
    # 'V3':  'ROI_V3 = 1',
    # 'V4':  'ROI_V4 = 1',
    # 'LOC': 'ROI_LOC = 1',
    # 'FFA': 'ROI_FFA = 1',
    # 'PPA': 'ROI_PPA = 1',
}
num_voxel = {
    'VC':  500,
    'LVC': 500,
    'HVC': 500,
    'V1':  500,
    'V2':  500,
    'V3':  500,
    'V4':  500,
    'LOC': 500,
    'FFA': 500,
    'PPA': 500,
}

# Image features
features_dir = './data/features/ImageNetTraining'
network = 'caffe/VGG19'
# All layers
#features_list = [d for d in os.listdir(os.path.join(features_dir, network)) if os.path.isdir(os.path.join(features_dir, network, d))]
# Selected layers
features_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                 'fc6', 'fc7', 'fc8']

print('DNN feature')
print(os.path.join(features_dir, network))
features_list = features_list[::-1]  # Start training from deep layers

# Model parameters
alpha = 100

# Results directory
results_dir_root = './data/feature_decoders/ImageNetTraining/deeprecon_500voxel_allunits_fastl2lir_alpha100'

# Misc settings
chunk_axis = 1
# If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
# Note that Y[0] should be sample dimension.


# Main #######################################################################

analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

# Print info -----------------------------------------------------------------
print('Subjects:        %s' % subjects_list.keys())
print('ROIs:            %s' % rois_list.keys())
print('Target features: %s' % network)
print('Layers:          %s' % features_list)
print('')

# Load data ------------------------------------------------------------------
print('----------------------------------------')
print('Loading data')

data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
              for sbj, dat_file in subjects_list.items()}
data_features = Features(os.path.join(features_dir, network))

# Initialize directories -----------------------------------------------------
makedir_ifnot(results_dir_root)
makedir_ifnot(os.path.join(results_dir_root, network))
makedir_ifnot('tmp')

# Save runtime information ---------------------------------------------------
info_dir = os.path.join(results_dir_root, network)
runtime_params = {
    'learning method':          'PyFastL2LiR',
    'regularization parameter': alpha,
    'fMRI data':                [os.path.abspath(os.path.join(brain_dir, v)) for v in subjects_list.values()],
    'ROIs':                     rois_list.keys(),
    'target DNN':               network,
    'target DNN features':      os.path.abspath(os.path.join(features_dir, network)),
    'target DNN layers':        features_list,
}
dump_info(info_dir, script=__file__, parameters=runtime_params)

# Analysis loop --------------------------------------------------------------
print('----------------------------------------')
print('Analysis loop')

for feat, sbj, roi in product(features_list, subjects_list, rois_list):
    print('--------------------')
    print('Feature:    %s' % feat)
    print('Subject:    %s' % sbj)
    print('ROI:        %s' % roi)
    print('Num voxels: %d' % num_voxel[roi])

    # Setup
    # -----
    analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
    results_dir = os.path.join(results_dir_root, network, feat, sbj, roi, 'model')
    makedir_ifnot(results_dir)

    # Check whether the analysis has been done or not.
    info_file = os.path.join(results_dir, 'info.yaml')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info = yaml.load(f)
        if '_status' in info and 'computation_status' in info['_status']:
            if info['_status']['computation_status'] == 'done':
                print('%s is already done and skipped' % analysis_id)
                continue

    # Preparing data
    # --------------
    print('Preparing data')

    start_time = time()

    # Brain data
    x = data_brain[sbj].select(rois_list[roi])          # Brain data
    x_labels = data_brain[sbj].select(label).flatten()  # Label (image index)

    # Target features and image labels (file names)
    y = data_features.get_features(feat)  # Target DNN features
    y_labels = data_features.index        # Label (image index)

    print('Elapsed time (data preparation): %f' % (time() - start_time))

    # Calculate normalization parameters
    # ----------------------------------

    # Normalize X (fMRI data)
    x_mean = np.mean(x, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
    x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

    # Normalize Y (DNN features)
    y_mean = np.mean(y, axis=0)[np.newaxis, :]
    y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

    # Y index to sort Y by X (matching samples)
    # -----------------------------------------
    y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()

    # Save normalization parameters
    # -----------------------------
    print('Saving normalization parameters.')
    norm_param = {'x_mean': x_mean, 'y_mean': y_mean,
                  'x_norm': x_norm, 'y_norm': y_norm}
    save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
    for sv in save_targets:
        save_file = os.path.join(results_dir, sv + '.mat')
        if not os.path.exists(save_file):
            try:
                save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                print('Saved %s' % save_file)
            except IOError:
                warnings.warn('Failed to save %s. Possibly double running.' % save_file)

    # Preparing learning
    # ------------------
    model = FastL2LiR()
    model_param = {'alpha':  alpha,
                   'n_feat': num_voxel[roi]}

    # Distributed computation setup
    # -----------------------------
    makedir_ifnot('./tmp')
    distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Model training
    # --------------
    print('Model training')
    start_time = time()

    train = ModelTraining(model, x, y)
    train.id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
    train.model_parameters = model_param

    train.X_normalize = {'mean': x_mean,
                         'std': x_norm}
    train.Y_normalize = {'mean': y_mean,
                         'std': y_norm}
    train.Y_sort = {'index': y_index}

    train.dtype = np.float32
    train.chunk_axis = chunk_axis
    train.save_format = 'bdmodel'
    train.save_path = results_dir
    train.distcomp = distcomp

    train.run()

    print('Total elapsed time (model training): %f' % (time() - start_time))

print('%s finished.' % analysis_basename)
