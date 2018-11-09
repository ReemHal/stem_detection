import sys
sys.path.append('..')
import os
import time
import json
import glob
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn
import cv2
from scipy import misc # Reading images

from lib.image_processing import ImgProc
import lib.evaluation_metrics as metrics
from neural_networks import build_network
from lib.image_annotation_parser import read

def build_segmentation_function(param_file, num_classes=10, networkType='uunet'):

    input_var = T.tensor4('x')

    softmax, net, net_crf, _ = build_network(networkType, input_var, num_classes)

    # Initialize with pretrained weights
    with np.load(param_file) as f:
        param_values = [f['arr_%d' % i] for i in xrange(len(f.files))]
    nn.set_all_param_values(softmax, param_values)

    # Get the output of the networks.
    output, output_crf = nn.get_output([net, net_crf], deterministic=True)

    # Process info for network path with & without CRF. The output of the
    # network is stored in 'n' different channels, where n represents the
    # number of unique classes in the dataset. To make a prediction, we take
    # the most probable class using argmax
    shape = (output.shape[2], output.shape[3])

    output = T.argmax(output, axis=1).reshape(shape)
    output_crf = T.argmax(output_crf, axis=1).reshape(shape)

    # Compile the function
    return theano.function([input_var], [output, output_crf], allow_input_downcast=True)

def detect_instances(segMode, dilateFlag, crfMode, neuralnet, key, curr_class, class_labels,\
                     centroid_stats, \
                     instance_stats, \
                     dataset_json, save_dir, \
                     segementation_function, \
                     mark_centroids=False, mark_gt=False, use_crf=False,\
                     instance_found_thresh=0.5):

    dilateMode=''
    if dilateFlag==True:
        dilateMode='dilate_'

    # Read as (rows, cols, channels) but need (1, channels, rows, cols)
    #rgb_orig = misc.imread(os.path.join(IMAGE_DIR, key  + '.jpg'))
    rgb_orig = misc.imread(key  + '.jpg')
    rgb_image = rgb_orig.transpose(2, 0, 1)[np.newaxis]
    use_img = rgb_orig

    # Normalization does not improve SLIC but may help watershed
    #normalizedImg = np.zeros(rgb_orig.shape)
    #normalizedImg = cv2.normalize(rgb_orig,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    #use_img= normalizedImg
    imgProcessor =  ImgProc(use_img, key=key)

    # Read the labeled (ground truth) images, and store them in appropriate channel
    annotated = read(glob.glob(key + '_annotation*')[0])
    yt_class_objects = imgProcessor.get_main_class_objects(curr_class, annotated)

    if yt_class_objects==[]:
        instance_mask= []
    else:
        instance_mask, composite_img, stats = imgProcessor.get_instance_mask(segMode, dilateFlag, key, rgb_image, curr_class,\
                                                                         class_labels,\
                                                                         segementation_function,\
                                                                          mark_centroids, use_crf)

    if (len(np.unique(instance_mask))>0) and (instance_mask is not None):

        if mark_gt==True:
            composite_img = imgProcessor.add_groundtruth_centroids(curr_class, key, dataset_json, composite_img)

        im = key.split(os.path.sep)[-1]
        im = os.path.join(save_dir, curr_class+'_detection', 'endpoint_dist_fit_'+segMode+'_'+dilateMode+neuralnet+'_'+ crfMode+ im + '.jpg')
        misc.imsave(im, composite_img)

        cstats = metrics.validate_centroids(stats['centroid'],
                                    dataset_json[curr_class][key]['bbox'],
                                    dataset_json[curr_class][key]['centroid'])
        centroid_stats.append(cstats)

    instance_stats[curr_class], iou_mask  = metrics.meanIOU_instances(instance_mask, \
                                        yt_class_objects, key, instance_stats[curr_class], instance_found_thresh=instance_found_thresh)
    #imgProcessor.showme(iou_mask,'iou mask')

    return instance_mask, centroid_stats, instance_stats
