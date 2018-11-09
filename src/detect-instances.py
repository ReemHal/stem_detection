"""Predicts & localizes tomatoes using a semantic segmentation network.
"""
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
import matplotlib.pyplot as plt
import cv2
from scipy import misc # Reading images

from lib.image_processing import ImgProc
import lib.evaluation_metrics as metrics
from lib.image_annotation_parser import AnnotationData, read
from segmentation import build_segmentation_function, detect_instances

# exclusion_files are files that I discover along the way which have serious annotation issues.
# The list should later move to create_dataset script
exclusion_files=['../data/images/image1691',\
                 '../data/images/image1698',\
                 '../data/images/image1699',\
                 '../data/images/image1898']

save_dir = '../output/'
label_file = '../data/loc-tomato-leaf-stem-labels.json'
#label_file = '../data/loc-tomato-labels.json'
pretrained_file = '../data/trained-weights.npz'
class_dict_path = '../data/segmentation-label-dict.json'

experiment_pars={'watershed+unet':{
                                   'segMode':'watershed',
                                   'crf':False,
                                   'dilate':False
                                  },
                'watershed+CRF':{
                                   'segMode':'watershed',
                                   'crf':True,
                                   'dilate':False
                                 },
                'slic+unet':{
                               'segMode':'slic',
                               'crf':False,
                               'dilate':False
                             },
                 'slic+CRF':{
                                'segMode':'slic',
                                'crf':True,
                                'dilate':False
                            },
                 'none+unet':{
                                'segMode':'none',
                                'crf':False,
                                'dilate':False
                            },
                 'none+CRF':{
                                'segMode':'none',
                                'crf':True,
                                'dilate':False
                            },
                 'watershed+unet+dilate':{
                                    'segMode':'watershed',
                                    'crf':False,
                                    'dilate':True
                                   },
                 'watershed+CRF+dilate':{
                                    'segMode':'watershed',
                                    'crf':True,
                                    'dilate':True
                                  },
                 'slic+unet+dilate':{
                                'segMode':'slic',
                                'crf':False,
                                'dilate':True
                              },
                  'slic+CRF+dilate':{
                                 'segMode':'slic',
                                 'crf':True,
                                 'dilate':True
                             },
                  'none+unet+dilate':{
                                 'segMode':'none',
                                 'crf':False,
                                 'dilate':True
                             },
                  'none+CRF+dilate':{
                                 'segMode':'none',
                                 'crf':True,
                                 'dilate':True
                             }
                 }


def load_dataset(fname, seed=1234):

    np.random.seed(seed)

    with open(fname) as data_file:
        data = json.load(data_file)

    image_names = data['tomato'].keys() # Assuming that all images containing tomatoes will also contain the other classes
    image_names = [key for key in image_names if key not in exclusion_files]
    np.random.shuffle(image_names)
    # Split the dataset into training / testing partitions
    return data, image_names

def split_dataset(image_names, n_test=10, num_loops=10):
    np.random.shuffle(image_names)
    if num_loops>len(image_names):
        return [],[]

    test_keys=[[] for i in range(num_loops)]
    train_keys=[[] for i in range(num_loops)]
    # Split the dataset into training / testing partitions
    for j in xrange(0, num_loops):
        start=n_test*j
        end= start+n_test
        test_keys[j]=image_names[start:end]
        train_keys[j]=np.append(image_names[0:start], image_names[end+1:])
        print test_keys[j]
    return train_keys, test_keys

def detect_all_instances(dataset_json, train_keys, test_keys, class_of_interest, dilateFlag=True, \
         segMode='none', crf=False, neuralnet='unet', \
         mark_centroids=False, mark_gt=False, instance_found_thresh=0.5):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(class_dict_path) as data_file:
        class_labels = json.load(data_file)

    # Build the function to get an output from the network
    num_classes = len(class_labels)
    segementation_function = build_segmentation_function(pretrained_file, num_classes)

    instance_stats=dict.fromkeys(classes_of_interest,{})

    crfMode=''
    if crf==True:
        crfMode='crf_'

    for curr_class in classes_of_interest:
        # We'll read the dictionary keys from our JSON file, which tells us which
        # images to load from disk that belong in the training / testing set
        centroid_stats = []
        instance_stats[curr_class]=dict.fromkeys(test_keys,{})

        for key in test_keys:
            instance_masks, centroid_stats, instance_stats = detect_instances(segMode, dilateFlag,\
                                                                                    crfMode, neuralnet, \
                                                                                    key, curr_class, \
                                                                                    class_labels,\
                                                                                    centroid_stats, \
                                                                                    instance_stats,\
                                                                                    dataset_json, save_dir,\
                                                                                    segementation_function,\
                                                                                    mark_centroids,\
                                                                                    mark_gt,\
                                                                                    crf,\
                                                                                    instance_found_thresh=instance_found_thresh)

        if instance_masks==[]:
            # class was not annotated in image
            continue
        if len(centroid_stats)>0:
            centroid_stats = np.vstack(centroid_stats)
            total = np.sum(centroid_stats, axis=0)
            tp, fp, fn, num_dis, num_cont, _, _ = total

            overall_dist = np.mean(centroid_stats, axis=0)[-2]
            tp_dist = np.mean(centroid_stats, axis=0)[-1]
            prec= float(tp) / float(tp + fp)
            recall = float(tp) / float(tp + fn)
        else:
            tp, fp, fn, num_dist, num_cont, overall_dist, tp_dist, prec, recall=\
            (0, 0, 0, 0, 0, 0, 0, 0, 0)

    return instance_stats

def main(n_test, num_loops, classes_of_interest, detailed=False, instance_found_thresh=0.5):
    summ_stats=np.full((len(experiment_pars.keys()), num_loops), {})
    total_summs=dict.fromkeys(experiment_pars.keys())

    dataset_json, image_names= \
        load_dataset(label_file, seed=1234)
    print "number of images:", len(image_names)
    print "number of test files:", n_test

    train_keys, test_keys = split_dataset(image_names, n_test=n_test, num_loops=num_loops)

    for i, experiment in enumerate(experiment_pars.keys()):

        print "==================================================================="
        print "Experiment:", experiment, 'Segmentation Mode:', experiment_pars[experiment]['segMode'],\
              'dilated?',experiment_pars[experiment]['dilate'],\
              'Use CRF?', experiment_pars[experiment]['crf']

        for j in xrange(0,num_loops):
            print "run#", j
            instance_stats = detect_all_instances(dataset_json, train_keys[j], \
                 test_keys[j], classes_of_interest, dilateFlag=experiment_pars[experiment]['dilate'], \
                 segMode=experiment_pars[experiment]['segMode'], \
                 crf=experiment_pars[experiment]['crf'], instance_found_thresh=instance_found_thresh)
            summ_stats[i][j]=metrics.summarize_instance_stats(classes_of_interest, test_keys[j], \
                                            instance_stats)
            metrics.display_instance_stats(classes_of_interest, test_keys[j], \
                                            instance_stats, summ_stats[i][j], detailed=detailed)
            print "*******************************************************"
        print "==================================================================="

        total_summs[experiment]=metrics.get_stats_per_experiment(classes_of_interest,summ_stats[i])

    metrics.display_overall_exp_stats(total_summs, classes_of_interest)


    return total_summs, summ_stats

if __name__ == '__main__':
    classes_of_interest=['tomato' ,'stem','leaf']
    num_loops=10
    n_test=12
    main(n_test, num_loops, classes_of_interest, detailed=False, instance_found_thresh=0.5)
