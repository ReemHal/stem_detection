import sys
sys.path.append('..')
import os
import glob
import json
import numpy as np
import cv2

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from scipy import misc # Reading images
from lib.dutils import validate_centroids
from vggunet import build_network


def load_dataset(n_test=10, seed=1234):
    """Loads a dataset and randomly shuffles training / testing instances."""

    # Seed ensures reproducability in train / test splits
    np.random.seed(seed)

    with open('../data/loc-tomato-labels.json') as data_file:    
        data = json.load(data_file)
  
    image_names = data.keys()
    np.random.shuffle(image_names) 

    # Split the dataset into training / testing partitions
    return data, image_names[:-n_test], image_names[-n_test:] 


def build_segmentation_function(param_file, num_classes=8):

    input_var = T.tensor4('x')

    softmax, net, net_crf, _ = build_network(input_var, num_classes)

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
    return theano.function([input_var], [output, output_crf],
                            allow_input_downcast=True)


def hough_circles(image, mask, blur_size=9, min_dist=30, min_rad=15, 
                  max_rad=45, canny_thresh_high=70, accumulator_threshold=20):
    """Performs a Hough circle detection using OpenCV.

    Mask can be either the ground-truth mask, or a mask from the output of 
    the semantic segmentation algorithm.
    """

    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    # Convert (channels, rows cols) to (rows, cols, channels)
    cimg = image * mask.astype(np.uint8)

    img = cv2.cvtColor(cimg, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
                               minDist=min_dist,
                               param1=canny_thresh_high,
                               param2=accumulator_threshold,
                               minRadius=min_rad,
                               maxRadius=max_rad)

    # If circles were found, round their centroids to the nearest integers 
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))[0]

    # Only keep predictions that fall in our "tomato cluster" area / predictions
    good_circles = []
    for row in circles:
        if mask[row[1], row[0], 0] == 1:
            good_circles.append(row)

    if len(good_circles) == 0:
        return None
    return np.vstack(good_circles)


def main(dataset_json, keys, **kwargs):
    """Computes statistics against labeled data."""
    
    save_dir = '../output/detection'
    pretrained_file = '../data/trained-weights.npz'
    class_dict_path = '../data/segmentation-label-dict.json'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(class_dict_path) as data_file:
        class_labels = json.load(data_file)

    # Build the function to get an output from the network
    num_classes = len(class_labels)
    get_segmentation = build_segmentation_function(pretrained_file, num_classes)


    # Keeps track of predictions
    centroid_stats = []

    for image_key in keys:
     
        print 'Image: ', image_key

        image = misc.imread(image_key + '.jpg')
        rgb_image = image.transpose(2, 0, 1)[np.newaxis]
        
        output, crf_output = get_segmentation(rgb_image)
        
        mask = np.where(crf_output == class_labels['tomato'], 1, 0)
        mask = mask.astype(np.uint8)

        circles = hough_circles(image, mask, **kwargs)
 

        # Compute error statics. Note for testing phase we won't have this
        bbox = dataset_json[image_key]['bbox']
        centroids = dataset_json[image_key]['centroid']

        stats = validate_centroids(circles[:, :2], bbox, centroids)
        centroid_stats.append(stats)

        # Save our detections to file
        for c in dataset_json[image_key]['centroid']:
            cv2.circle(image, (c[0], c[1]), 5, (0, 255, 0), -1)
        for c in circles:
            cv2.circle(image, (c[0], c[1]), c[2], (255, 255, 255), 2)

  
        im = image_key.split(os.path.sep)[-1]
        im = os.path.join(save_dir, 'hough-' + im + '.jpg')
        misc.imsave(im, image)


    centroid_stats = np.vstack(centroid_stats)
    total = np.sum(centroid_stats, axis=0)
    tp, fp, fn, num_dis, num_cont, _, _ = total

    overall_dist = np.mean(centroid_stats, axis=0)[-2]
    tp_dist = np.mean(centroid_stats, axis=0)[-1]

    print 'tp: ', tp
    print 'fp: ', fp
    print 'fn: ', fn
    print 'precision: ', float(tp) / float(tp + fp)
    print 'recall: ', float(tp) / float(tp + fn)
    print '# predictions ignored: ', num_dis
    print '# contours with multi: ', num_cont
    print 'Distance overall: ', overall_dist
    print 'Distance to matches: ', tp_dist


    print '--------Adding predictions with multiple to FP ---------'
    fp = fp + num_dis
    print 'modified tp: ', tp
    print 'modified fp: ', fp
    print 'modified fn: ', fn
    print 'modified precision: ', float(tp) / float(tp + fp)
    print 'modified recall: ', float(tp) / float(tp + fn)



def test(seed=1234, fname='image1075.jpg'):

    params = {}
    params['min_rad'] = 15
    params['min_dist'] = 30
    params['max_rad'] = 45
    params['blur_size'] = 9
    params['canny_thresh_high'] = 70
    params['accumulator_threshold'] = 10

    pretrained_file = '../data/trained-weights.npz'
    class_dict_path = '../data/segmentation-label-dict.json'

    with open(class_dict_path) as data_file:
        class_labels = json.load(data_file)
    num_classes = len(class_labels)

    # Build the function to get an output from the network
    get_segmentation = build_segmentation_function(pretrained_file, num_classes)

    # Read as (rows, cols, channels) but need (1, channels, rows, cols)
    #rgb_orig = misc.imread(os.path.join(IMAGE_DIR, key  + '.jpg'))
    rgb_orig = misc.imread(fname)
    rgb_image = rgb_orig[np.newaxis].transpose(0, 3, 1, 2)

    # Get the regular & post-processed segmentations from the network.
    # Here, we use the post-processed output for segmenting tomatoes
    output, crf_output = get_segmentation(rgb_image)

    mask = np.where(crf_output == class_labels['tomato'], 1, 0)
    mask = mask.astype(np.uint8)

    # Run the detector
    circles = hough_circles(rgb_orig, mask, **params)


    # ------------------- PLOT DETECTIONS -----------------------------
    # First convert the tomato mask into an RGB image by tiling it across
    # the channel dimension, then linearly blend with the original RGB image.

    for c in circles:
        cv2.circle(rgb_orig, (c[0], c[1]), c[2], (255, 255, 255), 2)
    for c in circles:
        cv2.circle(rgb_orig, (c[0], c[1]), 2, (255, 255, 0), -1)

    misc.imsave('../output/hough-' + fname, rgb_orig) 


if __name__ == '__main__':

    test()


    dataset_json, train_keys, test_keys = load_dataset()

    params = {}
    params['min_rad'] = 15
    params['min_dist'] = 30
    params['max_rad'] = 45
    params['blur_size'] = 9
    params['canny_thresh_high'] = 70
    params['accumulator_threshold'] = 10

    main(dataset_json, test_keys, **params)
