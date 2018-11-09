import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('..')

import json
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from scipy import misc
from vggunet import build_network
from lib.sutils import load_data, meanIOU, plot_segmentations

float_formatter = lambda x: '%1.3f'%x
np.set_printoptions(formatter={'float_kind':float_formatter})


def meanIOU(preds, targets, num_classes):
    """Returns the mean intersection over union.

    Notes
    -----
    num_classes is used to create a fixed-sized array, and is in the instance
    where an image might not contain the full set of output classes
    """

    if preds.ndim == 4:
        mask = np.argmax(preds, axis=1)
    else:
        mask = preds

    # Keep track of an array for each input prediction / target pair.
    # Also keep track of a "found" mask, so when we calculate the mean
    # it takes into account any instances where a class doesn't exist in image
    iou = np.zeros((num_classes,))
    found = np.zeros((num_classes,))

    # Calculate the IOU for each class
    for cl in np.unique(preds):

        intersection = (mask == cl) * (targets == cl)
        union = (mask == cl) + (targets == cl)

        iou[cl] = np.sum(intersection > 0) / (np.sum(union > 0) + 1e-8)

        if cl in targets:
            found[cl] = 1.

    return iou, found


def confusion(preds, targets):
    """Outputs a multivariable confusion matrix using predictions + targets."""

    yp = preds.reshape(-1, 1)
    yt = targets.reshape(-1, 1)

    unique = np.unique(yt)
    num_classes = len(unique)
    results = np.zeros((num_classes, num_classes + 2), dtype=np.float32)

    for i in unique:
        where = np.where(yt == i)[0]
        for j in unique:
            results[i, j] = np.sum(yp[where] == j)

    # Precision = TP / (TP + FP)
    tp = np.diag(results)
    results[:, -2] = tp / (np.sum(results[:, :num_classes], axis=0))

    # Recall = TP / (TP + FN)
    results[:, -1] = tp / (np.sum(results[:, :num_classes], axis=1))

    print '\n'
    for row in results:
        for col in row:
            if col > 1:
                print '%8d'%col,
            else:
                print '%7f'%col,
        print ''


if __name__ == '__main__':

    param_file = '../data/trained-weights.npz'
    image_path = '../data/seg-tomato-leaf-stem-images.npz'
    label_path = '../data/seg-tomato-leaf-stem-labels.npz'
    # image_path = '../data/seg-tomato-images.npz'
    # label_path = '../data/seg-tomato-labels.npz'
    class_dict_path = '../data/segmentation-label-dict.json'

    # theano symbolic tensors
    input_var = T.tensor4('x')
    target_var = T.itensor3('y')

    with open(class_dict_path) as data_file:
        class_labels = json.load(data_file)
    num_classes = len(class_labels)

    # Load data
    seed = 1234
    X_train, y_train, X_valid, y_valid = \
        load_data(image_path, label_path, seed, 10)

    softmax, net, net_crf, _ = build_network(input_var, num_classes)



    # Initialize with pretrained weights
    with np.load(param_file) as f:
        param_values = [f['arr_%d' % i] for i in xrange(len(f.files))]
    nn.set_all_param_values(softmax, param_values)

    # Get the output of the networks.
    output, output_crf = nn.get_output([net, net_crf], deterministic=True)

    # Process info for network path with & without CRF
    shape = (output.shape[2], output.shape[3])

    output = T.argmax(output, axis=1).reshape(shape)
    output_crf = T.argmax(output_crf, axis=1).reshape(shape)

    # Compile the function
    function = theano.function([input_var], [output, output_crf],
                               allow_input_downcast=True)


    # "Dense" and "CRF" post-processed output
    train_d_iou = np.zeros((num_classes,))
    train_c_iou = np.zeros_like(train_d_iou)
    valid_d_iou = np.zeros_like(train_d_iou)
    valid_c_iou = np.zeros_like(train_d_iou)
    train_found = np.zeros((num_classes,))
    valid_found = np.zeros((num_classes,))

    train_d_preds, train_c_preds = [], []
    valid_d_preds, valid_c_preds = [], []

    print "Getting train predictions..."
    for image, target in zip(X_train, y_train):

        out, crf_out = function(image[np.newaxis])
        iou, found = meanIOU(out, target, num_classes)
        train_d_iou += iou
        train_found += found

        iou, found = meanIOU(crf_out, target, num_classes)
        train_c_iou += iou

        train_d_preds.append(out[np.newaxis])
        train_c_preds.append(crf_out[np.newaxis])
    train_d_preds = np.vstack(train_d_preds)
    train_c_preds = np.vstack(train_c_preds)

    print "Getting test predictions..."
    for image, target in zip(X_valid, y_valid):

        out, crf_out = function(image[np.newaxis])
        iou, found = meanIOU(out, target, num_classes)
        valid_d_iou += iou
        valid_found += found

        iou, found = meanIOU(crf_out, target, num_classes)
        valid_c_iou += iou

        valid_d_preds.append(out[np.newaxis])
        valid_c_preds.append(crf_out[np.newaxis])
    valid_d_preds = np.vstack(valid_d_preds)
    valid_c_preds = np.vstack(valid_c_preds)

    print "Calculating iou..."
    labels = [(key, class_labels[key]) for key in class_labels]
    labels = sorted(labels, key=lambda x:x[1])

    for key, val in labels:
        print key, val
    print "  REG train IOU:\t{}".format(train_d_iou / train_found)
    print "  CRF train IOU:\t{}".format(train_c_iou / train_found)
    print "  REG valid IOU:\t{}".format(valid_d_iou / valid_found)
    print "  CRF valid IOU:\t{}".format(valid_c_iou / valid_found)


    confusion(train_d_preds, y_train)
    confusion(train_c_preds, y_train)
    confusion(valid_d_preds, y_valid)
    confusion(valid_c_preds, y_valid)

    # Read in as (rows, cols, channels), but we need (1, channels, rows, cols)
    #plot_segmentations(rgb, output, prefix='reg', save_dir='../output/sample-seg')
    #plot_segmentations(rgb, crf_output, prefix='crf', save_dir='../output/sample-seg')
