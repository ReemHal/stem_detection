import os
import sys
sys.path.append('..')
import time
import json
import numpy as np
import cPickle as pickle
from datetime import datetime

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import categorical_crossentropy

from lib.sutils import (load_data, meanIOU, plot_segmentations,
                        random_crop, confusion)

from vggunet import build_network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main():

    # Where we'll save sample data to
    fname = sys.argv[0].split('.py')[0]
    curr_time = datetime.now().strftime('%d%H%M')
    save_dir = '../output/segmentation/images-' + fname + curr_time

    image_path = '../data/seg-tomato-leaf-stem-images.npz'
    label_path = '../data/seg-tomato-leaf-stem-labels.npz'
    #image_path = '../data/seg-tomato-images.npz'
    #label_path = '../data/seg-tomato-labels.npz'
    pretrained = '../data/vgg16.pkl'
    class_dict_path = '../data/segmentation-label-dict.json'

    num_epochs = 300
    lrate = 1e-3
    batch_size = 1
    seed = 1234
    crop_size = 700

    # theano symbolic tensors
    input_var = T.tensor4('x')
    target_var = T.itensor3('y')
    input_shape = (None, 3, None, None)

    # Load data.
    # x_train and x_valid contains the images;
    # y_train and y_test contains the labels for the corresponding images
    # Number of validation/test samples is hard coded to 10 samples.
    # TO DO: change this to a percentage of the total size of the data
    X_train, y_train, X_valid, y_valid = load_data(image_path, label_path, seed, 10)
    unique_classes = np.unique(y_train)
    print "class: ", unique_classes

    # Slightly increase importance of segmenting the tomato border class. Note
    # that we can do this for all other classes as well by changing the index.
    with open(class_dict_path) as data_file:
        class_labels = json.load(data_file)
    print "class labels:", class_labels

    class_weights = np.ones(len(class_labels))
    class_weights[class_labels['tomato border']] *= 1.05
    class_weights[class_labels['leaf border']] *= 1.05
    class_weights[class_labels['stem border']] *= 1.05

    # Compute class weights to balance dataset. We first find the value to get
    # an equal contribution from each class, then transform weights to [0, 1]
    # First get the frequency of each class in the training set
    print "bgd count: ", [np.sum(y_train == 5)]
    counts = [np.sum(y_train == class_) for class_ in unique_classes]
    print "counts: ", counts
    for class_name in class_labels.keys():
        print "class: ", class_labels[class_name], "class name: ", class_name, " freq: ", counts[class_labels[class_name]]
    counts = np.asarray(counts).astype(theano.config.floatX)
    # The factor used to adjust the weights for each freq in counts is (minimum_freq in counts)/freq
    counts = np.min(counts) * (1. / counts)

    # Since 'stem' is the least frequent class in the training set
    # class_weights[class_labels['stem']] *= 1.0
    counts = counts * class_weights
    counts = T.as_tensor_variable(counts)

    print 'Building model'
    softmax, network, network_crf, vgg_layers = \
        build_network(input_var, len(unique_classes))
    print 'Number of parameters: ', nn.count_params(softmax)

    # If training, initialize weights with ImageNet pretrained weights.
    # otherwise, we can load full network weights from file and set all
    # by commenting this and uncommenting subsequent lines of code
    param_values = pickle.load(file(pretrained, mode='r'))['param values']
    nn.set_all_param_values(vgg_layers, param_values[:13 * 2])

    #with np.load('../data/trained-weights.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #lasagne.layers.set_all_param_values(softmax, param_values)


    # When building the loss, we'll weight class loss by frequency due to
    # consistency of labels and their overall desireability
    output = nn.get_output(softmax, deterministic=False)
    loss = categorical_crossentropy(output, target_var.flatten())
    loss = loss * counts[target_var.flatten()]
    loss = T.mean(loss)

    # When training, we only want to update the newly added layers
    pretrained_layers = nn.get_all_layers(vgg_layers)
    layers = nn.get_all_layers(softmax, treat_as_input=pretrained_layers)
    params = [l.get_params(trainable=True) for l in layers] #[[W,b], [W,b] ... ]
    trainable_params = [p for a in params for p in a]

    updates = lasagne.updates.adamax(loss, trainable_params, lrate)
    #updates = lasagne.updates.rmsprop(loss, trainable_params, lrate)

    # Take the most likely class and compare it to the provided labels
    train_acc = T.mean(T.eq(T.argmax(output, axis=1), target_var.flatten()))

    train_fn = theano.function([input_var, target_var],
                               [loss, train_acc], updates=updates,
                               allow_input_downcast=True)



    # Validation function
    output, preds = nn.get_output([softmax, network_crf], deterministic=True)
    loss = categorical_crossentropy(output, target_var.flatten())
    loss = loss * counts[target_var.flatten()]
    loss = T.mean(loss)

    test_acc = T.mean(T.eq(T.argmax(output, axis=1), target_var.flatten()))

    valid_fn = theano.function([input_var, target_var],
                               [loss, test_acc, preds],
                                allow_input_downcast=True)


    # Early stopping
    best_params = None
    count, best_err = 0, np.inf

    # Train the network: iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0

        start_time = time.time()
        for inputs, targets in iterate_minibatches(X_train, y_train, batch_size,
                                                   shuffle=True):

            inputs, targets = random_crop(inputs, targets,
                                          X_train.shape[2],
                                          X_train.shape[3])

            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1


        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        valid_iou = np.zeros((len(unique_classes), ))
        valid_found = np.zeros((len(unique_classes), ))
        val_preds, val_inputs, val_targets = [], [], []
        for inputs, targets in iterate_minibatches(X_valid, y_valid, batch_size,
                                                   shuffle=False):

            err, acc, preds = valid_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

            val_preds.append(preds)
            val_inputs.append(inputs)
            val_targets.append(targets)

            iou, found = meanIOU(preds, targets, len(unique_classes))
            valid_iou += iou
            valid_found += found



        if epoch % 3 == 0:
            val_preds = np.vstack(val_preds)
            val_inputs = np.vstack(val_inputs) / 255.0
            val_targets = np.vstack(val_targets)
            plot_segmentations(val_inputs, val_preds, val_targets, epoch, save_dir)

            '''
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.savez(os.path.join(output_dir, 'predictions.npz'), val_preds)
            np.savez(os.path.join(output_dir, 'targets.npz'), val_targets)
            np.savez(os.path.join(output_dir, 'rgb.npz'), val_inputs)
            '''

            confusion(val_preds, val_targets)


        # Then we print the results for this epoch:
        print "Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs,
                                                   time.time() - start_time)
        print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
        print "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
        print "  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100)
        print "  validation IOU:\t\t{}".format(valid_iou / valid_found)


        # Early stopping
        val_err = val_err / float(val_batches)
        if val_err > best_err*0.99:
            count += 1
        else:
            count = 0
            best_err = val_err
            best_params = nn.get_all_param_values(softmax)
        if count >= 6:
            nn.set_all_param_values(softmax, best_params)
            break


    # And a full pass over the validation data:
    val_preds, val_inputs, val_targets = [], [], []
    for batch in iterate_minibatches(X_valid, y_valid, batch_size,
                                     shuffle=False):
        inputs, targets = batch

        err, acc, preds = valid_fn(inputs, targets)
        val_preds.append(preds)
        val_inputs.append(inputs)
        val_targets.append(targets)

    val_preds = np.vstack(val_preds)
    val_inputs = np.vstack(val_inputs) / 255.0
    val_targets = np.vstack(val_targets)

    print 'Final confusion matrix: '
    confusion(val_preds, val_targets)

    plot_segmentations(val_inputs, val_preds, val_targets, 'final', save_dir)

    np.savez('../data/trained-weights.npz',
             *lasagne.layers.get_all_param_values(softmax))



if __name__ == '__main__':
    main()
